"""
전이학습 학습 스크립트

FSFM 모델을 딥페이크 탐지 태스크에 전이학습시킵니다.
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent))
from src.data import DeepfakeDataset, collate_fn
from src.models import create_model
from src.utils import ConfigLoader, calculate_metrics, print_metrics


def set_seed(seed: int) -> None:
    """
    재현성을 위한 시드 설정
    
    Args:
        seed: 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """
    전이학습 트레이너
    
    Args:
        config: 설정 객체
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        device: 학습 디바이스
    """
    
    def __init__(
        self,
        config: ConfigLoader,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 옵티마이저 설정
        self.optimizer = self._create_optimizer()
        
        # 학습률 스케줄러 설정
        self.scheduler = self._create_scheduler()
        
        # 손실 함수 설정
        self.criterion = self._create_criterion()
        
        # Mixed Precision Training
        self.use_amp = config.get("training.mixed_precision", False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = Path(config.get("training.checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early Stopping
        self.early_stopping_patience = config.get("training.early_stopping.patience", 7)
        self.early_stopping_counter = 0
        self.best_val_f1 = 0.0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        옵티마이저 생성
        
        Returns:
            옵티마이저
        """
        optimizer_type = self.config.get("training.optimizer.type", "adamw").lower()
        lr = float(self.config.get("training.optimizer.lr", 1e-4))
        weight_decay = float(self.config.get("training.optimizer.weight_decay", 0.05))
        
        # 학습 가능한 파라미터만 전달
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_type == "adamw":
            betas = self.config.get("training.optimizer.betas", [0.9, 0.999])
            optimizer = optim.AdamW(
                trainable_params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "sgd":
            momentum = self.config.get("training.optimizer.momentum", 0.9)
            optimizer = optim.SGD(
                trainable_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_type}")
        
        print(f"\n옵티마이저: {optimizer_type.upper()}")
        print(f"  - Learning Rate: {lr}")
        print(f"  - Weight Decay: {weight_decay}")
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """
        학습률 스케줄러 생성
        
        Returns:
            학습률 스케줄러
        """
        scheduler_type = self.config.get("training.scheduler.type", "cosine").lower()
        
        if scheduler_type == "cosine":
            epochs = int(self.config.get("training.epochs", 20))
            min_lr = float(self.config.get("training.scheduler.min_lr", 1e-6))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=min_lr,
            )
            print(f"학습률 스케줄러: CosineAnnealingLR")
        
        elif scheduler_type == "step":
            step_size = int(self.config.get("training.scheduler.step_size", 5))
            gamma = float(self.config.get("training.scheduler.gamma", 0.1))
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
            print(f"학습률 스케줄러: StepLR")
        
        elif scheduler_type == "plateau":
            patience = int(self.config.get("training.scheduler.patience", 3))
            factor = float(self.config.get("training.scheduler.factor", 0.5))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=factor,
                patience=patience,
                verbose=True,
            )
            print(f"학습률 스케줄러: ReduceLROnPlateau")
        
        else:
            scheduler = None
            print("학습률 스케줄러: 없음")
        
        return scheduler
    
    def _create_criterion(self) -> nn.Module:
        """
        손실 함수 생성
        
        Returns:
            손실 함수
        """
        loss_type = self.config.get("training.loss.type", "cross_entropy").lower()
        
        if loss_type == "cross_entropy":
            label_smoothing = float(self.config.get("training.loss.label_smoothing", 0.0))
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            print(f"손실 함수: CrossEntropyLoss (label_smoothing={label_smoothing})")
        else:
            raise ValueError(f"지원하지 않는 손실 함수: {loss_type}")
        
        return criterion
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        한 에폭 학습
        
        Args:
            epoch: 현재 에폭
            
        Returns:
            학습 지표 딕셔너리
        """
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass (Mixed Precision)
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 예측 및 통계
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            running_loss += loss.item()
            
            # Progress bar 업데이트
            if batch_idx % self.config.get("logging.log_interval", 10) == 0:
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                })
        
        # 에폭 평균 손실
        avg_loss = running_loss / len(self.train_loader)
        
        # 평가 지표 계산
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            detailed=True,
        )
        metrics["loss"] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        검증 수행
        
        Args:
            epoch: 현재 에폭
            
        Returns:
            검증 지표 딕셔너리
        """
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in progress_bar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # 예측 및 통계
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            running_loss += loss.item()
            
            # Progress bar 업데이트
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 평균 손실
        avg_loss = running_loss / len(self.val_loader)
        
        # 평가 지표 계산
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            detailed=True,
        )
        metrics["loss"] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        체크포인트 저장
        
        Args:
            epoch: 현재 에폭
            metrics: 평가 지표
            is_best: 최고 성능 여부
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✓ 최고 성능 모델 저장: {best_path}")
        
        # 주기적 저장
        save_every_n = self.config.get("training.save_every_n_epochs", 5)
        if epoch % save_every_n == 0:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_path)
            print(f"  ✓ 체크포인트 저장: {epoch_path}")
    
    def train(self) -> None:
        """
        전체 학습 루프
        """
        epochs = self.config.get("training.epochs", 20)
        
        print("\n" + "="*60)
        print("학습 시작")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")
            
            # 학습
            train_metrics = self.train_epoch(epoch)
            print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f} | "
                  f"Macro F1: {train_metrics['macro_f1']:.4f} | "
                  f"F1 (Fake): {train_metrics['f1_fake']:.4f} | "
                  f"F1 (Real): {train_metrics['f1_real']:.4f}")
            
            # 검증
            val_metrics = self.validate(epoch)
            print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"Macro F1: {val_metrics['macro_f1']:.4f} ⭐ | "
                  f"F1 (Fake): {val_metrics['f1_fake']:.4f} | "
                  f"F1 (Real): {val_metrics['f1_real']:.4f}")
            
            # 학습률 스케줄러 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['macro_f1'])
                else:
                    self.scheduler.step()
            
            # 최고 성능 체크
            current_val_f1 = val_metrics['macro_f1']
            is_best = current_val_f1 > self.best_val_f1
            
            if is_best:
                self.best_val_f1 = current_val_f1
                self.early_stopping_counter = 0
                print(f"\n  🎉 새로운 최고 성능! Macro F1: {self.best_val_f1:.4f}")
            else:
                self.early_stopping_counter += 1
            
            # 체크포인트 저장
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early Stopping
            if (self.config.get("training.early_stopping.enabled", False) 
                and self.early_stopping_counter >= self.early_stopping_patience):
                print(f"\n⚠️  Early Stopping: {self.early_stopping_patience} 에폭 동안 개선 없음")
                break
        
        print("\n" + "="*60)
        print("학습 완료!")
        print(f"최고 Validation Macro F1: {self.best_val_f1:.4f}")
        print("="*60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="FSFM 전이학습")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="전이학습 전략 (feature_extractor, fine_tuning, peft_lora)",
    )
    args, _ = parser.parse_known_args()
    
    # 설정 로드
    config = ConfigLoader(args.config)
    
    # 전략 오버라이드
    if args.strategy is not None:
        config.config["transfer_learning"]["strategy"] = args.strategy
    
    # 시드 설정
    seed = config.get("project.seed", 42)
    set_seed(seed)
    print(f"시드 설정: {seed}")
    
    # 디바이스 설정
    device = torch.device(
        config.get("project.device", "cuda") 
        if torch.cuda.is_available() 
        else "cpu"
    )
    print(f"디바이스: {device}")
    
    # 데이터 로더 생성
    print("\n데이터 로드 중...")
    train_dataset = DeepfakeDataset(
        data_dir=config.get("data.train_path"),
        is_training=True,
        num_frames=config.get("data.num_frames", 10),
        image_size=config.get("data.image_size", 224),
        mean=config.get("data.mean"),
        std=config.get("data.std"),
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=config.get("data.val_path"),
        is_training=False,
        num_frames=config.get("data.num_frames", 10),
        image_size=config.get("data.image_size", 224),
        mean=config.get("data.mean"),
        std=config.get("data.std"),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("training.batch_size", 32),
        shuffle=True,
        num_workers=config.get("training.num_workers", 4),
        pin_memory=config.get("training.pin_memory", True),
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("training.batch_size", 32),
        shuffle=False,
        num_workers=config.get("training.num_workers", 4),
        pin_memory=config.get("training.pin_memory", True),
        collate_fn=collate_fn,
    )
    
    print(f"  - 학습 샘플: {len(train_dataset)}")
    print(f"  - 검증 샘플: {len(val_dataset)}")
    
    # 모델 생성
    print("\n모델 생성 중...")
    strategy = config.get("transfer_learning.strategy", "fine_tuning")
    
    if strategy == "peft_lora":
        model = create_model(
            model_type=config.get("model.type"),
            num_classes=config.get("model.num_classes", 2),
            pretrained_path=config.get("model.pretrained_checkpoint"),
            drop_path_rate=config.get("model.drop_path_rate", 0.1),
            global_pool=config.get("model.global_pool", True),
            strategy=strategy,
            lora_r=config.get("transfer_learning.peft_lora.r", 16),
            lora_alpha=config.get("transfer_learning.peft_lora.lora_alpha", 32),
            lora_dropout=config.get("transfer_learning.peft_lora.lora_dropout", 0.1),
            target_modules=config.get("transfer_learning.peft_lora.target_modules", ["qkv"]),
        )
    else:
        freeze_layers = None
        if strategy == "fine_tuning":
            freeze_layers = config.get("transfer_learning.fine_tuning.freeze_layers", [])
        
        model = create_model(
            model_type=config.get("model.type"),
            num_classes=config.get("model.num_classes", 2),
            pretrained_path=config.get("model.pretrained_checkpoint"),
            drop_path_rate=config.get("model.drop_path_rate", 0.1),
            global_pool=config.get("model.global_pool", True),
            strategy=strategy,
            freeze_layers=freeze_layers,
        )
    
    # 트레이너 생성 및 학습
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()


