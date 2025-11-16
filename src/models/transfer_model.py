"""
전이학습 모델 구현 모듈

FSFM 모델을 기반으로 전이학습:
1. Feature Extractor: 백본 고정, 분류 헤드만 학습
2. Fine-tuning: 전체 또는 일부 레이어 미세조정
3. PEFT-LoRA: 파라미터 효율적 미세조정
"""

from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

import torch
import torch.nn as nn

# FSFM 모델 import
sys.path.append(str(Path(__file__).parent / "fsfm"))
import models_vit


class FSFMTransferModel(nn.Module):
    """
    FSFM 기반 전이학습 모델
    
    사전학습된 FSFM 모델을 로드하고, 선택된 전이학습 전략을 적용합니다.
    
    Args:
        model_type: 모델 타입 (vit_small_patch16, vit_base_patch16, vit_large_patch16)
        num_classes: 분류 클래스 수
        pretrained_path: 사전학습 체크포인트 경로
        drop_path_rate: Drop path rate
        global_pool: Global pooling 사용 여부
        strategy: 전이학습 전략 (feature_extractor, fine_tuning)
        freeze_layers: 고정할 레이어 리스트 (fine_tuning 전략 사용 시)
    """
    
    def __init__(
        self,
        model_type: str = "vit_base_patch16",
        num_classes: int = 2,
        pretrained_path: Optional[str] = None,
        drop_path_rate: float = 0.1,
        global_pool: str = "token",
        strategy: str = "fine_tuning",
        freeze_layers: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.strategy = strategy
        self.freeze_layers = freeze_layers or []
        
        # FSFM 모델 생성
        self.model = self._create_model(
            model_type=model_type,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )
        
        # 사전학습 가중치 로드
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path, num_classes)
        
        # 전이학습 전략 적용
        self._apply_transfer_strategy()
    
    def _create_model(
        self,
        model_type: str,
        num_classes: int,
        drop_path_rate: float,
        global_pool: str,
    ) -> nn.Module:
        """
        FSFM 모델 생성
        
        Args:
            model_type: 모델 타입
            num_classes: 분류 클래스 수
            drop_path_rate: Drop path rate
            global_pool: Global pooling 타입 ("token", "avg", "" 등)
            
        Returns:
            FSFM 모델
        """
        if model_type not in models_vit.__dict__:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        model = models_vit.__dict__[model_type](
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )
        
        return model
    
    def _load_pretrained_weights(self, checkpoint_path: str, num_classes: int) -> None:
        """
        사전학습된 가중치 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            num_classes: 현재 모델의 클래스 수
        """
        print(f"사전학습 가중치 로드 중: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # 체크포인트 구조 확인
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # head 크기 확인
            if "head.weight" in state_dict:
                pretrain_num_classes = state_dict["head.weight"].shape[0]
                print(f"  사전학습 모델 클래스 수: {pretrain_num_classes}")
                print(f"  현재 모델 클래스 수: {num_classes}")
                
                # 클래스 수가 다르면 head를 제외하고 로드
                if pretrain_num_classes != num_classes:
                    print("  클래스 수가 다름. head 가중치 제외하고 로드")
                    state_dict = {k: v for k, v in state_dict.items() 
                                 if not k.startswith("head.")}
            
            # 가중치 로드
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )
            
            if missing_keys:
                print(f"  누락된 키 (정상): {missing_keys}")
            
            print("  ✓ 사전학습 가중치 로드 완료")
        
        except Exception as e:
            raise RuntimeError(f"가중치 로드 실패: {e}")
    
    def _replace_head(self, num_classes: int) -> None:
        """
        분류 헤드를 새로운 클래스 수로 교체
        
        Args:
            num_classes: 새로운 클래스 수
        """
        print(f"\n분류 헤드 교체 중: {self.model.head.in_features} -> {num_classes} 클래스")
        
        # 기존 헤드의 입력 차원 가져오기
        in_features = self.model.head.in_features
        
        # 새로운 분류 헤드 생성
        self.model.head = nn.Linear(in_features, num_classes)
        
        # 가중치 초기화
        nn.init.trunc_normal_(self.model.head.weight, std=0.02)
        if self.model.head.bias is not None:
            nn.init.constant_(self.model.head.bias, 0)
        
        print("  ✓ 분류 헤드 교체 완료")
    
    def _apply_transfer_strategy(self) -> None:
        """
        선택된 전이학습 전략 적용
        """
        if self.strategy == "feature_extractor":
            self._apply_feature_extractor()
        elif self.strategy == "fine_tuning":
            self._apply_fine_tuning()
        else:
            raise ValueError(f"지원하지 않는 전이학습 전략: {self.strategy}")
    
    def _apply_feature_extractor(self) -> None:
        """
        Feature Extractor 전략 적용
        
        백본을 고정하고 분류 헤드만 학습합니다.
        """
        print("\n전이학습 전략: Feature Extractor")
        print("  - 백본 고정, 분류 헤드만 학습")
        
        # 모든 파라미터 고정
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        # 분류 헤드만 학습 가능하도록 설정
        if hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = True
            print("  ✓ 분류 헤드(head) 학습 가능")
        
        # fc_norm도 학습 가능하게 설정 (global_pool 사용 시)
        if hasattr(self.model, "fc_norm"):
            for param in self.model.fc_norm.parameters():
                param.requires_grad = True
            print("  ✓ fc_norm 학습 가능")
        
        self._print_trainable_params()
    
    def _apply_fine_tuning(self) -> None:
        """
        Fine-tuning 전략 적용
        
        전체 또는 일부 레이어를 미세조정합니다.
        """
        print("\n전이학습 전략: Fine-tuning")
        
        if len(self.freeze_layers) == 0:
            # 전체 미세조정
            print("  - 전체 레이어 미세조정")
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            # 부분 미세조정: 지정된 레이어만 고정
            print(f"  - 일부 레이어 고정: {self.freeze_layers}")
            
            # 먼저 모든 파라미터를 학습 가능하게 설정
            for param in self.model.parameters():
                param.requires_grad = True
            
            # 지정된 레이어만 고정
            for name, param in self.model.named_parameters():
                for freeze_layer in self.freeze_layers:
                    if name.startswith(freeze_layer):
                        param.requires_grad = False
                        break
        
        self._print_trainable_params()
    
    def _print_trainable_params(self) -> None:
        """
        학습 가능한 파라미터 수 출력
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params
        
        print(f"\n파라미터 통계:")
        print(f"  - 전체: {total_params:,}")
        print(f"  - 학습 가능: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  - 고정: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 이미지 텐서 [B, 3, H, W]
            
        Returns:
            로짓 텐서 [B, num_classes]
        """
        return self.model(x)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        학습 가능한 파라미터 리스트 반환
        
        Returns:
            학습 가능한 파라미터 리스트
        """
        return [p for p in self.model.parameters() if p.requires_grad]


class FSFMWithLoRA(nn.Module):
    """
    PEFT-LoRA를 적용한 FSFM 모델
    
    LoRA(Low-Rank Adaptation)를 사용하여 파라미터 효율적으로 미세조정합니다.
    
    Note:
        이 클래스를 사용하려면 peft 라이브러리가 설치되어 있어야 합니다:
        pip install peft
    
    Args:
        model_type: 모델 타입
        num_classes: 분류 클래스 수
        pretrained_path: 사전학습 체크포인트 경로
        drop_path_rate: Drop path rate
        global_pool: Global pooling 사용 여부
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout
        target_modules: LoRA를 적용할 모듈 이름 리스트
    """
    
    def __init__(
        self,
        model_type: str = "vit_base_patch16",
        num_classes: int = 2,
        pretrained_path: Optional[str] = None,
        drop_path_rate: float = 0.1,
        global_pool: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT-LoRA를 사용하려면 peft 라이브러리가 필요합니다.\n"
                "설치: pip install peft"
            )
        
        # 베이스 모델 생성
        base_model = FSFMTransferModel(
            model_type=model_type,
            num_classes=num_classes,
            pretrained_path=pretrained_path,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
            strategy="fine_tuning",  # LoRA 적용 전 전체 활성화
        )
        
        # LoRA 설정
        target_modules = target_modules or ["qkv"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=["head"],  # 분류 헤드는 완전히 학습
        )
        
        print("\n전이학습 전략: PEFT-LoRA")
        print(f"  - LoRA rank: {lora_r}")
        print(f"  - LoRA alpha: {lora_alpha}")
        print(f"  - LoRA dropout: {lora_dropout}")
        print(f"  - Target modules: {target_modules}")
        
        # LoRA 적용
        self.model = get_peft_model(base_model.model, lora_config)
        self.model.print_trainable_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 이미지 텐서 [B, 3, H, W]
            
        Returns:
            로짓 텐서 [B, num_classes]
        """
        return self.model(x)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        학습 가능한 파라미터 리스트 반환
        
        Returns:
            학습 가능한 파라미터 리스트
        """
        return [p for p in self.model.parameters() if p.requires_grad]


def create_model(
    model_type: str,
    num_classes: int,
    pretrained_path: Optional[str] = None,
    strategy: str = "fine_tuning",
    **kwargs
) -> nn.Module:
    """
    전이학습 모델 생성 팩토리 함수
    
    Args:
        model_type: 모델 타입
        num_classes: 분류 클래스 수
        pretrained_path: 사전학습 체크포인트 경로
        strategy: 전이학습 전략 (feature_extractor, fine_tuning, peft_lora)
        **kwargs: 추가 인자
        
    Returns:
        전이학습 모델
    """
    if strategy == "peft_lora":
        model = FSFMWithLoRA(
            model_type=model_type,
            num_classes=num_classes,
            pretrained_path=pretrained_path,
            **kwargs
        )
    else:
        model = FSFMTransferModel(
            model_type=model_type,
            num_classes=num_classes,
            pretrained_path=pretrained_path,
            strategy=strategy,
            **kwargs
        )
    
    return model


