"""
추론 스크립트

학습된 모델을 사용하여 테스트 데이터에 대한 예측을 수행하고
submission.csv 파일을 생성합니다.
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent))
from src.data import InferenceDataset
from src.models import create_model
from src.utils import ConfigLoader


class Inferencer:
    """
    추론 수행 클래스
    
    Args:
        config: 설정 객체
        model: 추론 모델
        checkpoint_path: 체크포인트 경로
        device: 추론 디바이스
    """
    
    def __init__(
        self,
        config: ConfigLoader,
        model: torch.nn.Module,
        checkpoint_path: str,
        device: torch.device,
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # 체크포인트 로드
        self._load_checkpoint(checkpoint_path)
        
        # 평가 모드
        self.model.eval()
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
        """
        print(f"체크포인트 로드 중: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # 모델 가중치 로드
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        # 성능 정보 출력
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            print(f"  ✓ 로드된 모델 성능:")
            print(f"    - Macro F1: {metrics.get('macro_f1', 'N/A')}")
            print(f"    - Accuracy: {metrics.get('accuracy', 'N/A')}")
        
        print("  ✓ 체크포인트 로드 완료")
    
    @torch.no_grad()
    def predict_single(self, image: torch.Tensor) -> Tuple[int, float]:
        """
        단일 이미지 예측
        
        Args:
            image: 입력 이미지 텐서 [1, 3, H, W] 또는 [N, 3, H, W]
            
        Returns:
            (예측 레이블, Fake 확률) 튜플
        """
        # 동영상인 경우 (여러 프레임)
        if image.dim() == 4 and image.size(0) > 1:
            # 각 프레임에 대해 예측 수행
            frame_probs = []
            
            for frame in image:
                frame_input = frame.unsqueeze(0).to(self.device)  # [1, 3, H, W]
                output = self.model(frame_input)
                prob = F.softmax(output, dim=1)[0, 1].item()  # Fake 확률
                frame_probs.append(prob)
            
            # 평균 확률로 최종 예측
            avg_prob = np.mean(frame_probs)
            prediction = 1 if avg_prob > 0.5 else 0
            
            return prediction, avg_prob
        
        # 이미지인 경우
        else:
            if image.dim() == 3:
                image = image.unsqueeze(0)  # [1, 3, H, W]
            
            image = image.to(self.device)
            output = self.model(image)
            
            prob = F.softmax(output, dim=1)
            prediction = torch.argmax(prob, dim=1).item()
            fake_prob = prob[0, 1].item()
            
            return prediction, fake_prob
    
    @torch.no_grad()
    def predict_batch(self, data_loader: DataLoader) -> List[Tuple[str, int, float]]:
        """
        배치 예측
        
        Args:
            data_loader: 데이터 로더
            
        Returns:
            (파일명, 예측 레이블, Fake 확률) 튜플의 리스트
        """
        results = []
        
        for batch in tqdm(data_loader, desc="추론 중"):
            images = batch["image"]  # List of tensors
            filenames = batch["filename"]
            is_videos = batch["is_video"]
            
            # 각 샘플에 대해 예측
            for img, filename, is_video in zip(images, filenames, is_videos):
                try:
                    # 동영상인 경우 프레임 처리
                    if is_video:
                        prediction, fake_prob = self.predict_single(img)
                    else:
                        # 이미지인 경우
                        prediction, fake_prob = self.predict_single(img)
                    
                    results.append((filename, prediction, fake_prob))
                
                except Exception as e:
                    print(f"  ⚠️ 예측 실패 ({filename}): {e}")
                    # 실패 시 Real(0)로 기본 처리
                    results.append((filename, 0, 0.0))
        
        return results
    
    def save_submission(
        self, 
        results: List[Tuple[str, int, float]], 
        output_path: str
    ) -> None:
        """
        submission.csv 파일 저장
        
        Args:
            results: (파일명, 예측 레이블, Fake 확률) 튜플의 리스트
            output_path: 출력 파일 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 작성
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            
            for filename, label, _ in results:
                writer.writerow([filename, label])
        
        print(f"\n✓ 제출 파일 저장: {output_path}")
        
        # 통계 출력
        total = len(results)
        fake_count = sum(1 for _, label, _ in results if label == 1)
        real_count = total - fake_count
        
        print(f"\n예측 결과 통계:")
        print(f"  - 총 샘플: {total}")
        print(f"  - Real (0): {real_count} ({real_count/total*100:.1f}%)")
        print(f"  - Fake (1): {fake_count} ({fake_count/total*100:.1f}%)")


def custom_collate_fn(batch):
    """
    추론용 커스텀 collate 함수
    
    배치를 개별 샘플로 유지합니다.
    """
    images = [sample["image"] for sample in batch]
    filenames = [sample["filename"] for sample in batch]
    is_videos = [sample["is_video"] for sample in batch]
    
    return {
        "image": images,
        "filename": filenames,
        "is_video": is_videos,
    }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="FSFM 추론")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="체크포인트 경로",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="추론 데이터 디렉토리 (설정 파일보다 우선)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="출력 파일 경로",
    )
    args, _ = parser.parse_known_args()
    
    # 설정 로드
    config = ConfigLoader(args.config)
    
    # 디바이스 설정
    device = torch.device(
        config.get("project.device", "cuda") 
        if torch.cuda.is_available() 
        else "cpu"
    )
    print(f"디바이스: {device}")
    
    # 데이터 디렉토리
    data_dir = args.data_dir or config.get("data.inference_path")
    print(f"\n추론 데이터 경로: {data_dir}")
    
    # 데이터셋 생성
    print("\n데이터 로드 중...")
    dataset = InferenceDataset(
        data_dir=data_dir,
        num_frames=config.get("data.num_frames", 10),
        image_size=config.get("data.image_size", 224),
        mean=config.get("data.mean"),
        std=config.get("data.std"),
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config.get("inference.batch_size", 16),
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )
    
    print(f"  - 추론 샘플: {len(dataset)}")
    
    # 모델 생성
    print("\n모델 생성 중...")
    strategy = config.get("transfer_learning.strategy", "fine_tuning")
    
    if strategy == "peft_lora":
        model = create_model(
            model_type=config.get("model.type"),
            num_classes=config.get("model.num_classes", 2),
            pretrained_path=None,  # 체크포인트에서 로드
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
            pretrained_path=None,  # 체크포인트에서 로드
            drop_path_rate=config.get("model.drop_path_rate", 0.1),
            global_pool=config.get("model.global_pool", True),
            strategy=strategy,
            freeze_layers=freeze_layers,
        )
    
    # 추론 수행
    inferencer = Inferencer(
        config=config,
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    
    print("\n추론 시작...")
    results = inferencer.predict_batch(data_loader)
    
    # 결과 저장
    inferencer.save_submission(results, args.output)
    
    print("\n✅ 추론 완료!")


if __name__ == "__main__":
    main()


