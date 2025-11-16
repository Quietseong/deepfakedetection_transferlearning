"""
모델 테스트 스크립트

모델 생성 및 전방향 패스가 올바르게 작동하는지 확인합니다.
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.utils import ConfigLoader


def test_model():
    """모델 생성 및 테스트"""
    
    # 설정 로드
    config = ConfigLoader("configs/config.yaml")
    
    print("="*60)
    print("모델 테스트")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n디바이스: {device}")
    
    # 전이학습 전략 리스트
    strategies = ["feature_extractor", "fine_tuning"]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"전략: {strategy.upper()}")
        print(f"{'='*60}")
        
        try:
            # 모델 생성
            if strategy == "fine_tuning":
                freeze_layers = config.get("transfer_learning.fine_tuning.freeze_layers", [])
            else:
                freeze_layers = None
            
            model = create_model(
                model_type=config.get("model.type"),
                num_classes=config.get("model.num_classes", 2),
                pretrained_path=config.get("model.pretrained_checkpoint"),
                drop_path_rate=config.get("model.drop_path_rate", 0.1),
                global_pool=config.get("model.global_pool", True),
                strategy=strategy,
                freeze_layers=freeze_layers,
            )
            
            model = model.to(device)
            model.eval()
            
            # 더미 입력으로 전방향 패스 테스트
            batch_size = 4
            image_size = config.get("data.image_size", 224)
            dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
            
            print(f"\n입력 shape: {dummy_input.shape}")
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"출력 shape: {output.shape}")
            print(f"✓ 전방향 패스 성공!")
            
            # 학습 가능한 파라미터 확인
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n파라미터 통계:")
            print(f"  - 전체: {total_params:,}")
            print(f"  - 학습 가능: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            
        except Exception as e:
            print(f"✗ 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)


if __name__ == "__main__":
    test_model()


