"""
데이터셋 테스트 스크립트

데이터 로딩 및 전처리가 올바르게 작동하는지 확인합니다.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data import DeepfakeDataset
from src.utils import ConfigLoader


def test_dataset():
    """데이터셋 로드 및 샘플 확인"""
    
    # 설정 로드
    config = ConfigLoader("configs/config.yaml")
    
    print("="*60)
    print("데이터셋 테스트")
    print("="*60)
    
    # 학습 데이터셋
    print("\n[학습 데이터셋]")
    try:
        train_dataset = DeepfakeDataset(
            data_dir=config.get("data.train_path"),
            is_training=True,
            num_frames=config.get("data.num_frames", 10),
            image_size=config.get("data.image_size", 224),
            mean=config.get("data.mean"),
            std=config.get("data.std"),
        )
        print(f"✓ 총 샘플 수: {len(train_dataset)}")
        
        # 첫 번째 샘플 확인
        sample = train_dataset[0]
        print(f"✓ 샘플 구조:")
        print(f"  - image shape: {sample['image'].shape}")
        print(f"  - label: {sample['label']}")
        print(f"  - filename: {sample['filename']}")
        print(f"  - is_video: {sample['is_video']}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 검증 데이터셋
    print("\n[검증 데이터셋]")
    try:
        val_dataset = DeepfakeDataset(
            data_dir=config.get("data.val_path"),
            is_training=False,
            num_frames=config.get("data.num_frames", 10),
            image_size=config.get("data.image_size", 224),
            mean=config.get("data.mean"),
            std=config.get("data.std"),
        )
        print(f"✓ 총 샘플 수: {len(val_dataset)}")
        
        # 첫 번째 샘플 확인
        sample = val_dataset[0]
        print(f"✓ 샘플 구조:")
        print(f"  - image shape: {sample['image'].shape}")
        print(f"  - label: {sample['label']}")
        print(f"  - filename: {sample['filename']}")
        print(f"  - is_video: {sample['is_video']}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)


if __name__ == "__main__":
    test_dataset()


