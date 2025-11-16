"""
학습 데이터를 학습/검증 세트로 분할하는 스크립트

학습 데이터의 일부(기본 20%)를 검증 데이터로 분리합니다.
"""

import shutil
from pathlib import Path
from typing import List, Tuple
import random


def split_dataset(
    train_dir: Path,
    val_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[int, int]:
    """
    학습 데이터를 학습/검증으로 분할
    
    Args:
        train_dir: 학습 데이터 디렉토리
        val_dir: 검증 데이터 디렉토리
        val_ratio: 검증 데이터 비율 (0~1)
        seed: 랜덤 시드
        
    Returns:
        (학습 샘플 수, 검증 샘플 수) 튜플
    """
    random.seed(seed)
    
    # 폴더 생성
    val_dir.mkdir(parents=True, exist_ok=True)
    
    total_moved = 0
    total_remaining = 0
    
    # Real과 Fake 각각 처리
    for class_name in ["Real", "Fake"]:
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        if not train_class_dir.exists():
            print(f"⚠️  경고: {train_class_dir} 폴더를 찾을 수 없습니다.")
            continue
        
        # 검증 폴더 생성
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # 모든 파일 목록 가져오기
        all_files = [
            f for f in train_class_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.mp4', '.avi']
        ]
        
        # 검증용 파일 수 계산
        num_val = int(len(all_files) * val_ratio)
        
        # 랜덤하게 검증용 파일 선택
        val_files = random.sample(all_files, num_val)
        
        # 파일 이동
        moved = 0
        for file_path in val_files:
            dest_path = val_class_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))
            moved += 1
        
        remaining = len(all_files) - moved
        total_moved += moved
        total_remaining += remaining
        
        print(f"✓ {class_name}: {moved}개 파일을 검증 세트로 이동 (학습 세트에 {remaining}개 남음)")
    
    return total_remaining, total_moved


def main():
    """메인 함수"""
    # 경로 설정
    project_root = Path(__file__).parent.parent
    train_dir = project_root / "data" / "train"
    val_dir = project_root / "data" / "val"
    
    print("="*60)
    print("학습/검증 데이터 분할")
    print("="*60)
    print(f"\n학습 데이터 경로: {train_dir}")
    print(f"검증 데이터 경로: {val_dir}")
    
    # 확인 메시지
    print(f"\n⚠️  학습 데이터의 20%를 검증 데이터로 이동합니다.")
    response = input("계속하시겠습니까? (y/n): ")
    
    if response.lower() != 'y':
        print("취소되었습니다.")
        return
    
    # 분할 수행
    print("\n데이터 분할 중...")
    num_train, num_val = split_dataset(train_dir, val_dir, val_ratio=0.2, seed=42)
    
    print("\n" + "="*60)
    print("✓ 분할 완료!")
    print("="*60)
    print(f"학습 샘플: {num_train}개")
    print(f"검증 샘플: {num_val}개")
    print(f"검증 비율: {num_val/(num_train+num_val)*100:.1f}%")


if __name__ == "__main__":
    main()

