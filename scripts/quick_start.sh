#!/bin/bash

# 빠른 시작 스크립트
# FSFM 전이학습 프로젝트를 빠르게 시작하기 위한 스크립트입니다.

echo "=========================================="
echo "FSFM 전이학습 빠른 시작"
echo "=========================================="

# 1. 환경 확인
echo ""
echo "[1/5] 환경 확인 중..."
python --version
echo "CUDA 사용 가능 여부: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 2. 패키지 설치
echo ""
echo "[2/5] 패키지 설치 확인 중..."
echo "필요한 패키지가 설치되어 있지 않다면 다음 명령을 실행하세요:"
echo "  pip install -r requirements.txt"

# 3. 데이터셋 테스트
echo ""
echo "[3/5] 데이터셋 테스트 중..."
python scripts/test_dataset.py

# 4. 모델 테스트
echo ""
echo "[4/5] 모델 테스트 중..."
python scripts/test_model.py

# 5. 학습 가이드
echo ""
echo "[5/5] 학습 준비 완료!"
echo ""
echo "=========================================="
echo "다음 단계:"
echo "=========================================="
echo ""
echo "1. configs/config.yaml 파일에서 설정 확인/수정"
echo ""
echo "2. 학습 시작:"
echo "   # Feature Extractor 전략"
echo "   python train.py --config configs/config.yaml --strategy feature_extractor"
echo ""
echo "   # Fine-tuning 전략"
echo "   python train.py --config configs/config.yaml --strategy fine_tuning"
echo ""
echo "   # PEFT-LoRA 전략 (peft 설치 필요)"
echo "   python train.py --config configs/config.yaml --strategy peft_lora"
echo ""
echo "3. 추론 수행:"
echo "   python inference.py \\"
echo "     --config configs/config.yaml \\"
echo "     --checkpoint checkpoints/best_model.pth \\"
echo "     --data_dir /path/to/test/data \\"
echo "     --output submission.csv"
echo ""
echo "=========================================="
echo "자세한 내용은 README.md를 참조하세요."
echo "=========================================="


