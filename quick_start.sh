#!/bin/bash
# Transfer Learning 빠른 시작 스크립트

echo "=========================================="
echo "FSFM Transfer Learning - 빠른 시작"
echo "=========================================="

# 1. 데이터 분할
echo ""
echo "1️⃣  검증 데이터 준비 중..."
python scripts/split_train_val.py << ANSWER
y
ANSWER

# 2. 데이터 확인
echo ""
echo "2️⃣  데이터셋 테스트 중..."
python scripts/test_dataset.py

# 3. 빠른 테스트 학습 (5 에포크)
echo ""
echo "3️⃣  빠른 테스트 학습 시작 (Feature Extractor, 5 에포크)..."
python train.py --config configs/config.yaml --strategy feature_extractor --epochs 5

echo ""
echo "=========================================="
echo "✓ 빠른 테스트 완료!"
echo "=========================================="
echo ""
echo "본격적인 학습을 시작하려면:"
echo "  python train.py --config configs/config.yaml --strategy fine_tuning --epochs 20"
