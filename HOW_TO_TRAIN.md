# 🚀 Transfer Learning 학습 가이드

FSFM 모델을 사용한 딥페이크 탐지 전이학습을 진행하는 방법입니다.

---

## 📋 사전 준비 체크리스트

### ✅ 1. 환경 확인
```bash
# PyTorch와 CUDA 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### ✅ 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### ✅ 3. 데이터 분할 (처음 한 번만 실행)
검증 데이터가 비어있다면, 학습 데이터의 일부를 검증 데이터로 분할해야 합니다.

```bash
python scripts/split_train_val.py
```

이 스크립트는:
- 학습 데이터의 **20%**를 검증 데이터로 이동
- Real과 Fake를 각각 비율에 맞춰 분할
- 랜덤 시드를 사용하여 재현 가능한 분할

### ✅ 4. 데이터셋 테스트
```bash
python scripts/test_dataset.py
```

예상 출력:
```
[학습 데이터셋]
✓ 총 샘플 수: 12926
✓ 샘플 구조:
  - image shape: torch.Size([3, 224, 224])
  - label: 0
  - filename: real_0.jpg
  - is_video: False

[검증 데이터셋]
✓ 총 샘플 수: 3232
...
```

### ✅ 5. 모델 테스트
```bash
python scripts/test_model.py
```

---

## 🎯 학습 전략 선택

FSFM 전이학습은 3가지 전략을 지원합니다:

### 전략 1: Feature Extractor (빠른 테스트용) ⚡
**특징:**
- 사전학습된 백본을 **완전히 고정**
- 분류 헤드만 학습
- 빠른 학습 속도 (메모리 적게 사용)
- 적은 데이터셋에 적합

**사용 시기:**
- 빠르게 베이스라인 성능을 확인하고 싶을 때
- GPU 메모리가 부족할 때
- 데이터셋이 매우 작을 때 (< 1000 샘플)

**실행 방법:**
```bash
python train.py \
  --config configs/config.yaml \
  --strategy feature_extractor
```

### 전략 2: Fine-tuning (권장) 🎯
**특징:**
- 전체 모델을 **함께 학습**
- 최고의 성능
- 중간 학습 시간
- 중간 크기 이상의 데이터셋에 적합

**사용 시기:**
- **최고 성능**이 필요할 때
- 충분한 데이터가 있을 때 (> 5000 샘플)
- GPU 메모리가 충분할 때

**실행 방법:**
```bash
python train.py \
  --config configs/config.yaml \
  --strategy fine_tuning
```

### 전략 3: PEFT-LoRA (효율적 학습) 🔧
**특징:**
- LoRA 어댑터만 학습
- 메모리 효율적
- 빠른 학습 속도
- Fine-tuning과 비슷한 성능

**사용 시기:**
- 메모리를 절약하고 싶을 때
- 여러 실험을 빠르게 돌리고 싶을 때
- 다양한 하이퍼파라미터를 시도하고 싶을 때

**실행 방법:**
```bash
python train.py \
  --config configs/config.yaml \
  --strategy peft_lora
```

---

## 📊 학습 시작하기

### 기본 학습 (Fine-tuning, 권장)
```bash
python train.py \
  --config configs/config.yaml \
  --strategy fine_tuning
```

### 추가 옵션과 함께 실행
```bash
python train.py \
  --config configs/config.yaml \
  --strategy fine_tuning \
  --batch_size 16 \
  --epochs 30 \
  --lr 5e-5
```

### 모델 선택
`configs/config.yaml`에서 모델 타입을 변경할 수 있습니다:

```yaml
model:
  type: "vit_base_patch16"  # 옵션: vit_small_patch16, vit_base_patch16, vit_large_patch16
  pretrained_checkpoint: "/workspace/transfer_learning_fsfm/src/models/fsfm/checkpoint/vit_base_patch16/checkpoint-min_val_loss.pth"
  norm_file: "/workspace/transfer_learning_fsfm/src/models/fsfm/checkpoint/vit_base_patch16/pretrain_ds_mean_std.txt"
```

**모델 크기 비교:**
- `vit_small_patch16`: 작고 빠름 (~22M params)
- `vit_base_patch16`: 중간 크기, 균형 잡힘 (~86M params) ⭐ **권장**
- `vit_large_patch16`: 크고 느림, 최고 성능 (~304M params)

---

## 📈 학습 모니터링

### 터미널 출력
학습 중 실시간으로 다음 정보가 표시됩니다:
```
Epoch 1/20
Train: 100%|██████████| 404/404 [03:24<00:00,  1.97it/s, loss=0.234, acc=0.923]
Val: 100%|██████████| 101/101 [00:42<00:00,  2.38it/s, loss=0.189, acc=0.941]

Epoch 1 완료 - Train Loss: 0.2340 | Train Acc: 0.9230 | Val Loss: 0.1890 | Val Acc: 0.9410
✓ Best model saved!
```

### TensorBoard (선택사항)
```bash
# 로그 디렉토리 확인
tensorboard --logdir logs/

# 브라우저에서 열기
# http://localhost:6006
```

### 저장된 파일 확인
```bash
ls -lh checkpoints/
# best_model.pth - 최고 성능 모델
# last_model.pth - 마지막 에포크 모델
# checkpoint_epoch_5.pth - 5 에포크마다 저장
```

---

## 🛠️ 하이퍼파라미터 튜닝

### 학습률 조정
```yaml
training:
  optimizer:
    lr: 1e-4  # 기본값
    # Fine-tuning: 1e-5 ~ 1e-4
    # Feature Extractor: 1e-4 ~ 1e-3
    # PEFT-LoRA: 1e-4 ~ 3e-4
```

### 배치 크기 조정
```yaml
training:
  batch_size: 32  # GPU 메모리에 따라 조정
  # 16: 12GB GPU
  # 32: 24GB GPU
  # 64: 40GB+ GPU
```

### 데이터 증강 설정
```yaml
training:
  augmentation:
    enabled: true
    horizontal_flip: 0.5
    rotation: 10
    color_jitter:
      brightness: 0.2
      contrast: 0.2
```

---

## 🎓 학습 팁

### 1. 빠른 실험을 위한 설정
작은 에포크로 시작하여 빠르게 테스트:
```bash
python train.py \
  --config configs/config.yaml \
  --strategy feature_extractor \
  --epochs 5 \
  --batch_size 64
```

### 2. 오버피팅 방지
- Label smoothing 활성화
- Weight decay 증가
- Dropout 증가
- 데이터 증강 강화

```yaml
training:
  optimizer:
    weight_decay: 0.1  # 기본 0.05
  loss:
    label_smoothing: 0.1
```

### 3. 학습이 불안정할 때
- 학습률 감소
- Warmup 에포크 증가
- Gradient clipping 추가

```yaml
training:
  optimizer:
    lr: 5e-5  # 낮춤
  scheduler:
    warmup_epochs: 3  # 증가
```

### 4. GPU 메모리 부족 시
```yaml
training:
  batch_size: 16  # 줄이기
  mixed_precision: true  # 활성화
```

또는:
```bash
# Gradient accumulation 사용
python train.py \
  --config configs/config.yaml \
  --batch_size 8 \
  --gradient_accumulation_steps 4
```

---

## 🔍 학습 후 평가

### 베스트 모델로 추론
```bash
python inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --input_dir data/test \
  --output submission.csv
```

### 체크포인트 비교
여러 체크포인트의 성능을 비교:
```bash
for ckpt in checkpoints/*.pth; do
    echo "Evaluating $ckpt"
    python inference.py \
      --checkpoint $ckpt \
      --input_dir data/val \
      --output results_$(basename $ckpt .pth).csv
done
```

---

## ⚠️ 문제 해결

### 문제 1: "데이터를 찾을 수 없습니다"
**원인:** 데이터 경로가 잘못되었거나 폴더가 비어있음

**해결:**
```bash
# 데이터 구조 확인
ls -la data/train/
ls -la data/val/

# 검증 데이터가 비어있다면
python scripts/split_train_val.py
```

### 문제 2: CUDA Out of Memory
**해결:**
- 배치 크기 감소: `batch_size: 8` 또는 `16`
- Mixed precision 활성화: `mixed_precision: true`
- 작은 모델 사용: `vit_small_patch16`
- Num workers 감소: `num_workers: 2`

### 문제 3: 학습이 진행되지 않음 (loss가 안 떨어짐)
**원인:** 학습률이 너무 낮거나 높음

**해결:**
```bash
# 학습률 스케줄 확인
python train.py --config configs/config.yaml --strategy fine_tuning --lr 1e-4
python train.py --config configs/config.yaml --strategy fine_tuning --lr 5e-5
python train.py --config configs/config.yaml --strategy fine_tuning --lr 3e-4
```

### 문제 4: 검증 정확도가 학습 정확도보다 높음
**원인:** 데이터 증강이 강하거나 정규화가 강함

**해결:**
- 데이터 증강 약화
- Weight decay 감소
- Dropout 감소

---

## 📝 체크리스트

학습 시작 전 확인:

- [ ] 데이터가 올바른 구조로 준비되어 있는가? (`Real/`, `Fake/`)
- [ ] 검증 데이터가 있는가? (없으면 `split_train_val.py` 실행)
- [ ] `configs/config.yaml`에서 경로가 올바른가?
- [ ] 사전학습 체크포인트가 존재하는가?
- [ ] GPU 메모리가 충분한가?
- [ ] 필요한 패키지가 모두 설치되어 있는가?

---

## 🎯 권장 워크플로우

### 첫 실행 (빠른 테스트)
```bash
# 1. 데이터 분할
python scripts/split_train_val.py

# 2. 데이터셋 테스트
python scripts/test_dataset.py

# 3. Feature Extractor로 빠른 베이스라인 확인 (5 에포크)
python train.py --config configs/config.yaml --strategy feature_extractor --epochs 5

# 4. 결과 확인
ls -lh checkpoints/
```

### 본격적인 학습
```bash
# Fine-tuning으로 20 에포크 학습
python train.py --config configs/config.yaml --strategy fine_tuning --epochs 20
```

### 최적화
```bash
# 다양한 하이퍼파라미터로 실험
python train.py --config configs/config.yaml --strategy fine_tuning --lr 5e-5 --epochs 30
python train.py --config configs/config.yaml --strategy peft_lora --lr 1e-4 --epochs 25
```

---

## 📚 추가 자료

- **README.md**: 프로젝트 전체 개요
- **IMPLEMENTATION_GUIDE.md**: 코드 구현 상세 가이드
- **PROJECT_SUMMARY.md**: 프로젝트 요약 및 구조

---

**Good Luck! 🚀**

