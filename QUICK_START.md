# 빠른 시작 가이드

FSFM 전이학습 프로젝트를 빠르게 시작하는 방법입니다.

## 📋 사전 준비

### 1. 데이터 구조 확인

현재 원본 데이터 위치:
```
/workspace/ai_factory_submission/data/
├── train/  # 학습 데이터
└── val/    # 검증 데이터
```

### 2. 모델 체크포인트 확인

현재 사전학습 모델 위치:
```
/workspace/ai_factory_submission/model/fsfm/checkpoint/
├── vit_base_patch16/
│   ├── checkpoint-min_val_loss.pth
│   └── pretrain_ds_mean_std.txt
└── vit_large_patch16/
    ├── checkpoint-min_val_loss.pth
    └── pretrain_ds_mean_std.txt
```

---

## 🚀 시작하기

### Step 1: 설정 파일 수정

`configs/config.yaml` 파일을 열어 데이터 경로가 올바른지 확인하세요:

```yaml
data:
  train_path: "/workspace/transfer_learning_fsfm/data/train"
  val_path: "/workspace/transfer_learning_fsfm/data/val"
  inference_path: "/workspace/transfer_learning_fsfm/data/inf"

model:
  type: "vit_base_patch16"
  pretrained_checkpoint: "/workspace/transfer_learning_fsfm/src/models/fsfm/checkpoint/vit_base_patch16/checkpoint-min_val_loss.pth"
  norm_file: "/workspace/transfer_learning_fsfm/src/models/fsfm/checkpoint/vit_base_patch16/pretrain_ds_mean_std.txt"
```

### Step 2: 데이터 구조 변경 (필요한 경우)

현재 데이터가 `real/` 및 `fake/` 폴더로 구분되어 있지 않다면, 다음과 같이 구성해야 합니다:

```bash
# 예시: 데이터 재구성
mkdir -p /workspace/transfer_learning_fsfm/data/train/real
mkdir -p /workspace/transfer_learning_fsfm/data/train/fake
mkdir -p /workspace/transfer_learning_fsfm/data/val/real
mkdir -p /workspace/transfer_learning_fsfm/data/val/fake

# 레이블에 따라 파일 이동
# (실제 레이블 정보가 있는 경우)
```

또는 데이터셋 코드를 수정하여 다른 구조를 지원하도록 변경할 수 있습니다.

### Step 3: 환경 테스트

```bash
# 프로젝트 루트로 이동
cd /workspace/transfer_learning_fsfm

# 데이터셋 테스트
python scripts/test_dataset.py

# 모델 테스트
python scripts/test_model.py
```

### Step 4: 학습 시작

#### 옵션 1: Feature Extractor (권장 - 빠른 테스트용)

```bash
python train.py \
  --config configs/config.yaml \
  --strategy feature_extractor
```

**장점:**
- 빠른 학습 속도
- 적은 메모리 사용
- 작은 데이터셋에 적합

#### 옵션 2: Fine-tuning (권장 - 최고 성능)

```bash
python train.py \
  --config configs/config.yaml \
  --strategy fine_tuning
```

**장점:**
- 최고 성능
- 전체 모델 최적화

#### 옵션 3: PEFT-LoRA (메모리 부족 시)

```bash
# peft 설치 필요
pip install peft transformers accelerate

python train.py \
  --config configs/config.yaml \
  --strategy peft_lora
```

**장점:**
- 메모리 효율적
- GPU 메모리 부족 시 유용

### Step 5: 추론 수행

```bash
python inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data_dir /workspace/ai_factory_submission/data/val \
  --output submission.csv
```

---

## ⚙️ 주요 설정 조정

### GPU 메모리 부족 시

`configs/config.yaml`에서:

```yaml
training:
  batch_size: 16  # 32 -> 16으로 줄이기
  mixed_precision: true  # Mixed Precision 활성화
```

또는 PEFT-LoRA 전략 사용:

```yaml
transfer_learning:
  strategy: "peft_lora"
```

### 학습 속도 향상

```yaml
training:
  batch_size: 64  # 배치 크기 증가 (GPU 메모리 충분 시)
  num_workers: 8  # 데이터 로딩 워커 증가

data:
  num_frames: 5  # 동영상 프레임 수 감소 (10 -> 5)
```

### 과적합 방지

```yaml
training:
  augmentation:
    enabled: true
    horizontal_flip: 0.5
    random_erasing: 0.2
  
  loss:
    label_smoothing: 0.1
  
  optimizer:
    weight_decay: 0.1
  
  early_stopping:
    enabled: true
    patience: 5
```

---

## 📊 학습 모니터링

학습 중 다음 정보가 출력됩니다:

```
Epoch 1/20
[Train] Loss: 0.4532 | Acc: 0.8234 | Macro F1: 0.8156 | F1 (Fake): 0.8423 | F1 (Real): 0.7889
[Val]   Loss: 0.3891 | Acc: 0.8567 | Macro F1: 0.8489 ⭐ | F1 (Fake): 0.8623 | F1 (Real): 0.8355

  🎉 새로운 최고 성능! Macro F1: 0.8489
  ✓ 최고 성능 모델 저장: checkpoints/best_model.pth
```

**주요 지표:**
- **Macro F1**: 대회의 주요 평가 지표 (높을수록 좋음)
- **F1 (Fake)**: Fake 클래스의 F1-score
- **F1 (Real)**: Real 클래스의 F1-score
- **Accuracy**: 전체 정확도

---

## 🐛 문제 해결

### 1. "데이터를 찾을 수 없습니다" 오류

**원인:** 데이터 경로가 잘못되었거나 데이터 구조가 올바르지 않음

**해결:**
1. `configs/config.yaml`에서 경로 확인
2. 데이터 폴더 구조 확인 (real/, fake/ 폴더 필요)

### 2. "CUDA out of memory" 오류

**원인:** GPU 메모리 부족

**해결:**
1. 배치 크기 줄이기: `batch_size: 16` 또는 `8`
2. Mixed Precision 활성화: `mixed_precision: true`
3. PEFT-LoRA 전략 사용
4. 동영상 프레임 수 줄이기: `num_frames: 5`

### 3. 모델 로드 실패

**원인:** 체크포인트 경로 또는 형식 문제

**해결:**
1. 체크포인트 경로 확인
2. 전략이 학습 시와 동일한지 확인
3. 모델 타입 확인 (vit_base_patch16 등)

### 4. 학습이 너무 느림

**해결:**
1. 배치 크기 증가 (GPU 메모리 허용 시)
2. `num_workers` 증가
3. 동영상 프레임 수 줄이기
4. Feature Extractor 전략 사용

---

## 📈 성능 향상 팁

### 1. 데이터 증강 조정

```yaml
training:
  augmentation:
    enabled: true
    horizontal_flip: 0.5    # 확률 조정
    rotation: 15            # 회전 각도 증가
    color_jitter:
      brightness: 0.3       # 밝기 변화 증가
      contrast: 0.3
```

### 2. 학습률 실험

```yaml
training:
  optimizer:
    lr: 5e-4  # 1e-4, 5e-4, 1e-3 등 실험
```

### 3. 스케줄러 변경

```yaml
training:
  scheduler:
    type: "plateau"  # cosine -> plateau
    patience: 3
    factor: 0.5
```

### 4. 에폭 수 증가

```yaml
training:
  epochs: 30  # 20 -> 30
```

---

## 📝 다음 단계

1. ✅ 기본 학습 완료
2. ⬜ 하이퍼파라미터 튜닝
3. ⬜ 다양한 전략 비교
4. ⬜ 앙상블 고려 (규칙 허용 시)
5. ⬜ 대회 제출 준비

---

## 💡 대회 제출 시 주의사항

### task.ipynb 통합

학습된 모델을 대회에 제출하려면 `task.ipynb`에 통합해야 합니다:

1. 학습된 체크포인트를 `./model/fsfm/checkpoint/` 경로에 복사
2. `task.ipynb`의 모델 로드 부분 수정
3. 추론 코드 적용

### 추론 시간 제한

- **최대 3시간** 제한
- 동영상 프레임 수 조정으로 속도 최적화
- 배치 크기 증가로 처리 속도 향상

### 모델 크기 제한

- 제출 파일 크기 확인
- 불필요한 파일 제거
- `.gitignore` 참조

---

## 📞 도움말

더 자세한 정보는 다음 문서를 참조하세요:
- **README.md**: 전체 프로젝트 문서
- **configs/config.yaml**: 설정 파일 상세 설명
- **transfer_guide.txt**: 원본 가이드

문제가 있다면 이슈를 등록해주세요!


