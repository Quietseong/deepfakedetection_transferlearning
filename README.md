# FSFM 전이학습 프로젝트

딥페이크 탐지를 위한 FSFM(Few-Shot Face Manipulation) 모델 전이학습 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 사전학습된 FSFM 모델을 활용하여 이미지 및 동영상의 딥페이크 여부를 판별하는 분류 모델을 개발합니다.

### 주요 특징
- ✅ **다양한 전이학습 전략 지원**
  - Feature Extractor: 백본 고정, 헤드만 학습
  - Fine-tuning: 전체/부분 미세조정
  - PEFT-LoRA: 파라미터 효율적 미세조정
- ✅ **이미지 및 동영상 처리**
- ✅ **Macro F1-score 기반 평가**
- ✅ **Mixed Precision Training 지원**
- ✅ **Early Stopping 및 체크포인트 관리**

---

## 🗂️ 프로젝트 구조

```
transfer_learning_fsfm/
├── configs/
│   └── config.yaml              # 설정 파일
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # 데이터셋 클래스
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transfer_model.py    # 전이학습 모델
│   │   └── fsfm/                # FSFM 모델 (복사됨)
│   │       ├── models_vit.py
│   │       └── checkpoint/
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # 평가 지표
│       └── config_loader.py     # 설정 로더
├── checkpoints/                 # 학습 체크포인트
├── logs/                        # 학습 로그
├── train.py                     # 학습 스크립트
├── inference.py                 # 추론 스크립트
├── requirements.txt             # 의존성 패키지
└── README.md                    # 이 문서
```

---

## 🚀 시작하기

### 1. 환경 설정

#### Python 버전
- Python 3.10 이상 권장

#### 패키지 설치

```bash
# CUDA 12.6 환경
pip install -U torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# 기타 패키지
pip install -r requirements.txt
```

#### PEFT-LoRA 사용 시 (선택사항)
```bash
pip install peft transformers accelerate
```

### 2. 데이터 준비

데이터는 다음 구조로 구성되어야 합니다:

```
data/
├── train/
│   ├── real/       # 진짜 이미지/동영상 (라벨 0)
│   │   ├── real_001.jpg
│   │   ├── real_002.mp4
│   │   └── ...
│   └── fake/       # 가짜 이미지/동영상 (라벨 1)
│       ├── fake_001.jpg
│       ├── fake_002.mp4
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/           # 추론용 (라벨 없음)
    ├── sample_001.jpg
    ├── sample_002.mp4
    └── ...
```

### 3. 설정 파일 수정

`configs/config.yaml` 파일을 열어 다음 항목을 수정하세요:

```yaml
# 데이터 경로
data:
  train_path: "/path/to/your/data/train"
  val_path: "/path/to/your/data/val"
  inference_path: "/path/to/your/data/test"

# 모델 설정
model:
  pretrained_checkpoint: "/path/to/fsfm/checkpoint/vit_base_patch16/checkpoint-min_val_loss.pth"

# 전이학습 전략 선택
transfer_learning:
  strategy: "fine_tuning"  # feature_extractor, fine_tuning, peft_lora
```

---

## 🎯 전이학습 전략

### 1. Feature Extractor

백본을 고정하고 분류 헤드만 학습합니다.

**장점:**
- 빠른 학습
- 적은 메모리 사용
- 작은 데이터셋에 적합

**사용법:**
```yaml
# configs/config.yaml
transfer_learning:
  strategy: "feature_extractor"
```

### 2. Fine-tuning

전체 또는 일부 레이어를 미세조정합니다.

**장점:**
- 높은 성능
- 유연한 조정

**사용법:**
```yaml
# 전체 미세조정
transfer_learning:
  strategy: "fine_tuning"
  fine_tuning:
    freeze_layers: []  # 비어있으면 전체 학습

# 부분 미세조정 (초기 4개 블록 고정)
transfer_learning:
  strategy: "fine_tuning"
  fine_tuning:
    freeze_layers: ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
```

### 3. PEFT-LoRA

LoRA 어댑터만 학습하여 메모리 효율적으로 미세조정합니다.

**장점:**
- 메모리 효율적
- 대규모 모델에 적합
- Fine-tuning에 준하는 성능

**사용법:**
```yaml
transfer_learning:
  strategy: "peft_lora"
  peft_lora:
    r: 16                      # LoRA rank
    lora_alpha: 32             # Scaling factor
    lora_dropout: 0.1
    target_modules: ["qkv"]    # Attention QKV에 적용
```

---

## 🏋️ 학습

### 기본 학습

```bash
python train.py --config configs/config.yaml
```

### 전략 지정 학습

```bash
# Feature Extractor
python train.py --config configs/config.yaml --strategy feature_extractor

# Fine-tuning
python train.py --config configs/config.yaml --strategy fine_tuning

# PEFT-LoRA
python train.py --config configs/config.yaml --strategy peft_lora
```

### 학습 모니터링

학습 중 다음 정보가 실시간으로 출력됩니다:
- Loss
- Accuracy
- Macro F1-score ⭐ (대회 주요 지표)
- F1-score (Real / Fake)

최고 성능 모델은 `checkpoints/best_model.pth`에 자동 저장됩니다.

---

## 🔮 추론

### 기본 추론

```bash
python inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data_dir /path/to/test/data \
  --output submission.csv
```

### 출력 형식

`submission.csv`:
```csv
filename,label
sample_001.jpg,0
sample_002.mp4,1
sample_003.jpg,1
...
```

- `filename`: 파일명 (확장자 포함)
- `label`: 예측 결과 (Real: 0, Fake: 1)

---

## 📊 평가 지표

### Macro F1-score

대회의 주요 평가 지표로, 각 클래스(Real, Fake)의 F1-score 평균입니다.

```
F1_Real = 2 * TP_Real / (2 * TP_Real + FP_Real + FN_Real)
F1_Fake = 2 * TP_Fake / (2 * TP_Fake + FP_Fake + FN_Fake)

Macro F1 = (F1_Real + F1_Fake) / 2
```

### 클래스 정의
- **Positive (양성)**: Fake (라벨 1)
- **Negative (음성)**: Real (라벨 0)

---

## ⚙️ 주요 하이퍼파라미터

### 학습 설정

```yaml
training:
  epochs: 20                # 학습 에폭
  batch_size: 32            # 배치 크기
  
  # 옵티마이저
  optimizer:
    type: "adamw"
    lr: 1e-4                # 학습률 (전이학습용 낮은 값)
    weight_decay: 0.05
  
  # 학습률 스케줄러
  scheduler:
    type: "cosine"          # cosine, step, plateau
    min_lr: 1e-6
  
  # 손실 함수
  loss:
    type: "cross_entropy"
    label_smoothing: 0.1
  
  # Mixed Precision
  mixed_precision: true
  
  # Early Stopping
  early_stopping:
    enabled: true
    patience: 7
```

### 데이터 증강

```yaml
training:
  augmentation:
    enabled: true
    horizontal_flip: 0.5
    rotation: 10
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    random_erasing: 0.1
```

---

## 🎓 성능 향상 팁

### 1. 전이학습 전략 선택
- **데이터가 적을 때**: Feature Extractor
- **데이터가 충분할 때**: Fine-tuning
- **GPU 메모리 부족 시**: PEFT-LoRA

### 2. 학습률 조정
- Feature Extractor: `1e-3` ~ `5e-3`
- Fine-tuning: `1e-4` ~ `5e-4`
- PEFT-LoRA: `1e-4` ~ `1e-3`

### 3. 데이터 증강
- 학습 데이터가 적을 때 증강 강도 높이기
- 과적합 발생 시 증강 활성화

### 4. 배치 크기
- GPU 메모리에 따라 조정 (32, 64, 128 등)
- 작은 배치 사용 시 learning rate도 낮추기

### 5. 동영상 프레임 수
- 성능: 많을수록 좋음 (10~32 프레임)
- 속도: 적을수록 빠름 (5~10 프레임)

---

## 🐛 문제 해결

### GPU 메모리 부족
```yaml
# 배치 크기 줄이기
training:
  batch_size: 16  # 32 -> 16

# PEFT-LoRA 사용
transfer_learning:
  strategy: "peft_lora"

# Mixed Precision 활성화
training:
  mixed_precision: true
```

### 과적합 (Overfitting)
```yaml
# 데이터 증강 강화
training:
  augmentation:
    enabled: true
    horizontal_flip: 0.5
    random_erasing: 0.2

# Label Smoothing
training:
  loss:
    label_smoothing: 0.1

# Weight Decay 증가
training:
  optimizer:
    weight_decay: 0.1
```

### 과소적합 (Underfitting)
```yaml
# 학습률 증가
training:
  optimizer:
    lr: 5e-4  # 1e-4 -> 5e-4

# 에폭 수 증가
training:
  epochs: 30

# 더 많은 레이어 학습
transfer_learning:
  strategy: "fine_tuning"
  fine_tuning:
    freeze_layers: []  # 전체 미세조정
```

---

## 📝 라이선스

이 프로젝트는 FSFM 모델을 기반으로 하며, 원본 코드는 Attribution-NonCommercial 4.0 International License를 따릅니다.

---

## 🙏 감사의 말

- FSFM 모델: [원본 레포지토리 링크]
- timm 라이브러리: https://github.com/rwightman/pytorch-image-models

---

## 📧 문의

문제가 발생하거나 질문이 있으시면 이슈를 등록해주세요.


