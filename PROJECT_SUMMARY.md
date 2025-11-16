# 프로젝트 구조 요약

## 📁 디렉토리 구조

```
transfer_learning_fsfm/
├── configs/                     # 설정 파일
│   └── config.yaml              # 메인 설정 파일
│
├── src/                         # 소스 코드
│   ├── data/                    # 데이터 모듈
│   │   ├── __init__.py
│   │   └── dataset.py           # 데이터셋 클래스 (이미지/동영상 처리)
│   │
│   ├── models/                  # 모델 모듈
│   │   ├── __init__.py
│   │   ├── transfer_model.py    # 전이학습 모델 (3가지 전략)
│   │   └── fsfm/                # FSFM 원본 모델
│   │       ├── models_vit.py
│   │       └── checkpoint/      # 사전학습 체크포인트
│   │
│   └── utils/                   # 유틸리티 모듈
│       ├── __init__.py
│       ├── metrics.py           # 평가 지표 (Macro F1-score)
│       └── config_loader.py     # 설정 로더
│
├── scripts/                     # 테스트 및 유틸리티 스크립트
│   ├── quick_start.sh           # 빠른 시작 스크립트
│   ├── test_dataset.py          # 데이터셋 테스트
│   └── test_model.py            # 모델 테스트
│
├── checkpoints/                 # 학습 체크포인트 저장 폴더
│   └── best_model.pth           # 최고 성능 모델
│
├── logs/                        # 학습 로그
│
├── train.py                     # 학습 메인 스크립트
├── inference.py                 # 추론 메인 스크립트
│
├── requirements.txt             # 의존성 패키지
├── README.md                    # 전체 프로젝트 문서
├── QUICK_START.md               # 빠른 시작 가이드
├── PROJECT_SUMMARY.md           # 이 문서
└── .gitignore                   # Git 무시 파일
```

---

## 🎯 주요 컴포넌트

### 1. 데이터 처리 (`src/data/dataset.py`)

**DeepfakeDataset**
- 이미지 및 동영상 데이터 로드
- 자동 전처리 및 증강
- 동영상에서 균등하게 프레임 추출
- Real/Fake 폴더 구조 지원

**InferenceDataset**
- 레이블 없는 추론 전용 데이터셋
- DeepfakeDataset 기반

**collate_fn**
- 배치 구성 시 이미지/동영상 처리

### 2. 모델 (`src/models/transfer_model.py`)

**FSFMTransferModel**
- 3가지 전이학습 전략 지원:
  1. **Feature Extractor**: 백본 고정, 헤드만 학습
  2. **Fine-tuning**: 전체/부분 미세조정
  3. **PEFT-LoRA**: 파라미터 효율적 학습

**FSFMWithLoRA**
- LoRA 기반 전이학습
- 메모리 효율적

**create_model**
- 모델 생성 팩토리 함수

### 3. 평가 지표 (`src/utils/metrics.py`)

- `calculate_macro_f1()`: Macro F1-score 계산
- `calculate_binary_f1_scores()`: 클래스별 F1-score
- `calculate_metrics()`: 전체 지표 계산
- `print_metrics()`: 지표 출력
- `print_classification_report()`: sklearn 리포트

### 4. 학습 (`train.py`)

**Trainer 클래스**
- 학습 루프 관리
- 검증 수행
- 체크포인트 저장
- Early Stopping
- Mixed Precision Training
- 학습률 스케줄링

**주요 기능:**
- 옵티마이저: AdamW, SGD
- 스케줄러: Cosine, Step, Plateau
- 손실 함수: CrossEntropyLoss (Label Smoothing)
- 데이터 증강

### 5. 추론 (`inference.py`)

**Inferencer 클래스**
- 체크포인트 로드
- 배치 예측
- 동영상 프레임 집계
- submission.csv 생성

---

## ⚙️ 설정 파일 (`configs/config.yaml`)

### 주요 섹션

```yaml
# 프로젝트 기본 설정
project:
  name: "fsfm_deepfake_transfer_learning"
  seed: 42
  device: "cuda"

# 데이터 경로 및 전처리
data:
  train_path: "/path/to/train"
  val_path: "/path/to/val"
  inference_path: "/path/to/test"
  image_size: 224
  num_frames: 10
  mean: [...]  # FSFM 정규화 파라미터
  std: [...]

# 모델 설정
model:
  type: "vit_base_patch16"
  num_classes: 2
  pretrained_checkpoint: "/path/to/checkpoint.pth"
  norm_file: "/path/to/pretrain_ds_mean_std.txt"

# 전이학습 전략
transfer_learning:
  strategy: "fine_tuning"  # feature_extractor, fine_tuning, peft_lora
  
  # 각 전략별 설정
  feature_extractor: {...}
  fine_tuning: {...}
  peft_lora: {...}

# 학습 설정
training:
  epochs: 20
  batch_size: 32
  optimizer: {...}
  scheduler: {...}
  loss: {...}
  augmentation: {...}
  mixed_precision: true
  early_stopping: {...}

# 추론 설정
inference:
  batch_size: 16
  num_frames: 10
  video_aggregation: "mean"
  output_path: "./submission.csv"

# 로깅 설정
logging:
  log_dir: "./logs"
  log_interval: 10
  tensorboard: true
```

---

## 🔄 워크플로우

### 학습 워크플로우

```
1. 설정 로드 (config.yaml)
   ↓
2. 시드 설정 (재현성)
   ↓
3. 데이터셋 생성
   - DeepfakeDataset (train, val)
   - DataLoader 생성
   ↓
4. 모델 생성
   - 전략에 따라 FSFMTransferModel 또는 FSFMWithLoRA
   - 사전학습 가중치 로드
   - 전이학습 전략 적용
   ↓
5. Trainer 생성
   - 옵티마이저 설정
   - 스케줄러 설정
   - 손실 함수 설정
   ↓
6. 학습 루프
   - Epoch 반복:
     * 학습 (train_epoch)
     * 검증 (validate)
     * 평가 지표 계산
     * 최고 성능 체크
     * 체크포인트 저장
     * Early Stopping 체크
   ↓
7. 최고 성능 모델 저장
   - checkpoints/best_model.pth
```

### 추론 워크플로우

```
1. 설정 로드 (config.yaml)
   ↓
2. 데이터셋 생성
   - InferenceDataset
   - DataLoader 생성
   ↓
3. 모델 생성
   - 학습 시와 동일한 전략 사용
   ↓
4. Inferencer 생성
   - 체크포인트 로드 (best_model.pth)
   - 평가 모드 설정
   ↓
5. 추론 수행
   - 배치별로 예측
   - 동영상: 프레임별 예측 후 집계
   - 이미지: 직접 예측
   ↓
6. 결과 저장
   - submission.csv 생성
   - (filename, label) 형식
```

---

## 🚀 사용 예시

### 1. 빠른 테스트 (Feature Extractor)

```bash
# 10 에폭, 작은 배치 크기
python train.py --config configs/config.yaml --strategy feature_extractor

# 추론
python inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data_dir /path/to/test \
  --output submission.csv
```

### 2. 전체 미세조정 (Fine-tuning)

```bash
# 20 에폭, 전체 레이어 학습
python train.py --config configs/config.yaml --strategy fine_tuning

# 추론
python inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data_dir /path/to/test \
  --output submission.csv
```

### 3. 메모리 효율적 학습 (PEFT-LoRA)

```bash
# peft 설치
pip install peft transformers accelerate

# 학습
python train.py --config configs/config.yaml --strategy peft_lora

# 추론
python inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data_dir /path/to/test \
  --output submission.csv
```

---

## 📊 평가 지표 해석

### Macro F1-score (대회 주요 지표)

```
Macro F1 = (F1_Real + F1_Fake) / 2
```

- **범위**: 0.0 ~ 1.0
- **높을수록 좋음**
- Real과 Fake 클래스의 균형 있는 성능 측정

### F1-score (클래스별)

```
F1 = 2 * TP / (2 * TP + FP + FN)
```

- **F1 (Fake)**: Fake 클래스 탐지 성능
- **F1 (Real)**: Real 클래스 탐지 성능

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   TN      FP
       Fake   FN      TP
```

---

## 🎓 전이학습 전략 비교

| 전략 | 학습 속도 | 메모리 사용 | 성능 | 적합한 상황 |
|------|----------|------------|------|------------|
| Feature Extractor | ⭐⭐⭐ 빠름 | ⭐⭐⭐ 적음 | ⭐⭐ 보통 | 작은 데이터셋, 빠른 실험 |
| Fine-tuning | ⭐⭐ 보통 | ⭐⭐ 보통 | ⭐⭐⭐ 높음 | 충분한 데이터셋, 최고 성능 |
| PEFT-LoRA | ⭐⭐ 보통 | ⭐⭐⭐ 적음 | ⭐⭐⭐ 높음 | GPU 메모리 부족, 대규모 모델 |

---

## 📝 핵심 파일

### 1. `train.py`
- **역할**: 학습 메인 스크립트
- **주요 클래스**: `Trainer`
- **입력**: config.yaml
- **출력**: checkpoints/best_model.pth

### 2. `inference.py`
- **역할**: 추론 메인 스크립트
- **주요 클래스**: `Inferencer`
- **입력**: config.yaml, best_model.pth, 테스트 데이터
- **출력**: submission.csv

### 3. `src/data/dataset.py`
- **역할**: 데이터 로딩 및 전처리
- **주요 클래스**: `DeepfakeDataset`, `InferenceDataset`
- **기능**: 이미지/동영상 처리, 증강

### 4. `src/models/transfer_model.py`
- **역할**: 전이학습 모델 정의
- **주요 클래스**: `FSFMTransferModel`, `FSFMWithLoRA`
- **기능**: 3가지 전이학습 전략

### 5. `src/utils/metrics.py`
- **역할**: 평가 지표 계산
- **주요 함수**: `calculate_macro_f1`, `calculate_metrics`

### 6. `configs/config.yaml`
- **역할**: 전체 프로젝트 설정
- **내용**: 데이터, 모델, 학습, 추론 설정

---

## 🔧 커스터마이징 포인트

### 1. 데이터 증강 조정
`configs/config.yaml` → `training.augmentation`

### 2. 학습률 및 옵티마이저
`configs/config.yaml` → `training.optimizer`

### 3. 전이학습 전략
`configs/config.yaml` → `transfer_learning.strategy`

### 4. 모델 타입 변경
`configs/config.yaml` → `model.type`
- vit_small_patch16
- vit_base_patch16
- vit_large_patch16

### 5. 동영상 프레임 수
`configs/config.yaml` → `data.num_frames`

---

## 📚 참고 자료

- **README.md**: 전체 문서
- **QUICK_START.md**: 빠른 시작 가이드
- **transfer_guide.txt**: 원본 전이학습 가이드
- **configs/config.yaml**: 설정 파일 상세 설명

---

## ✅ 체크리스트

### 학습 전
- [ ] 데이터 경로 확인 (`configs/config.yaml`)
- [ ] 사전학습 체크포인트 경로 확인
- [ ] 데이터 구조 확인 (real/, fake/ 폴더)
- [ ] GPU 사용 가능 여부 확인
- [ ] 패키지 설치 완료

### 학습 중
- [ ] 학습 로그 모니터링
- [ ] Macro F1-score 추이 확인
- [ ] 과적합/과소적합 체크
- [ ] 체크포인트 저장 확인

### 추론 전
- [ ] 최고 성능 모델 확인 (best_model.pth)
- [ ] 테스트 데이터 경로 확인
- [ ] 출력 파일명 확인 (submission.csv)

### 제출 전
- [ ] submission.csv 형식 확인
- [ ] 모든 파일에 대한 예측 완료
- [ ] 추론 시간 확인 (3시간 이내)
- [ ] task.ipynb 통합 (대회 제출 시)

---

이 문서는 프로젝트 구조와 사용법을 간략하게 요약합니다.
자세한 내용은 README.md와 QUICK_START.md를 참조하세요.


