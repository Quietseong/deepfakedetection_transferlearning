# 구현 완료 가이드

## ✅ 완료된 작업

전이학습 프로젝트가 성공적으로 구축되었습니다!

---

## 📦 생성된 파일 목록

### 핵심 스크립트
- ✅ `train.py` - 학습 메인 스크립트
- ✅ `inference.py` - 추론 메인 스크립트

### 소스 코드
- ✅ `src/data/dataset.py` - 데이터셋 클래스
- ✅ `src/models/transfer_model.py` - 전이학습 모델 (3가지 전략)
- ✅ `src/utils/metrics.py` - 평가 지표
- ✅ `src/utils/config_loader.py` - 설정 로더

### 설정 및 문서
- ✅ `configs/config.yaml` - 메인 설정 파일
- ✅ `requirements.txt` - 의존성 패키지
- ✅ `README.md` - 전체 프로젝트 문서
- ✅ `QUICK_START.md` - 빠른 시작 가이드
- ✅ `PROJECT_SUMMARY.md` - 프로젝트 구조 요약
- ✅ `.gitignore` - Git 무시 파일

### 테스트 및 유틸리티
- ✅ `scripts/test_dataset.py` - 데이터셋 테스트
- ✅ `scripts/test_model.py` - 모델 테스트
- ✅ `scripts/quick_start.sh` - 빠른 시작 스크립트

---

## 🎯 주요 기능

### 1. 3가지 전이학습 전략 지원

#### Feature Extractor
```python
# 백본 고정, 헤드만 학습
# 빠른 학습, 적은 메모리 사용
python train.py --strategy feature_extractor
```

#### Fine-tuning
```python
# 전체/부분 미세조정
# 최고 성능
python train.py --strategy fine_tuning
```

#### PEFT-LoRA
```python
# 파라미터 효율적 학습
# 메모리 효율적
python train.py --strategy peft_lora
```

### 2. 이미지 및 동영상 처리

- 이미지: `.jpg`, `.png` 등
- 동영상: `.mp4` 등 (프레임 추출 후 처리)
- 자동 전처리 및 정규화
- 데이터 증강 지원

### 3. Macro F1-score 기반 평가

- 대회의 주요 평가 지표
- 클래스별 F1-score 추적
- Confusion Matrix 제공

### 4. 학습 최적화

- Mixed Precision Training
- Early Stopping
- 학습률 스케줄링 (Cosine, Step, Plateau)
- 체크포인트 자동 저장

### 5. 추론 및 제출

- 배치 추론
- 동영상 프레임 집계
- submission.csv 자동 생성

---

## 🚀 다음 단계

### 1단계: 환경 설정 확인

```bash
cd /workspace/transfer_learning_fsfm

# Python 버전 확인 (3.10 이상)
python --version

# CUDA 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2단계: 패키지 설치

```bash
# CUDA 12.6 환경
pip install -U torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# 기타 패키지
pip install -r requirements.txt
```

### 3단계: 설정 파일 수정

`configs/config.yaml` 파일을 열어 다음 항목을 확인/수정하세요:

```yaml
data:
  train_path: "/workspace/ai_factory_submission/data/train"
  val_path: "/workspace/ai_factory_submission/data/val"

model:
  pretrained_checkpoint: "/workspace/ai_factory_submission/model/fsfm/checkpoint/vit_base_patch16/checkpoint-min_val_loss.pth"
```

### 4단계: 데이터 구조 확인

현재 데이터가 다음 구조로 되어 있어야 합니다:

```
data/
├── train/
│   ├── real/    # 진짜 이미지/동영상 (라벨 0)
│   └── fake/    # 가짜 이미지/동영상 (라벨 1)
└── val/
    ├── real/
    └── fake/
```

**만약 다른 구조라면:**

옵션 1: 데이터 재구성
```bash
mkdir -p /workspace/ai_factory_submission/data/train/real
mkdir -p /workspace/ai_factory_submission/data/train/fake
# ... 파일 이동
```

옵션 2: 데이터셋 코드 수정
`src/data/dataset.py`의 `_load_samples()` 메서드를 수정하여 다른 구조 지원

### 5단계: 테스트 실행

```bash
# 데이터셋 테스트
python scripts/test_dataset.py

# 모델 테스트
python scripts/test_model.py
```

### 6단계: 학습 시작

```bash
# 추천: Fine-tuning 전략으로 시작
python train.py --config configs/config.yaml --strategy fine_tuning

# 또는 빠른 테스트를 위해 Feature Extractor
python train.py --config configs/config.yaml --strategy feature_extractor
```

### 7단계: 추론 수행

```bash
python inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data_dir /workspace/ai_factory_submission/data/val \
  --output submission.csv
```

---

## ⚠️ 주의사항

### 1. 데이터 구조

현재 구현은 다음 폴더 구조를 기대합니다:
- `train/real/` - 진짜 데이터
- `train/fake/` - 가짜 데이터

만약 데이터가 다른 형식이라면:
1. 데이터를 재구성하거나
2. `src/data/dataset.py`의 `_load_samples()` 메서드를 수정

### 2. 메모리 관리

GPU 메모리 부족 시:
```yaml
# configs/config.yaml
training:
  batch_size: 16  # 또는 8
  mixed_precision: true

# 또는 PEFT-LoRA 사용
transfer_learning:
  strategy: "peft_lora"
```

### 3. 학습 시간

- Feature Extractor: 빠름 (~1-2시간, 데이터셋 크기에 따라)
- Fine-tuning: 보통 (~3-5시간)
- PEFT-LoRA: 보통 (~2-4시간)

### 4. 체크포인트

최고 성능 모델은 자동으로 `checkpoints/best_model.pth`에 저장됩니다.
주기적 체크포인트는 `checkpoint_epoch_N.pth` 형식으로 저장됩니다.

---

## 🐛 트러블슈팅

### 문제 1: "데이터를 찾을 수 없습니다"

**해결:**
1. `configs/config.yaml`에서 경로 확인
2. 데이터 폴더 구조 확인 (real/, fake/ 필요)
3. 절대 경로 사용 권장

### 문제 2: "CUDA out of memory"

**해결:**
1. 배치 크기 줄이기: `batch_size: 16` 또는 `8`
2. Mixed Precision 활성화: `mixed_precision: true`
3. PEFT-LoRA 전략 사용
4. 동영상 프레임 수 줄이기: `num_frames: 5`

### 문제 3: 모델 로드 실패

**해결:**
1. 체크포인트 경로 확인
2. 전략이 학습 시와 동일한지 확인
3. 모델 타입 확인

### 문제 4: Import 오류

**해결:**
```bash
# timm 버전 확인
pip install timm==0.4.5

# 경로 문제 시
export PYTHONPATH=/workspace/transfer_learning_fsfm:$PYTHONPATH
```

---

## 📈 성능 최적화 팁

### 1. 하이퍼파라미터 튜닝

```yaml
# 학습률 실험
training:
  optimizer:
    lr: 1e-4  # 또는 5e-4, 1e-3

# 배치 크기 조정
training:
  batch_size: 32  # 또는 64 (GPU 메모리 충분 시)

# 에폭 수 증가
training:
  epochs: 30  # 또는 40
```

### 2. 데이터 증강

```yaml
training:
  augmentation:
    enabled: true
    horizontal_flip: 0.5
    rotation: 15
    color_jitter:
      brightness: 0.3
      contrast: 0.3
```

### 3. 동영상 프레임 수

```yaml
# 성능 우선
data:
  num_frames: 16  # 또는 32

# 속도 우선
data:
  num_frames: 5  # 또는 8
```

### 4. 전략 비교

모든 전략을 실험해보고 최적의 전략을 선택하세요:
1. Feature Extractor (베이스라인)
2. Fine-tuning (최고 성능)
3. PEFT-LoRA (메모리 효율적)

---

## 🎓 대회 제출 통합

### task.ipynb 통합 방법

학습된 모델을 대회 제출용 `task.ipynb`에 통합하는 방법:

#### 1. 체크포인트 복사

```bash
# 학습된 모델을 제출 폴더로 복사
cp checkpoints/best_model.pth \
   /workspace/ai_factory_submission/model/fsfm/checkpoint/vit_base_patch16/finetuned_model.pth
```

#### 2. task.ipynb 수정

기존 추론 코드를 전이학습 모델로 교체:

```python
# 모델 로드 부분
from src.models import create_model

model = create_model(
    model_type="vit_base_patch16",
    num_classes=2,
    pretrained_path=None,  # 체크포인트에서 로드
    strategy="fine_tuning",  # 학습 시 사용한 전략
)

# 체크포인트 로드
checkpoint = torch.load("./model/fsfm/checkpoint/vit_base_patch16/finetuned_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

#### 3. 추론 코드 적용

`inference.py`의 추론 로직을 `task.ipynb`에 통합

---

## 📚 참고 문서

- **README.md**: 전체 프로젝트 문서 (상세)
- **QUICK_START.md**: 빠른 시작 가이드 (실용적)
- **PROJECT_SUMMARY.md**: 프로젝트 구조 요약 (개괄)
- **transfer_guide.txt**: 원본 전이학습 가이드 (배경)

---

## ✨ 주요 특징 요약

### 코드 품질
- ✅ Google 스타일 docstring
- ✅ Type hints 사용
- ✅ 한글 주석 (중학생 수준 설명)
- ✅ 모듈화된 구조
- ✅ 에러 처리

### 기능
- ✅ 3가지 전이학습 전략
- ✅ 이미지/동영상 처리
- ✅ Macro F1-score 평가
- ✅ Mixed Precision Training
- ✅ Early Stopping
- ✅ 자동 체크포인트 저장
- ✅ submission.csv 생성

### 문서화
- ✅ 상세한 README
- ✅ 빠른 시작 가이드
- ✅ 프로젝트 구조 요약
- ✅ 테스트 스크립트
- ✅ 설정 파일 주석

---

## 🎉 완료!

전이학습 프로젝트가 준비되었습니다!

다음 단계:
1. ✅ 코드 구현 완료
2. ⬜ 환경 설정
3. ⬜ 데이터 준비
4. ⬜ 학습 시작
5. ⬜ 추론 수행
6. ⬜ 대회 제출

**시작하기:**
```bash
cd /workspace/transfer_learning_fsfm
bash scripts/quick_start.sh
```

궁금한 점이 있으면 README.md와 QUICK_START.md를 참조하세요!


