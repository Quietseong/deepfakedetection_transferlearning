"""
평가 지표 계산 모듈

Macro F1-score를 포함한 다양한 평가 지표를 제공합니다.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def calculate_macro_f1(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    Macro F1-score 계산
    
    각 클래스별 F1-score의 평균을 계산합니다.
    
    Args:
        y_true: 실제 레이블 [N]
        y_pred: 예측 레이블 [N]
        
    Returns:
        Macro F1-score
    """
    return f1_score(y_true, y_pred, average="macro")


def calculate_binary_f1_scores(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    이진 분류의 각 클래스별 F1-score 계산
    
    Args:
        y_true: 실제 레이블 [N]
        y_pred: 예측 레이블 [N]
        
    Returns:
        (F1_positive, F1_negative) 튜플
        - F1_positive: Fake (라벨 1)의 F1-score
        - F1_negative: Real (라벨 0)의 F1-score
    """
    f1_scores = f1_score(y_true, y_pred, average=None)
    
    if len(f1_scores) == 2:
        f1_negative = f1_scores[0]  # Real (라벨 0)
        f1_positive = f1_scores[1]  # Fake (라벨 1)
    else:
        # 한 클래스만 예측된 경우
        f1_negative = 0.0
        f1_positive = 0.0
    
    return f1_positive, f1_negative


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    detailed: bool = False
) -> Dict[str, float]:
    """
    전체 평가 지표 계산
    
    Args:
        y_true: 실제 레이블 [N]
        y_pred: 예측 레이블 [N]
        detailed: 상세 지표 포함 여부
        
    Returns:
        평가 지표 딕셔너리
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Macro F1-score (주요 지표)
    metrics["macro_f1"] = calculate_macro_f1(y_true, y_pred)
    
    # 각 클래스별 F1-score
    f1_positive, f1_negative = calculate_binary_f1_scores(y_true, y_pred)
    metrics["f1_fake"] = f1_positive  # Fake (라벨 1)
    metrics["f1_real"] = f1_negative  # Real (라벨 0)
    
    if detailed:
        # Precision & Recall
        metrics["precision_macro"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        
        # 클래스별 Precision & Recall
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        
        if len(precision_per_class) == 2:
            metrics["precision_real"] = precision_per_class[0]
            metrics["precision_fake"] = precision_per_class[1]
            metrics["recall_real"] = recall_per_class[0]
            metrics["recall_fake"] = recall_per_class[1]
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negative"] = int(tn)
            metrics["false_positive"] = int(fp)
            metrics["false_negative"] = int(fn)
            metrics["true_positive"] = int(tp)
    
    return metrics


def print_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    prefix: str = ""
) -> None:
    """
    평가 지표를 보기 좋게 출력
    
    Args:
        y_true: 실제 레이블 [N]
        y_pred: 예측 레이블 [N]
        prefix: 출력 접두사 (예: "Train", "Val")
    """
    metrics = calculate_metrics(y_true, y_pred, detailed=True)
    
    print(f"\n{'='*60}")
    print(f"{prefix} 평가 지표")
    print(f"{'='*60}")
    
    # 주요 지표
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Macro F1:     {metrics['macro_f1']:.4f} ⭐ (대회 주요 지표)")
    print(f"  - F1 (Real):  {metrics['f1_real']:.4f}")
    print(f"  - F1 (Fake):  {metrics['f1_fake']:.4f}")
    
    # 상세 지표
    if "precision_macro" in metrics:
        print(f"\nPrecision:")
        print(f"  - Macro:      {metrics['precision_macro']:.4f}")
        if "precision_real" in metrics:
            print(f"  - Real (0):   {metrics['precision_real']:.4f}")
            print(f"  - Fake (1):   {metrics['precision_fake']:.4f}")
        
        print(f"\nRecall:")
        print(f"  - Macro:      {metrics['recall_macro']:.4f}")
        if "recall_real" in metrics:
            print(f"  - Real (0):   {metrics['recall_real']:.4f}")
            print(f"  - Fake (1):   {metrics['recall_fake']:.4f}")
    
    # Confusion Matrix
    if "true_positive" in metrics:
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"               Real    Fake")
        print(f"  Actual Real  {metrics['true_negative']:4d}    {metrics['false_positive']:4d}")
        print(f"         Fake  {metrics['false_negative']:4d}    {metrics['true_positive']:4d}")
    
    print(f"{'='*60}\n")


def print_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    target_names: list = None
) -> None:
    """
    sklearn classification report 출력
    
    Args:
        y_true: 실제 레이블 [N]
        y_pred: 예측 레이블 [N]
        target_names: 클래스 이름 리스트
    """
    if target_names is None:
        target_names = ["Real", "Fake"]
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))


