"""
Evaluation metrics for classification.
"""

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import numpy as np
from typing import List, Dict
from loguru import logger


def calculate_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of all possible labels
        
    Returns:
        Dict of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Macro average (treats all classes equally)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    
    # Weighted average (accounts for class imbalance)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Per-class metrics dict
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    metrics = {
        'overall_accuracy': float(accuracy),
        'macro_avg': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1_score': float(macro_f1)
        },
        'weighted_avg': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1_score': float(weighted_f1)
        },
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"Overall Accuracy: {accuracy:.3f}")
    logger.info(f"Macro F1: {macro_f1:.3f}")
    logger.info(f"Weighted F1: {weighted_f1:.3f}")
    
    return metrics


def calculate_confidence_calibration(confidences: List[float], correct: List[bool], n_bins: int = 10) -> Dict:
    """
    Calculate confidence calibration metrics.
    
    Args:
        confidences: Predicted confidence scores
        correct: Whether predictions were correct
        n_bins: Number of bins for calibration
        
    Returns:
        Calibration metrics
    """
    confidences = np.array(confidences)
    correct = np.array(correct)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate accuracy per bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracies.append(correct[mask].mean())
            bin_confidences.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    # Expected Calibration Error (ECE)
    ece = sum([
        (count / len(confidences)) * abs(acc - conf)
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
    ])
    
    return {
        'expected_calibration_error': float(ece),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }
