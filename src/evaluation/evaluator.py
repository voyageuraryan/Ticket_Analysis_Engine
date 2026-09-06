"""
Model evaluation pipeline.
"""

import pandas as pd
from typing import List, Dict
from loguru import logger
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import calculate_metrics, calculate_confidence_calibration


class ModelEvaluator:
    """Evaluate SAP module classifier."""
    
    def __init__(self, classifier, sap_modules: List[str]):
        """
        Initialize evaluator.
        
        Args:
            classifier: SAPModuleClassifier instance
            sap_modules: List of SAP modules
        """
        self.classifier = classifier
        self.sap_modules = sap_modules
        logger.info("Initialized ModelEvaluator")
    
    def evaluate(self, test_df: pd.DataFrame, output_dir: str = "outputs/results") -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_df: Test DataFrame
            output_dir: Directory to save results
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on {len(test_df)} test samples")
        
        # Prepare output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run predictions
        y_true = []
        y_pred = []
        confidences = []
        predictions = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            try:
                result = self.classifier.predict(
                    summary=row['Summary'],
                    description=row['Description'],
                    incident_number=str(row['Incident'])
                )
                
                y_true.append(row['Module'])
                y_pred.append(result['module'])
                confidences.append(result['confidence'])
                predictions.append(result)
                
            except Exception as e:
                logger.error(f"Prediction failed for {row['Incident']}: {e}")
                y_true.append(row['Module'])
                y_pred.append('Unknown')
                confidences.append(0.0)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, self.sap_modules)
        
        # Calculate confidence calibration
        correct = [true == pred for true, pred in zip(y_true, y_pred)]
        calibration = calculate_confidence_calibration(confidences, correct)
        
        # Combine results
        results = {
            'metrics': metrics,
            'calibration': calibration,
            'predictions': predictions[:100]  # Save first 100 for inspection
        }
        
        # Save results
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(
            metrics['confusion_matrix'],
            self.sap_modules,
            output_path / "confusion_matrix.png"
        )
        
        # Plot per-class F1 scores
        self._plot_per_class_f1(
            metrics['per_class'],
            output_path / "per_class_f1.png"
        )
        
        # Generate report
        self._generate_report(results, output_path / "evaluation_report.txt")
        
        return results
    
    def _plot_confusion_matrix(self, cm: List[List[int]], labels: List[str], save_path: Path):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def _plot_per_class_f1(self, per_class_metrics: Dict, save_path: Path):
        """Plot per-class F1 scores."""
        modules = list(per_class_metrics.keys())
        f1_scores = [per_class_metrics[m]['f1_score'] for m in modules]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modules, f1_scores, color='skyblue', edgecolor='navy')
        
        # Color bars based on score
        for bar, score in zip(bars, f1_scores):
            if score >= 0.8:
                bar.set_color('green')
            elif score >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.title('Per-Module F1 Scores')
        plt.xlabel('SAP Module')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1.0)
        plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.3, label='Target (0.8)')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Per-class F1 plot saved to {save_path}")
    
    def _generate_report(self, results: Dict, save_path: Path):
        """Generate text report."""
        metrics = results['metrics']
        calibration = results['calibration']
        
        report = []
        report.append("=" * 80)
        report.append("SAP MODULE CLASSIFIER - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append(f"  Accuracy: {metrics['overall_accuracy']:.3f}")
        report.append(f"  Macro F1: {metrics['macro_avg']['f1_score']:.3f}")
        report.append(f"  Weighted F1: {metrics['weighted_avg']['f1_score']:.3f}")
        report.append(f"  Expected Calibration Error: {calibration['expected_calibration_error']:.3f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-MODULE METRICS:")
        report.append("-" * 80)
        report.append(f"{'Module':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support':<10}")
        report.append("-" * 80)
        
        for module, module_metrics in metrics['per_class'].items():
            report.append(
                f"{module:<20} "
                f"{module_metrics['precision']:<12.3f} "
                f"{module_metrics['recall']:<12.3f} "
                f"{module_metrics['f1_score']:<12.3f} "
                f"{module_metrics['support']:<10}"
            )
        
        report.append("=" * 80)
        
        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Evaluation report saved to {save_path}")
        
        # Also print to console
        print('\n'.join(report))
