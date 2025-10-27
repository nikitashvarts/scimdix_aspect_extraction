"""
Span-level evaluator for aspect extraction with exact span matching.
Calculates precision, recall, F1 for each aspect class and provides confusion matrix.
"""

import numpy as np
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpanMetrics:
    """Metrics for a single span type."""
    
    precision: float
    recall: float
    f1: float
    support: int  # Number of true spans
    
    def __str__(self) -> str:
        return f"P: {self.precision:.3f}, R: {self.recall:.3f}, F1: {self.f1:.3f}, Support: {self.support}"


@dataclass
class PartialSpanMetrics:
    """Metrics for partial span matching alongside exact matching."""
    
    exact_precision: float
    exact_recall: float 
    exact_f1: float
    
    partial_precision: float  # >=50% overlap + correct type
    partial_recall: float
    partial_f1: float
    
    overlap_precision: float  # Any overlap + correct type
    overlap_recall: float
    overlap_f1: float
    
    support: int
    
    def __str__(self) -> str:
        return f"""Exact: P={self.exact_precision:.3f}, R={self.exact_recall:.3f}, F1={self.exact_f1:.3f}
Partial: P={self.partial_precision:.3f}, R={self.partial_recall:.3f}, F1={self.partial_f1:.3f}
Overlap: P={self.overlap_precision:.3f}, R={self.overlap_recall:.3f}, F1={self.overlap_f1:.3f}"""


@dataclass
class EntitySpan:
    """Represents an entity span."""
    
    start: int
    end: int  # Exclusive end
    label: str
    
    def __hash__(self):
        return hash((self.start, self.end, self.label))
    
    def __eq__(self, other):
        return (self.start, self.end, self.label) == (other.start, other.end, other.label)
    
    def __str__(self):
        return f"{self.label}({self.start}, {self.end})"


class SpanLevelEvaluator:
    """Evaluator for span-level aspect extraction metrics."""
    
    def __init__(self, label_to_id: Dict[str, int], ignore_label_id: int = -100):
        """
        Initialize evaluator.
        
        Args:
            label_to_id: Mapping from label strings to IDs
            ignore_label_id: Label ID to ignore in evaluation
        """
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        self.ignore_label_id = ignore_label_id
        
        # Extract aspect classes (exclude O and special tokens)
        self.aspect_classes = set()
        for label in label_to_id.keys():
            if label.startswith('B-') or label.startswith('I-'):
                aspect_class = label[2:]  # Remove B- or I- prefix
                self.aspect_classes.add(aspect_class)
        
        self.aspect_classes = sorted(list(self.aspect_classes))
        logger.info(f"Aspect classes for evaluation: {self.aspect_classes}")
    
    def extract_spans_from_bio_tags(
        self, 
        labels: List[int], 
        attention_mask: Optional[List[int]] = None
    ) -> Set[EntitySpan]:
        """
        Extract entity spans from BIO tags.
        
        Args:
            labels: List of label IDs
            attention_mask: Optional attention mask to ignore padding
            
        Returns:
            Set of EntitySpan objects
        """
        spans = set()
        current_span_start = None
        current_span_label = None
        
        # Convert to lists if tensors
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        if attention_mask is not None and hasattr(attention_mask, 'tolist'):
            attention_mask = attention_mask.tolist()
        
        # Apply attention mask if provided
        if attention_mask is not None:
            labels = [label if mask == 1 else self.ignore_label_id 
                     for label, mask in zip(labels, attention_mask)]
        
        for i, label_id in enumerate(labels):
            if label_id == self.ignore_label_id:
                # End current span if any
                if current_span_start is not None:
                    spans.add(EntitySpan(current_span_start, i, current_span_label))
                    current_span_start = None
                    current_span_label = None
                continue
            
            label = self.id_to_label.get(label_id, 'O')
            
            if label == 'O':
                # End current span if any
                if current_span_start is not None:
                    spans.add(EntitySpan(current_span_start, i, current_span_label))
                    current_span_start = None
                    current_span_label = None
            
            elif label.startswith('B-'):
                # End previous span if any
                if current_span_start is not None:
                    spans.add(EntitySpan(current_span_start, i, current_span_label))
                
                # Start new span
                current_span_start = i
                current_span_label = label[2:]  # Remove B- prefix
            
            elif label.startswith('I-'):
                aspect_class = label[2:]  # Remove I- prefix
                
                if current_span_start is not None and current_span_label == aspect_class:
                    # Continue current span (do nothing)
                    pass
                else:
                    # Invalid I- tag (no matching B- or different class)
                    # End previous span if any
                    if current_span_start is not None:
                        spans.add(EntitySpan(current_span_start, i, current_span_label))
                    
                    # Start new span from this I- tag
                    current_span_start = i
                    current_span_label = aspect_class
        
        # Handle span at end of sequence
        if current_span_start is not None:
            spans.add(EntitySpan(current_span_start, len(labels), current_span_label))
        
        return spans
    
    def calculate_metrics_for_class(
        self, 
        true_spans: Set[EntitySpan], 
        pred_spans: Set[EntitySpan], 
        aspect_class: str
    ) -> SpanMetrics:
        """
        Calculate precision, recall, F1 for a specific aspect class.
        
        Args:
            true_spans: Set of true spans
            pred_spans: Set of predicted spans
            aspect_class: Aspect class to evaluate
            
        Returns:
            SpanMetrics object
        """
        # Filter spans for this class
        true_class_spans = {span for span in true_spans if span.label == aspect_class}
        pred_class_spans = {span for span in pred_spans if span.label == aspect_class}
        
        # Calculate metrics
        tp = len(true_class_spans & pred_class_spans)  # True positives
        fp = len(pred_class_spans - true_class_spans)  # False positives
        fn = len(true_class_spans - pred_class_spans)  # False negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = len(true_class_spans)
        
        return SpanMetrics(precision=precision, recall=recall, f1=f1, support=support)
    
    def spans_overlap(self, span1: EntitySpan, span2: EntitySpan) -> bool:
        """Check if two spans have any overlap."""
        return not (span1.end <= span2.start or span2.end <= span1.start)
    
    def spans_partial_match(self, true_span: EntitySpan, pred_span: EntitySpan, min_overlap: float = 0.5) -> bool:
        """
        Check if spans have significant overlap (>=50% by default) and same type.
        
        Args:
            true_span: Ground truth span
            pred_span: Predicted span  
            min_overlap: Minimum overlap ratio (0.0 to 1.0)
        """
        if true_span.label != pred_span.label:
            return False
            
        # Calculate overlap
        overlap_start = max(true_span.start, pred_span.start)
        overlap_end = min(true_span.end, pred_span.end)
        
        if overlap_start >= overlap_end:
            return False  # No overlap
            
        overlap_len = overlap_end - overlap_start
        true_len = true_span.end - true_span.start
        pred_len = pred_span.end - pred_span.start
        
        # Calculate overlap ratio relative to the shorter span
        min_len = min(true_len, pred_len)
        overlap_ratio = overlap_len / min_len if min_len > 0 else 0.0
        
        return overlap_ratio >= min_overlap
    
    def calculate_partial_metrics_for_class(
        self,
        true_spans: Set[EntitySpan],
        pred_spans: Set[EntitySpan],
        aspect_class: str
    ) -> PartialSpanMetrics:
        """
        Calculate partial matching metrics for a single class.
        
        Returns:
        - Exact match (original strict matching)
        - Partial match (>=50% overlap + correct type)  
        - Overlap match (any overlap + correct type)
        """
        # Filter spans for this class
        true_class_spans = {s for s in true_spans if s.label == aspect_class}
        pred_class_spans = {s for s in pred_spans if s.label == aspect_class}
        
        if len(true_class_spans) == 0 and len(pred_class_spans) == 0:
            return PartialSpanMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        
        # 1. Exact matching (original strict)
        exact_tp = len(true_class_spans & pred_class_spans)
        exact_fp = len(pred_class_spans - true_class_spans)
        exact_fn = len(true_class_spans - pred_class_spans)
        
        exact_precision = exact_tp / (exact_tp + exact_fp) if (exact_tp + exact_fp) > 0 else 0.0
        exact_recall = exact_tp / (exact_tp + exact_fn) if (exact_tp + exact_fn) > 0 else 0.0
        exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if (exact_precision + exact_recall) > 0 else 0.0
        
        # 2. Partial matching (>=50% overlap)
        partial_matched_pred = set()
        partial_matched_true = set()
        
        for pred_span in pred_class_spans:
            for true_span in true_class_spans:
                if self.spans_partial_match(true_span, pred_span, min_overlap=0.5):
                    partial_matched_pred.add(pred_span)
                    partial_matched_true.add(true_span)
        
        partial_tp = len(partial_matched_pred)
        partial_fp = len(pred_class_spans - partial_matched_pred)
        partial_fn = len(true_class_spans - partial_matched_true)
        
        partial_precision = partial_tp / (partial_tp + partial_fp) if (partial_tp + partial_fp) > 0 else 0.0
        partial_recall = partial_tp / (partial_tp + partial_fn) if (partial_tp + partial_fn) > 0 else 0.0
        partial_f1 = 2 * partial_precision * partial_recall / (partial_precision + partial_recall) if (partial_precision + partial_recall) > 0 else 0.0
        
        # 3. Overlap matching (any overlap)
        overlap_matched_pred = set()
        overlap_matched_true = set()
        
        for pred_span in pred_class_spans:
            for true_span in true_class_spans:
                if true_span.label == pred_span.label and self.spans_overlap(true_span, pred_span):
                    overlap_matched_pred.add(pred_span)
                    overlap_matched_true.add(true_span)
        
        overlap_tp = len(overlap_matched_pred)
        overlap_fp = len(pred_class_spans - overlap_matched_pred)
        overlap_fn = len(true_class_spans - overlap_matched_true)
        
        overlap_precision = overlap_tp / (overlap_tp + overlap_fp) if (overlap_tp + overlap_fp) > 0 else 0.0
        overlap_recall = overlap_tp / (overlap_tp + overlap_fn) if (overlap_tp + overlap_fn) > 0 else 0.0
        overlap_f1 = 2 * overlap_precision * overlap_recall / (overlap_precision + overlap_recall) if (overlap_precision + overlap_recall) > 0 else 0.0
        
        return PartialSpanMetrics(
            exact_precision=exact_precision,
            exact_recall=exact_recall,
            exact_f1=exact_f1,
            partial_precision=partial_precision,
            partial_recall=partial_recall,
            partial_f1=partial_f1,
            overlap_precision=overlap_precision,
            overlap_recall=overlap_recall,
            overlap_f1=overlap_f1,
            support=len(true_class_spans)
        )
    
    def calculate_micro_metrics(
        self, 
        true_spans_list: List[Set[EntitySpan]], 
        pred_spans_list: List[Set[EntitySpan]]
    ) -> SpanMetrics:
        """
        Calculate micro-averaged metrics across all classes and examples.
        
        Args:
            true_spans_list: List of true span sets (one per example)
            pred_spans_list: List of predicted span sets (one per example)
            
        Returns:
            Micro-averaged SpanMetrics
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for true_spans, pred_spans in zip(true_spans_list, pred_spans_list):
            tp = len(true_spans & pred_spans)
            fp = len(pred_spans - true_spans)
            fn = len(true_spans - pred_spans)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = total_tp + total_fn  # Total true spans
        
        return SpanMetrics(precision=precision, recall=recall, f1=f1, support=support)
    
    def calculate_macro_metrics(
        self, 
        class_metrics: Dict[str, SpanMetrics]
    ) -> SpanMetrics:
        """
        Calculate macro-averaged metrics across all classes.
        
        Args:
            class_metrics: Metrics for each class
            
        Returns:
            Macro-averaged SpanMetrics
        """
        if not class_metrics:
            return SpanMetrics(precision=0.0, recall=0.0, f1=0.0, support=0)
        
        precisions = [metrics.precision for metrics in class_metrics.values()]
        recalls = [metrics.recall for metrics in class_metrics.values()]
        f1s = [metrics.f1 for metrics in class_metrics.values()]
        total_support = sum(metrics.support for metrics in class_metrics.values())
        
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1s)
        
        return SpanMetrics(
            precision=macro_precision, 
            recall=macro_recall, 
            f1=macro_f1, 
            support=total_support
        )
    
    def create_confusion_matrix(
        self, 
        true_spans_list: List[Set[EntitySpan]], 
        pred_spans_list: List[Set[EntitySpan]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Create confusion matrix at span level.
        
        Args:
            true_spans_list: List of true span sets
            pred_spans_list: List of predicted span sets
            
        Returns:
            Confusion matrix as nested dictionary
        """
        # Collect all positions with their true and predicted labels
        position_true = {}  # (example_idx, start, end) -> true_label
        position_pred = {}  # (example_idx, start, end) -> pred_label
        
        for example_idx, (true_spans, pred_spans) in enumerate(zip(true_spans_list, pred_spans_list)):
            for span in true_spans:
                key = (example_idx, span.start, span.end)
                position_true[key] = span.label
            
            for span in pred_spans:
                key = (example_idx, span.start, span.end)
                position_pred[key] = span.label
        
        # Build confusion matrix
        all_classes = set(self.aspect_classes) | {'O'}
        confusion_matrix = {true_class: {pred_class: 0 for pred_class in all_classes} 
                           for true_class in all_classes}
        
        # Count true positives (correct spans)
        for key in position_true:
            true_label = position_true[key]
            pred_label = position_pred.get(key, 'O')  # If not predicted, it's O
            confusion_matrix[true_label][pred_label] += 1
        
        # Count false positives (incorrectly predicted spans)
        for key in position_pred:
            if key not in position_true:  # This span was predicted but not true
                pred_label = position_pred[key]
                confusion_matrix['O'][pred_label] += 1
        
        return confusion_matrix
    
    def evaluate(
        self, 
        true_labels_list: List[List[int]], 
        pred_labels_list: List[List[int]],
        attention_masks_list: Optional[List[List[int]]] = None
    ) -> Dict[str, any]:
        """
        Perform complete span-level evaluation.
        
        Args:
            true_labels_list: List of true label sequences
            pred_labels_list: List of predicted label sequences
            attention_masks_list: Optional list of attention masks
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        if len(true_labels_list) != len(pred_labels_list):
            raise ValueError("Number of true and predicted sequences must match")
        
        if attention_masks_list is None:
            attention_masks_list = [None] * len(true_labels_list)
        
        # Extract spans for all examples
        true_spans_list = []
        pred_spans_list = []
        
        for true_labels, pred_labels, attention_mask in zip(
            true_labels_list, pred_labels_list, attention_masks_list
        ):
            true_spans = self.extract_spans_from_bio_tags(true_labels, attention_mask)
            pred_spans = self.extract_spans_from_bio_tags(pred_labels, attention_mask)
            
            true_spans_list.append(true_spans)
            pred_spans_list.append(pred_spans)
        
        # Calculate per-class metrics
        class_metrics = {}
        for aspect_class in self.aspect_classes:
            # Combine all spans for this class across examples
            all_true_spans = set()
            all_pred_spans = set()
            
            for example_idx, (true_spans, pred_spans) in enumerate(zip(true_spans_list, pred_spans_list)):
                # Add example index to make spans unique across examples
                for span in true_spans:
                    if span.label == aspect_class:
                        all_true_spans.add(EntitySpan(
                            span.start + example_idx * 10000,  # Offset by example
                            span.end + example_idx * 10000,
                            span.label
                        ))
                
                for span in pred_spans:
                    if span.label == aspect_class:
                        all_pred_spans.add(EntitySpan(
                            span.start + example_idx * 10000,
                            span.end + example_idx * 10000,
                            span.label
                        ))
            
            class_metrics[aspect_class] = self.calculate_metrics_for_class(
                all_true_spans, all_pred_spans, aspect_class
            )
        
        # Calculate partial metrics for all classes
        partial_class_metrics = {}
        for aspect_class in self.aspect_classes:
            # Combine all spans for this class across examples
            all_true_spans = set()
            all_pred_spans = set()
            
            for example_idx, (true_spans, pred_spans) in enumerate(zip(true_spans_list, pred_spans_list)):
                # Add example index to make spans unique across examples
                for span in true_spans:
                    if span.label == aspect_class:
                        all_true_spans.add(EntitySpan(
                            span.start + example_idx * 10000,  # Offset by example
                            span.end + example_idx * 10000,
                            span.label
                        ))
                
                for span in pred_spans:
                    if span.label == aspect_class:
                        all_pred_spans.add(EntitySpan(
                            span.start + example_idx * 10000,
                            span.end + example_idx * 10000,
                            span.label
                        ))
            
            partial_class_metrics[aspect_class] = self.calculate_partial_metrics_for_class(
                all_true_spans, all_pred_spans, aspect_class
            )
        
        # Calculate micro and macro metrics
        micro_metrics = self.calculate_micro_metrics(true_spans_list, pred_spans_list)
        macro_metrics = self.calculate_macro_metrics(class_metrics)
        
        # Calculate aggregated partial metrics
        partial_f1_scores = []
        overlap_f1_scores = []
        for class_name, metrics in partial_class_metrics.items():
            if metrics.support > 0:  # Only include classes with true examples
                partial_f1_scores.append(metrics.partial_f1)
                overlap_f1_scores.append(metrics.overlap_f1)
        
        avg_partial_f1 = sum(partial_f1_scores) / len(partial_f1_scores) if partial_f1_scores else 0.0
        avg_overlap_f1 = sum(overlap_f1_scores) / len(overlap_f1_scores) if overlap_f1_scores else 0.0
        
        # Create confusion matrix
        confusion_matrix = self.create_confusion_matrix(true_spans_list, pred_spans_list)
        
        # Prepare results
        results = {
            'class_metrics': class_metrics,
            'partial_class_metrics': partial_class_metrics,  # NEW: Partial metrics
            'micro_avg': micro_metrics,
            'macro_avg': macro_metrics,
            'confusion_matrix': confusion_matrix,
            'num_examples': len(true_labels_list),
            'total_true_spans': sum(len(spans) for spans in true_spans_list),
            'total_pred_spans': sum(len(spans) for spans in pred_spans_list),
            # Add trainer-compatible keys
            'test_micro_f1': micro_metrics.f1,
            'test_macro_f1': macro_metrics.f1,
            # Add aggregated partial metrics
            'avg_partial_f1': avg_partial_f1,
            'avg_overlap_f1': avg_overlap_f1
        }
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, any]):
        """Print a detailed evaluation report."""
        print("\n" + "="*60)
        print("SPAN-LEVEL EVALUATION REPORT")
        print("="*60)
        
        print("\nDataset Summary:")
        print(f"  Examples: {results['num_examples']}")
        print(f"  True spans: {results['total_true_spans']}")
        print(f"  Predicted spans: {results['total_pred_spans']}")
        
        print("\nPer-Class Metrics:")
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name in sorted(results['class_metrics'].keys()):
            metrics = results['class_metrics'][class_name]
            print(f"{class_name:<12} {metrics.precision:<10.3f} {metrics.recall:<10.3f} "
                  f"{metrics.f1:<10.3f} {metrics.support:<10}")
        
        print("-" * 60)
        micro = results['micro_avg']
        macro = results['macro_avg']
        print(f"{'micro avg':<12} {micro.precision:<10.3f} {micro.recall:<10.3f} "
              f"{micro.f1:<10.3f} {micro.support:<10}")
        print(f"{'macro avg':<12} {macro.precision:<10.3f} {macro.recall:<10.3f} "
              f"{macro.f1:<10.3f} {macro.support:<10}")
        
        # Add partial metrics section
        if 'partial_class_metrics' in results:
            print("\n" + "="*60)
            print("PARTIAL MATCHING METRICS")
            print("="*60)
            print("Legend:")
            print("  Exact: Perfect boundary and type match")
            print("  Partial: >=50% overlap + correct type")
            print("  Overlap: Any overlap + correct type")
            print()
            
            for class_name in sorted(results['partial_class_metrics'].keys()):
                metrics = results['partial_class_metrics'][class_name]
                if metrics.support > 0:  # Only show classes that have true examples
                    print(f"Class: {class_name}")
                    print(f"  Exact:   P={metrics.exact_precision:.3f}, R={metrics.exact_recall:.3f}, F1={metrics.exact_f1:.3f}")
                    print(f"  Partial: P={metrics.partial_precision:.3f}, R={metrics.partial_recall:.3f}, F1={metrics.partial_f1:.3f}")
                    print(f"  Overlap: P={metrics.overlap_precision:.3f}, R={metrics.overlap_recall:.3f}, F1={metrics.overlap_f1:.3f}")
                    print(f"  Support: {metrics.support}")
                    print()
        
        print("\nConfusion Matrix (True vs Predicted):")
        confusion = results['confusion_matrix']
        classes = sorted(confusion.keys())
        
        # Print header
        header = "True\\Pred"
        print(f"{header:<12}", end="")
        for pred_class in classes:
            print(f"{pred_class:<8}", end="")
        print()
        
        # Print matrix
        for true_class in classes:
            print(f"{true_class:<12}", end="")
            for pred_class in classes:
                count = confusion[true_class].get(pred_class, 0)
                print(f"{count:<8}", end="")
            print()
        
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Test evaluator with sample data
    from src.model.model import create_label_mapping
    
    label_to_id, id_to_label = create_label_mapping()
    evaluator = SpanLevelEvaluator(label_to_id)
    
    # Sample true and predicted labels (BIO format)
    true_labels = [
        [0, 1, 2, 2, 0, 3, 4, 0, 0],  # O B-AIM I-AIM I-AIM O B-METHOD I-METHOD O O
        [0, 0, 7, 8, 8, 0, 11, 12]    # O O B-TASK I-TASK I-TASK O B-RESULT I-RESULT
    ]
    
    pred_labels = [
        [0, 1, 2, 0, 0, 3, 4, 4, 0],  # Different prediction
        [0, 0, 7, 8, 0, 0, 11, 12]    # Missing one I-TASK
    ]
    
    print("Testing span-level evaluator...")
    print(f"Label mapping: {label_to_id}")
    
    # Evaluate
    results = evaluator.evaluate(true_labels, pred_labels)
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    print("\n✅ Evaluator test completed!")