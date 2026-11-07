import math
import statistics
from .data_apps import parse
from .validation import ValidationError

def number(value, name, low, high):
    if isinstance(value, bool):
        raise ValidationError(f'{name} must be numeric')
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValidationError(f'{name} must be numeric') from None
    if not math.isfinite(result) or not low <= result <= high:
        raise ValidationError(f'{name} must be finite and between {low} and {high}')
    return result

def auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(scores, labels))
    rank_sum, start = (0, 0)
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        rank_sum += sum((y for _, y in pairs[start:end])) * (start + 1 + end) / 2
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

def threshold_metrics(labels, scores, threshold=0.5):
    tp = fp = tn = fn = 0
    for y, p in zip(labels, scores):
        predicted = p >= threshold
        tp += y == 1 and predicted
        fp += y == 0 and predicted
        tn += y == 0 and (not predicted)
        fn += y == 1 and (not predicted)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'accuracy': (tp + tn) / len(labels), 'precision': precision, 'recall': recall, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None, 'confusion': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}}

def probability_metrics(labels, scores):
    return {'roc_auc': auc(labels, scores), 'brier': statistics.mean(((p - y) ** 2 for y, p in zip(labels, scores))), 'log_loss': -statistics.mean((y * math.log(max(1e-15, p)) + (1 - y) * math.log(max(1e-15, 1 - p)) for y, p in zip(labels, scores)))}

def evaluate(data):
    headers, rows = parse(data)
    if 'actual' not in headers or 'score' not in headers:
        raise ValidationError('CSV requires actual and score columns')
    labels, scores = ([], [])
    for index, row in enumerate(rows, 2):
        y = number(row[headers.index('actual')], f'actual at record {index}', 0, 1)
        if y not in (0, 1):
            raise ValidationError('actual labels must be 0 or 1')
        labels.append(int(y))
        scores.append(number(row[headers.index('score')], f'score at record {index}', 0, 1))
    threshold = number(data.get('threshold', 0.5), 'threshold', 0, 1)
    bins = []
    for index in range(5):
        selected = [i for i, p in enumerate(scores) if min(4, int(p * 5)) == index]
        bins.append({'range': f'{index / 5:.1f}–{(index + 1) / 5:.1f}', 'count': len(selected), 'mean_score': statistics.mean((scores[i] for i in selected)) if selected else None, 'observed_rate': statistics.mean((labels[i] for i in selected)) if selected else None})
    quality = probability_metrics(labels, scores)
    return {'rows': len(rows), 'threshold': threshold, 'positive_rate': sum(labels) / len(labels), 'metrics': {**threshold_metrics(labels, scores, threshold), **quality}, 'calibration': bins, 'predictions': [{'record': index, 'actual': actual, 'score': score, 'predicted': int(score >= threshold)} for index, (actual, score) in enumerate(zip(labels, scores), 2)], 'thresholds': [{'threshold': i / 10, **threshold_metrics(labels, scores, i / 10), **quality} for i in range(11)]}
