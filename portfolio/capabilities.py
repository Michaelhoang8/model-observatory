from . import data_apps, ml_apps

def table(title, headers, rows):
    return {'title': title, 'headers': headers, 'rows': rows}

def enrich(path, data, result):
    """Attach diagnostics without changing existing response fields."""
    tables, notes = ([], [])
    fp = ml_apps.number(data.get('false_positive_cost', 1), 'False positive cost', 0, 1000000.0)
    fn = ml_apps.number(data.get('false_negative_cost', 5), 'False negative cost', 0, 1000000.0)
    candidates = []
    for row in result['thresholds']:
        c = row['confusion']
        cost = c['fp'] * fp + c['fn'] * fn
        candidates.append([row['threshold'], c['fp'], c['fn'], cost, round(cost / result['rows'], 5)])
    best = min(candidates, key=lambda row: (row[3], -row[0]))
    result['cost_analysis'] = {'fp_cost': fp, 'fn_cost': fn, 'best_grid_threshold': best[0], 'minimum_cost': best[3]}
    tables = [table('Cost-sensitive threshold grid', ['Threshold', 'False positives', 'False negatives', 'Total cost', 'Cost / row'], candidates)]
    notes = [f'Lowest cost on this 0.1-step grid: threshold {best[0]} (cost {best[3]:g}). Ties choose the higher threshold. Select thresholds on validation data, then assess once on an untouched test set. Costs are user-specified units.']
    audit = []
    for item in result['predictions']:
        actual, predicted = (item['actual'], item['predicted'])
        outcome = ('true ' if actual == predicted else 'false ') + ('positive' if predicted else 'negative')
        cost = 0 if actual == predicted else fp if predicted else fn
        audit.append([item['record'], actual, item['score'], predicted, outcome, result['threshold'], cost])
    errors = [row for row in audit if row[1] != row[3]]
    result['prediction_audit_csv'] = data_apps.safe_csv(['record', 'actual', 'score', 'predicted', 'outcome', 'threshold', 'error_cost'], [[str(value) for value in row] for row in audit])
    tables.append(table(f'Misclassified records · {min(20, len(errors))} of {len(errors)} shown', ['Record', 'Actual', 'Score', 'Predicted', 'Outcome', 'Threshold', 'Error cost'], errors[:20]))
    notes.append('The error preview and full prediction audit use your selected threshold, not the lowest-cost grid threshold. Record numbers include the CSV header as record 1. Other source columns are omitted; join by record number when investigating errors.')
    if tables or notes:
        result['capabilities'] = {'tables': tables, 'notes': notes}
    return result
