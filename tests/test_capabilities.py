import json
import csv
import io
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from portfolio import capabilities as c, data_apps as d, ml_apps as ml
from portfolio.db import initialize, connect, SCHEMA
from portfolio.validation import ValidationError, NotFoundError

def enhanced(path, data, function):
    return c.enrich(path, data, function(data))

class AnalysisCapabilities(unittest.TestCase):

    def test_evaluation_audit_reconciles_threshold_errors_and_costs(self):
        data = {'csv': 'actual,score,private_note\n0,0.2,OMIT_THIS\n1,0.7,OMIT_THIS\n0,0.7,OMIT_THIS\n1,0.69,OMIT_THIS', 'threshold': 0.7, 'false_positive_cost': 2, 'false_negative_cost': 9}
        result = enhanced('/api/ml/evaluate', data, ml.evaluate)
        rows = list(csv.DictReader(io.StringIO(result['prediction_audit_csv'])))
        self.assertEqual([row['outcome'] for row in rows], ['true negative', 'true positive', 'false positive', 'false negative'])
        self.assertEqual([row['record'] for row in rows], ['2', '3', '4', '5'])
        self.assertEqual([float(row['error_cost']) for row in rows], [0, 0, 2, 9])
        self.assertTrue(all((float(row['threshold']) == 0.7 for row in rows)))
        self.assertEqual(result['metrics']['confusion'], {'tp': 1, 'fp': 1, 'tn': 1, 'fn': 1})
        self.assertEqual([row[0] for row in result['capabilities']['tables'][-1]['rows']], [4, 5])
        self.assertNotIn('OMIT_THIS', json.dumps(result))

    def test_evaluation_audit_exports_every_record_beyond_error_preview(self):
        result = enhanced('/api/ml/evaluate', {'csv': 'actual,score\n' + '1,0.123456789012345\n' * 5000}, ml.evaluate)
        rows = list(csv.DictReader(io.StringIO(result['prediction_audit_csv'])))
        self.assertEqual(len(rows), 5000)
        self.assertEqual(rows[-1]['record'], '5001')
        self.assertEqual(rows[0]['score'], '0.123456789012345')
        self.assertEqual(len(result['capabilities']['tables'][-1]['rows']), 20)
        self.assertEqual(result['metrics']['confusion']['fn'], 5000)

    def test_cost_analysis_zero_costs_and_null_validation(self):
        data = {'csv': 'actual,score\n1,0.2\n0,0.8', 'false_positive_cost': 0, 'false_negative_cost': 0}
        r = enhanced('/api/ml/evaluate', data, ml.evaluate)
        self.assertEqual(r['cost_analysis']['best_grid_threshold'], 1)
        with self.assertRaises(ValidationError):
            enhanced('/api/ml/evaluate', {**data, 'false_negative_cost': None}, ml.evaluate)

    def test_cost_threshold_changes_with_error_costs(self):
        data = {'csv': 'actual,score\n1,0.4\n0,0.6\n', 'false_positive_cost': 1, 'false_negative_cost': 100}
        result = enhanced('/api/ml/evaluate', data, ml.evaluate)
        self.assertEqual(result['cost_analysis']['best_grid_threshold'], 0.4)
        self.assertEqual(result['cost_analysis']['minimum_cost'], 1)
        data['false_negative_cost'] = -1
        with self.assertRaises(ValidationError):
            enhanced('/api/ml/evaluate', data, ml.evaluate)
if __name__ == '__main__':
    unittest.main()
