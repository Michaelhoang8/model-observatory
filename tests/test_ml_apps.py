import json
import math
import statistics
import unittest
from portfolio import ml_apps as m
from portfolio.validation import ValidationError

def training_csv():
    return 'signal,constant,label\n' + '\n'.join((f'{i},7,{int(i > 0)}' for i in range(-20, 21) if i))

class MLTests(unittest.TestCase):

    def test_oversized_numeric_parameters_raise_validation_errors(self):
        with self.assertRaises(ValidationError):
            m.number(10 ** 1000, 'threshold', 0, 1)
        with self.assertRaises(ValidationError):
            m.evaluate({'csv': 'actual,score\n0,0.1\n1,0.9', 'threshold': 10 ** 1000})

    def test_auc_known_rankings_and_ties(self):
        self.assertEqual(m.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(m.auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(m.auc([0, 1], [0.9, 0.1]), 0)
        self.assertIsNone(m.auc([1, 1], [0.1, 0.9]))

    def test_evaluation_calibration_includes_endpoints(self):
        r = m.evaluate({'csv': 'actual,score\n0,0\n1,1\n1,0.2'})
        self.assertEqual(sum((b['count'] for b in r['calibration'])), 3)
        self.assertEqual(r['calibration'][0]['count'], 1)
        self.assertEqual(r['calibration'][1]['count'], 1)
        self.assertEqual(r['calibration'][-1]['count'], 1)

    def test_evaluation_rejects_invalid_inputs(self):
        for csv in ['actual,score\n0,NaN', 'actual,score\n0,1.1', 'actual,score\n0.5,0.2', 'x,y\n1,2']:
            with self.assertRaises(ValidationError):
                m.evaluate({'csv': csv})
