"""Synthetic checks for M3 native AP extraction and explicit baseline differences."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_onnx import (BASELINE_SETTINGS, EVALUATION, EXPECTED_IMAGES,
                                 EXPECTED_OBJECTS, PROTOCOL, metric_deltas,
                                 summarize_ap, validate_baseline)


class EvaluationTests(unittest.TestCase):
    def test_native_class_order_and_ap_columns(self):
        ap = np.array([[.9, .8, .7, .6, .5, .4, .3, .2, .1, .0],
                       [.8, .7, .6, .5, .4, .3, .2, .1, .0, .0]])
        result = summarize_ap(ap, [1, 0])
        self.assertEqual(result['classes']['weed']['AP50'], .9)
        self.assertEqual(result['classes']['crop']['AP75'], .3)
        self.assertEqual(result['aggregate']['mAP50-95'], float(ap.mean()))
        self.assertEqual(result['iou_thresholds'], [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95])

    def test_bad_native_ap_is_not_reported_as_success(self):
        for ap, indices in [(np.zeros((1, 10)), [0]), (np.zeros((2, 10)), [0, 0]),
                            (np.full((2, 10), np.nan), [0, 1]), (np.full((2, 10), 1.1), [0, 1])]:
            with self.subTest(shape=ap.shape, indices=indices), self.assertRaises(ValueError):
                summarize_ap(ap, indices)

    def test_signed_and_absolute_differences_preserve_regression(self):
        old = summarize_ap(np.full((2, 10), .5), [0, 1])
        new = deepcopy(old)
        new['classes']['weed']['AP50'] = .49
        comparison = metric_deltas(old, new, .005)
        row = comparison['classes']['weed']['AP50']
        self.assertAlmostEqual(row['signed_delta'], -.01)
        self.assertAlmostEqual(row['absolute_delta'], .01)
        self.assertEqual(comparison['metrics_requiring_investigation'], ['weed.AP50'])
        self.assertTrue(comparison['investigation_required'])
        self.assertFalse(metric_deltas(old, old, .005)['investigation_required'])

    def test_threshold_equality_is_not_exceeded(self):
        old = summarize_ap(np.zeros((2, 10)), [0, 1])
        new = deepcopy(old)
        new['aggregate']['AP50'] = .005
        self.assertFalse(metric_deltas(old, new, .005)['investigation_required'])
        new['aggregate']['AP50'] = np.nextafter(.005, np.inf)
        self.assertTrue(metric_deltas(old, new, .005)['investigation_required'])

    def test_frozen_baseline_preconditions(self):
        baseline = {**summarize_ap(np.zeros((2, 10)), [0, 1]),
                    'checkpoint_sha256': 'test', 'settings': dict(BASELINE_SETTINGS),
                    'protocol': PROTOCOL, 'images': EXPECTED_IMAGES,
                    'objects': EXPECTED_OBJECTS, 'inference_precision': 'FP32'}
        validate_baseline(baseline, 'test')
        for key, value in [('checkpoint_sha256', 'changed'), ('protocol', 'filtered'),
                           ('images', 1), ('objects', 0), ('inference_precision', 'FP16')]:
            changed = {**baseline, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_baseline(changed, 'test')
        changed = deepcopy(baseline)
        changed['settings']['conf'] = .25
        with self.assertRaises(ValueError):
            validate_baseline(changed, 'test')

    def test_only_documented_execution_settings_differ(self):
        changed = {key for key in BASELINE_SETTINGS if BASELINE_SETTINGS[key] != EVALUATION[key]}
        self.assertEqual(changed, {'device', 'batch', 'workers', 'plots'})


if __name__ == '__main__':
    unittest.main()
