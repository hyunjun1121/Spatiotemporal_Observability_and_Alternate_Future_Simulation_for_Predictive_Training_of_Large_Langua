from __future__ import annotations

import numpy as np

from src.eval.stats import lock_judgement


def _mk_group(instab, wasted, pplx):
    seeds = [1337, 2337, 3337]
    seed_index = {seed: idx for idx, seed in enumerate(seeds)}
    metrics = {
        'instability_events_per_100k': np.asarray(instab, dtype=float),
        'wasted_flops_est_total': np.asarray(wasted, dtype=float),
        'val_pplx': np.asarray(pplx, dtype=float),
        'time_to_recover_mean': np.asarray([12.0, 14.0, 11.0], dtype=float),
        'branch_invoke_rate': np.asarray([0.12, 0.10, 0.11], dtype=float),
        'accept_rate': np.asarray([0.42, 0.44, 0.45], dtype=float),
        'proxy_real_gap_mean': np.asarray([0.08, 0.09, 0.07], dtype=float),
    }
    return {'seeds': seeds, 'seed_index': seed_index, 'metrics': metrics}


def test_lock_judgement_pass():
    baseline = _mk_group([110.0, 108.0, 112.0], [210.0, 205.0, 215.0], [10.05, 10.10, 10.00])
    method = _mk_group([70.0, 72.0, 68.0], [160.0, 158.0, 162.0], [10.02, 10.08, 9.98])
    passed, reason, details = lock_judgement(baseline, method)
    assert passed, reason
    assert 'instability_events_per_100k' in details
    assert details['instability_events_per_100k']['improvement']  > 0


def test_lock_judgement_fail_requirements():
    baseline = _mk_group([100.0, 102.0, 98.0], [200.0, 202.0, 198.0], [9.95, 10.05, 9.90])
    method = _mk_group([95.0, 92.0, 96.0], [190.0, 189.0, 191.0], [10.10, 10.12, 10.11])
    passed, reason, details = lock_judgement(baseline, method)
    assert not passed
    assert '미달' in reason
    assert 'instability_events_per_100k' in details

