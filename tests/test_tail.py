"""(b) Tail-class set construction."""
import numpy as np

from eval.tail import per_class_error, r_tail_dynamic, r_tail_static, static_tail_classes


def test_static_tail_selects_worst_30pct():
    # 10 classes, accuracy ascending 0.10..1.00 -> worst 30% = classes 0,1,2
    acc = np.linspace(0.10, 1.00, 10)
    tail = static_tail_classes(acc, frac=0.30)
    assert tail.tolist() == [0, 1, 2]


def test_static_tail_tie_break_by_class_id():
    acc = np.array([0.5, 0.5, 0.5, 0.9, 0.9])  # 30% of 5 -> 2 worst, ties -> lowest ids
    tail = static_tail_classes(acc, frac=0.30)
    assert tail.tolist() == [0, 1]


def test_per_class_error_and_static_mean():
    correct = np.array([90, 50, 10, 100])
    total = np.array([100, 100, 100, 100])
    err = per_class_error(correct, total)
    np.testing.assert_allclose(err, [0.10, 0.50, 0.90, 0.00])
    # tail = classes {2,1} -> mean error (0.90+0.50)/2
    assert abs(r_tail_static(err, np.array([1, 2])) - 0.70) < 1e-9


def test_dynamic_tail_matches_worst_fraction():
    err = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # worst 30% of 10 = 3 classes: 0.9,0.8,0.7 -> mean 0.8
    assert abs(r_tail_dynamic(err, frac=0.30) - 0.8) < 1e-9


def test_zero_sample_classes_are_ignored():
    err = per_class_error(np.array([10, 0, 5]), np.array([10, 0, 10]))
    assert np.isnan(err[1])
    # static mean over {0,2} ignores nan class only if not selected; nanmean guards
    assert abs(r_tail_static(err, np.array([0, 2])) - 0.25) < 1e-9
