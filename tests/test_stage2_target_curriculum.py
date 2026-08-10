from scripts.train_stage2 import scheduled_measurement_consistency_weight


def test_measurement_consistency_schedule_is_piecewise_linear():
    cfg = {
        "measurement_consistency_schedule": [
            {"epoch": 1, "weight": 0.0},
            {"epoch": 5, "weight": 0.0},
            {"epoch": 10, "weight": 0.005},
            {"epoch": 20, "weight": 0.02},
        ]
    }
    assert scheduled_measurement_consistency_weight(cfg, 1) == 0.0
    assert scheduled_measurement_consistency_weight(cfg, 5) == 0.0
    assert scheduled_measurement_consistency_weight(cfg, 10) == 0.005
    assert scheduled_measurement_consistency_weight(cfg, 20) == 0.02
    assert scheduled_measurement_consistency_weight(cfg, 25) == 0.02
    assert scheduled_measurement_consistency_weight(cfg, 15) == 0.0125
