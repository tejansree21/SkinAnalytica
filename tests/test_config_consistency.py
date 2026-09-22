"""
Guards against the exact bug this whole review session started with:
config/skin_config.yaml said mel threshold=0.45 while api/SA05_api.py's
deployed code used 0.312, silently, for who knows how long. skin_config.yaml
is documentation, not a loaded config file (see the comment block at the top
of its `inference:` section for why runtime YAML-loading was rejected) -- so
the only thing keeping it honest is this test. If it fails, either the YAML
comment was edited without updating the code, or vice versa; fix whichever
one is stale, don't just update the other side of this test.
"""
import os

import yaml

import SA05_api as api

CONFIG_PATH = os.path.join(api.BASE, "config", "skin_config.yaml")


def _load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_mel_threshold_matches_config():
    cfg = _load_config()
    assert cfg["inference"]["thresholds"]["mel"] == api.MEL_THRESHOLD


def test_age_conditional_thresholds_match_config():
    cfg = _load_config()
    cfg_thresholds = cfg["inference"]["age_conditional_mel_thresholds"]
    # config keys are display strings ("0-30", "75+"); code keys are (lo, hi) tuples
    expected = {
        "0-30": api.AGE_BAND_THRESHOLDS[(0, 30)],
        "30-45": api.AGE_BAND_THRESHOLDS[(30, 45)],
        "45-60": api.AGE_BAND_THRESHOLDS[(45, 60)],
        "60-75": api.AGE_BAND_THRESHOLDS[(60, 75)],
        "75+": api.AGE_BAND_THRESHOLDS[(75, 200)],
    }
    assert cfg_thresholds == expected, (
        "config/skin_config.yaml's age_conditional_mel_thresholds has drifted "
        "from api/SA05_api.py's AGE_BAND_THRESHOLDS -- update whichever one is stale"
    )


def test_review_band_factor_matches_config():
    cfg = _load_config()
    assert cfg["inference"]["review_band_factor"] == api.MEL_REVIEW_BAND_FACTOR


def test_tbp_threshold_matches_config():
    cfg = _load_config()
    assert cfg["inference"]["tbp_mel_threshold"] == api.TBP_MEL_THRESHOLD


def test_skin_tone_thresholds_match_config():
    cfg = _load_config()
    cfg_thresholds = cfg["inference"]["skin_tone_mel_thresholds"]
    expected = {str(k): v for k, v in api.SKIN_TONE_THRESHOLDS.items()}
    assert cfg_thresholds == expected, (
        "config/skin_config.yaml's skin_tone_mel_thresholds has drifted "
        "from api/SA05_api.py's SKIN_TONE_THRESHOLDS -- update whichever one is stale"
    )
