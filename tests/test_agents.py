"""
Tests for agents/ -- flagged as a gap in the independent audit ("no tests
exist for the agents, delivery, or assistant services at all"). Targets
the real, pure logic in each agent (lifecycle/error-handling, drift
detection math, fairness disparity detection, annotation-QA thresholds)
rather than superficial smoke coverage -- same discipline as the rest of
this test suite. Model-loading agents (VerificationAgent's __init__ needs
a real ONNX file) are tested via their pure methods directly, bypassing
__init__, so these tests don't need production model files present.
"""
import json
import os

import numpy as np
import pytest

from base_agent import BaseAgent
from drift_monitor_agent import DriftMonitorAgent, TRAIN_DIST, TRAIN_CONF_MEAN
from fairness_monitor_agent import FairnessMonitorAgent
from verification_agent import VerificationAgent, AMBIGUOUS_GAP, CONFIDENCE_FLOOR, CONFLICT_SCORE


# ---------------------------------------------------------------------------
# base_agent.py -- lifecycle, error handling, persistence
# ---------------------------------------------------------------------------

class _EchoAgent(BaseAgent):
    """Minimal concrete agent for exercising BaseAgent's lifecycle."""
    def __init__(self, name="echo_agent_test", fail=False):
        super().__init__(name)
        self._fail = fail

    def _run(self, value=None):
        if self._fail:
            raise ValueError("deliberate failure")
        return {"echoed": value}


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    """Redirect agent log persistence to a tmp dir so tests never touch
    the real outputs/agent_logs/ directory."""
    import base_agent
    monkeypatch.setattr(base_agent, "LOG_DIR", str(tmp_path))
    return tmp_path


def test_base_agent_run_success_path_sets_status_and_result(isolated_log):
    agent = _EchoAgent()
    agent.log_path = os.path.join(isolated_log, f"{agent.name}.jsonl")
    result = agent.run(value=42)

    assert result["status"] == "done"
    assert result["echoed"] == 42
    assert result["agent"] == agent.name
    assert result["run_count"] == 1
    assert agent.status == "done"
    assert agent.last_result == result


def test_base_agent_run_error_path_does_not_raise(isolated_log):
    agent = _EchoAgent(fail=True)
    agent.log_path = os.path.join(isolated_log, f"{agent.name}.jsonl")
    result = agent.run(value=1)

    # the whole point of run()'s try/except -- a failing agent reports an
    # error result instead of crashing whatever called it
    assert result["status"] == "error"
    assert "deliberate failure" in result["error"]
    assert agent.status == "error"


def test_base_agent_run_count_increments_across_calls(isolated_log):
    agent = _EchoAgent()
    agent.log_path = os.path.join(isolated_log, f"{agent.name}.jsonl")
    agent.run(value=1)
    agent.run(value=2)
    result = agent.run(value=3)
    assert result["run_count"] == 3
    assert agent.run_count == 3


def test_base_agent_get_history_returns_last_n_and_skips_corrupt_lines(isolated_log):
    agent = _EchoAgent()
    log_path = os.path.join(isolated_log, f"{agent.name}.jsonl")
    agent.log_path = log_path
    for i in range(5):
        agent.run(value=i)
    # inject a corrupt line -- get_history() must not crash on it
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("not valid json\n")

    history = agent.get_history(last_n=3)
    assert len(history) == 3
    assert history[-1]["echoed"] == 4


def test_base_agent_get_history_empty_when_no_log_file(isolated_log):
    agent = _EchoAgent()
    agent.log_path = os.path.join(isolated_log, "never_run.jsonl")
    assert agent.get_history() == []


def test_base_agent_summary_reflects_current_state(isolated_log):
    agent = _EchoAgent()
    agent.log_path = os.path.join(isolated_log, f"{agent.name}.jsonl")
    agent.run(value=1)
    summary = agent.summary()
    assert summary["status"] == "done"
    assert summary["run_count"] == 1
    assert summary["name"] == agent.name


# ---------------------------------------------------------------------------
# verification_agent.py -- annotation QA thresholds (pure methods, no
# model loading -- neither _confidence_score nor _annotation_flag touch
# self.sess/self.T, so __init__ (which loads a real ONNX model) can be
# bypassed entirely via __new__).
# ---------------------------------------------------------------------------

def _bare_verification_agent():
    return VerificationAgent.__new__(VerificationAgent)


def test_confidence_score_scales_max_prob_to_ten_point_scale():
    agent = _bare_verification_agent()
    probs = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    assert agent._confidence_score(probs) == pytest.approx(9.0)


def test_annotation_flag_ambiguous_when_top_two_scores_are_close():
    agent = _bare_verification_agent()
    # top-2 gap of (0.30-0.29)*10 = 0.1, well under AMBIGUOUS_GAP=2.0
    probs = np.array([0.30, 0.29, 0.15, 0.10, 0.08, 0.05, 0.03])
    assert agent._annotation_flag(probs) == "AMBIGUOUS"


def test_annotation_flag_low_confidence_below_floor():
    agent = _bare_verification_agent()
    # max prob 0.5 < CONFIDENCE_FLOOR (0.65), but gap is large enough to
    # not trip AMBIGUOUS first
    probs = np.array([0.50, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05])
    assert probs.max() < CONFIDENCE_FLOOR
    assert agent._annotation_flag(probs) == "LOW_CONFIDENCE"


def test_annotation_flag_ok_when_confident_and_unambiguous():
    agent = _bare_verification_agent()
    probs = np.array([0.85, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01])
    assert agent._annotation_flag(probs) == "OK"


def test_annotation_flag_conflict_when_human_disagrees_at_high_confidence():
    agent = _bare_verification_agent()
    # model confidently says mel (idx 0), human says nv -- score must
    # clear CONFLICT_SCORE (7.0) for this to fire
    probs = np.array([0.90, 0.03, 0.03, 0.02, 0.01, 0.005, 0.005])
    assert agent._confidence_score(probs) >= CONFLICT_SCORE
    assert agent._annotation_flag(probs, human_label="nv") == "ANNOTATION_CONFLICT"


def test_annotation_flag_no_conflict_when_human_agrees():
    agent = _bare_verification_agent()
    probs = np.array([0.90, 0.03, 0.03, 0.02, 0.01, 0.005, 0.005])
    assert agent._annotation_flag(probs, human_label="mel") == "OK"


# ---------------------------------------------------------------------------
# drift_monitor_agent.py -- KL divergence + severity flags
# ---------------------------------------------------------------------------

@pytest.fixture
def drift_agent(isolated_log):
    import drift_monitor_agent as dm
    agent = DriftMonitorAgent.__new__(DriftMonitorAgent)
    agent.name = "drift_monitor_agent_test"
    agent.log_path = os.path.join(isolated_log, "drift.jsonl")
    import logging
    agent.logger = logging.getLogger("test.drift")
    return agent


def _record(pred_class, confidence_score):
    return {"pred_class": pred_class, "confidence_score": confidence_score}


def test_kl_divergence_is_zero_for_identical_distributions():
    agent = DriftMonitorAgent.__new__(DriftMonitorAgent)
    assert agent._kl_divergence(TRAIN_DIST, TRAIN_DIST) == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_is_positive_for_different_distributions():
    agent = DriftMonitorAgent.__new__(DriftMonitorAgent)
    skewed = {"mel": 0.5, "nv": 0.3, "bcc": 0.1, "akiec": 0.05, "bkl": 0.03, "df": 0.01, "vasc": 0.01}
    assert agent._kl_divergence(skewed, TRAIN_DIST) > 0


def test_drift_run_flags_high_severity_on_large_distribution_shift(drift_agent, tmp_path, monkeypatch):
    import drift_monitor_agent as dm
    monkeypatch.setattr(dm, "OUT_DIR", str(tmp_path))
    # 100 records, all "mel" -- wildly different from TRAIN_DIST's ~9% mel rate
    records = [_record("mel", 9.0) for _ in range(100)]
    result = drift_agent._run(records=records, output_name="test_batch")
    assert result["severity"] == "HIGH"
    assert result["mel_anomaly"] is True
    assert any("DISTRIBUTION_SHIFT" in f for f in result["flags"])


def test_drift_run_reports_ok_when_batch_matches_training_distribution(drift_agent, tmp_path, monkeypatch):
    import drift_monitor_agent as dm
    monkeypatch.setattr(dm, "OUT_DIR", str(tmp_path))
    # build a batch whose class counts closely mirror TRAIN_DIST
    n = 1000
    records = []
    for cls, frac in TRAIN_DIST.items():
        records += [_record(cls, TRAIN_CONF_MEAN * 10) for _ in range(round(n * frac))]
    result = drift_agent._run(records=records, output_name="test_matched_batch")
    assert result["severity"] == "OK"
    assert result["mel_anomaly"] is False


def test_drift_run_handles_empty_records(drift_agent):
    result = drift_agent._run(records=[], output_name="test_empty")
    assert result == {"status": "no_records"}


def test_drift_run_requires_records_or_report_path(drift_agent):
    with pytest.raises(ValueError):
        drift_agent._run(records=None, verification_report_path=None)


# ---------------------------------------------------------------------------
# fairness_monitor_agent.py -- subgroup AUC + disparity gap detection
# ---------------------------------------------------------------------------

@pytest.fixture
def fairness_agent(isolated_log):
    agent = FairnessMonitorAgent.__new__(FairnessMonitorAgent)
    agent.name = "fairness_monitor_agent_test"
    agent.log_path = os.path.join(isolated_log, "fairness.jsonl")
    import logging
    agent.logger = logging.getLogger("test.fairness")
    return agent


def _fairness_records(n_per_group, sex, mel_frac, seed):
    """n_per_group records for one sex, mel_frac of them truly mel with a
    correspondingly higher mel_prob (rank-separated so AUC isn't ~0.5)."""
    rng = np.random.RandomState(seed)
    n_mel = int(n_per_group * mel_frac)
    recs = []
    for i in range(n_per_group):
        is_mel = i < n_mel
        recs.append({
            "human_label": "mel" if is_mel else "nv",
            "mel_prob": float(rng.uniform(0.6, 1.0) if is_mel else rng.uniform(0.0, 0.4)),
            "sex": sex,
        })
    return recs


def test_mel_auc_subset_returns_nan_with_too_few_positive_cases(fairness_agent):
    import pandas as pd
    df = pd.DataFrame([{"human_label": "nv", "mel_prob": 0.1}] * 10 +
                       [{"human_label": "mel", "mel_prob": 0.9}] * 2)  # only 2 positives
    assert np.isnan(fairness_agent._mel_auc_subset(df))


def test_mel_auc_subset_computes_real_auc_with_enough_data(fairness_agent):
    import pandas as pd
    recs = _fairness_records(50, "male", mel_frac=0.3, seed=1)
    df = pd.DataFrame(recs)
    auc = fairness_agent._mel_auc_subset(df)
    assert not np.isnan(auc)
    assert 0.0 <= auc <= 1.0


def test_fairness_run_flags_disparity_when_one_group_has_much_lower_auc(fairness_agent, tmp_path, monkeypatch):
    import fairness_monitor_agent as fm
    monkeypatch.setattr(fm, "OUT_DIR", str(tmp_path))
    # male: well-separated scores (high AUC); female: near-random scores (low AUC)
    male_recs = _fairness_records(60, "male", mel_frac=0.3, seed=1)
    rng = np.random.RandomState(2)
    female_recs = [{"human_label": "mel" if i < 18 else "nv",
                     "mel_prob": float(rng.uniform(0.0, 1.0)), "sex": "female"}
                    for i in range(60)]
    result = fairness_agent._run(records=male_recs + female_recs, output_name="test_disparity")
    assert len(result["disparity_warnings"]) >= 1
    axes_flagged = {w["axis"] for w in result["disparity_warnings"]}
    assert "sex" in axes_flagged


def test_fairness_run_reports_metadata_unavailable_for_missing_axis(fairness_agent, tmp_path, monkeypatch):
    import fairness_monitor_agent as fm
    monkeypatch.setattr(fm, "OUT_DIR", str(tmp_path))
    records = [{"human_label": "mel", "mel_prob": 0.9}]  # no sex/age/site at all
    result = fairness_agent._run(records=records, output_name="test_no_metadata")
    assert result["fairness_results"]["by_sex"]["status"] == "metadata_unavailable"
