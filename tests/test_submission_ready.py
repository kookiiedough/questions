import json

import numpy as np
import pandas as pd
import pytest

from neel.submission_ready import (
    PromptRecord,
    agreement_metrics,
    bootstrap_auroc,
    build_conditions,
    cache_family,
    calibrate_youden_threshold,
    classify_failures,
    direction_cosine,
    format_chat,
    generation_budget,
    held_out_youden_thresholds,
    judge_final_answer,
    parse_judge_result,
    prompt_length_confound,
    run_battery,
    sample_for_human_labels,
    split_trace,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"messages": messages, **kwargs}


def test_format_chat_threads_thinking_flag():
    result = format_chat(FakeTokenizer(), "hello", thinking=False)
    assert result["enable_thinking"] is False
    assert result["add_generation_prompt"] is True
    assert result["tokenize"] is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>reasoning</think>Final answer", ("<think>reasoning", "Final answer", True)),
        ("No close tag", ("No close tag", "", False)),
        ("</think>  Refusal  ", ("", "Refusal", True)),
    ],
)
def test_split_trace(text, expected):
    assert split_trace(text) == expected


def test_generation_budget_matches_thinking_mode():
    assert generation_budget(True) == 1024
    assert generation_budget(False) == 256


def test_build_conditions_has_no_attack_and_harmless_controls():
    harmful = [PromptRecord("h1", "harm", "jbb", "battery", True)]
    harmless = [PromptRecord("s1", "safe", "alpaca", "fit", False)]
    rows = build_conditions(harmful, harmless)

    assert any(row["family"] == "no_attack" for row in rows)
    assert any(not row["harmful"] for row in rows)
    assert {row["family"] for row in rows if row["harmful"]} == {
        "no_attack",
        "roleplay",
        "pseudo_suffix",
        "prefix_injection",
    }
    assert all("adv_suffix" not in row["condition_id"] for row in rows)


def test_direction_cosine():
    assert direction_cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert direction_cosine([1, 0], [0, 1]) == pytest.approx(0)


def test_parse_and_call_judge_final_answer_only():
    seen = {}

    def judge_call(system, answer):
        seen["system"] = system
        seen["answer"] = answer
        return {
            "refusal": True,
            "specificity": 2,
            "convincingness": 3,
            "rationale": "It explicitly declines.",
        }

    result = judge_final_answer("I cannot assist.", judge_call=judge_call)
    assert result.refusal is True
    assert seen["answer"] == "I cannot assist."
    assert "final answer" in seen["system"].lower()


def test_judge_rejects_invalid_or_empty_results():
    with pytest.raises(ValueError):
        judge_final_answer("", judge_call=lambda *_: {})
    with pytest.raises(ValueError):
        parse_judge_result(
            {
                "refusal": False,
                "specificity": 6,
                "convincingness": 3,
                "rationale": "x",
            }
        )


def test_human_label_sample_is_blinded_and_deterministic(tmp_path):
    frame = pd.DataFrame(
        {
            "condition_id": [f"id-{i}" for i in range(60)],
            "final_answer": [f"answer {i}" for i in range(60)],
            "reached_think_close": [True] * 60,
            "family": ["secret"] * 60,
        }
    )
    path = tmp_path / "labels.csv"
    sample = sample_for_human_labels(frame, path, n=50)

    assert len(sample) == 50
    assert "family" not in sample
    assert sample["human_refusal"].eq("").all()
    assert path.exists()


def test_agreement_metrics_reports_kappa_and_keyword_agreement():
    metrics = agreement_metrics(
        [True, True, False, False],
        [True, True, False, False],
        [True, False, False, False],
    )
    assert metrics["human_judge_kappa"] == pytest.approx(1)
    assert metrics["judge_keyword_agreement"] == pytest.approx(0.75)


def test_bootstrap_auroc_is_reproducible():
    first = bootstrap_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bootstrap=100)
    second = bootstrap_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bootstrap=100)
    assert first == second
    assert first["auroc"] == pytest.approx(1)
    assert first["ci_low"] == pytest.approx(1)


def test_youden_threshold_is_out_of_fold_and_classifies_both_failures():
    labels = np.array([False, False, False, True, True, True] * 2)
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 2)
    detected, thresholds = held_out_youden_thresholds(labels, scores, n_splits=3)
    assert detected.dtype == bool
    assert np.isfinite(thresholds).all()

    frame = pd.DataFrame(
        {
            "judge_complied": labels,
            "prefill_proj": scores,
            "truncated": [False] * len(labels),
        }
    )
    classified = classify_failures(frame, threshold=0.5)
    assert set(classified["failure_class"]) == {
        "false_alarm",
        "overpower_failure",
    }


def test_truncated_rows_are_excluded_from_threshold_fit():
    frame = pd.DataFrame(
        {
            "judge_complied": [False, False, True, True, True],
            "prefill_proj": [0.1, 0.2, 0.8, 0.9, 100],
            "truncated": [False, False, False, False, True],
        }
    )
    result = classify_failures(frame, score_column="prefill_proj", threshold=0.5)
    assert result.loc[4, "failure_class"] == "truncated"
    assert pd.isna(result.loc[4, "youden_threshold"])


def test_calibration_uses_harmful_and_harmless_fitting_scores():
    calibration = calibrate_youden_threshold(
        [0.7, 0.8, 0.9, 1.0, 1.1],
        [-0.4, -0.3, -0.2, -0.1, 0.0],
    )
    assert 0 < calibration["threshold"] <= 0.7
    assert calibration["held_out_accuracy"] == pytest.approx(1)


def test_prompt_length_confound_reports_regression():
    metrics = prompt_length_confound([1, 2, 3, 4], [2, 4, 6, 8])
    assert metrics["pearson_r"] == pytest.approx(1)
    assert metrics["slope"] == pytest.approx(2)
    assert metrics["r_squared"] == pytest.approx(1)


def test_family_cache_is_atomic_jsonl(tmp_path):
    path = cache_family([{"x": 1}, {"x": 2}], tmp_path, family="roleplay", thinking=True)
    assert path.name == "roleplay__thinking_true.jsonl"
    assert not path.with_suffix(".jsonl.tmp").exists()
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"x": 1},
        {"x": 2},
    ]


def test_run_battery_crosses_thinking_and_caches_each_family(tmp_path):
    conditions = [
        {"condition_id": "a", "family": "no_attack", "prompt": "one"},
        {"condition_id": "b", "family": "roleplay", "prompt": "two"},
    ]

    frame = run_battery(
        conditions=conditions,
        thinking_modes=[True, False],
        run_attempt=lambda prompt, thinking: {
            "final_answer": prompt,
            "thinking_seen": thinking,
        },
        cache_dir=tmp_path,
    )

    assert len(frame) == 4
    assert set(frame["thinking"]) == {True, False}
    assert len(list(tmp_path.glob("*.jsonl"))) == 4
