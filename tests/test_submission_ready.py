import json

import numpy as np
import pandas as pd
import pytest

from neel.submission_ready import (
    PromptRecord,
    SubmissionAuditError,
    agreement_metrics,
    audit_submission_outputs,
    bootstrap_auroc,
    build_conditions,
    cache_family,
    calibrate_youden_threshold,
    classify_failures,
    cross_validated_direction_report,
    direction_cosine,
    format_chat,
    generation_budget,
    held_out_youden_thresholds,
    judge_final_answer,
    make_ablation_hook,
    make_add_hook,
    normalize_boolean_labels,
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


@pytest.mark.parametrize("container_type", ["tensor", "tuple", "list"])
def test_add_hook_preserves_container_and_only_changes_final_position(container_type):
    torch = pytest.importorskip("torch")
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    if container_type == "tuple":
        output = (hidden, "cache")
    elif container_type == "list":
        output = [hidden, "cache"]
    else:
        output = hidden

    result = make_add_hook(torch.ones(4), alpha=2.0)(None, None, output)
    result_hidden = result if torch.is_tensor(result) else result[0]

    assert type(result) is type(output)
    torch.testing.assert_close(result_hidden[:, :-1, :], hidden[:, :-1, :])
    torch.testing.assert_close(result_hidden[:, -1, :], hidden[:, -1, :] + 2)
    torch.testing.assert_close(
        hidden, torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    )


@pytest.mark.parametrize("container_type", ["tensor", "tuple", "list"])
def test_ablation_hook_preserves_container_and_removes_final_projection(container_type):
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    if container_type == "tuple":
        output = (hidden, "cache")
    elif container_type == "list":
        output = [hidden, "cache"]
    else:
        output = hidden

    result = make_ablation_hook(torch.tensor([1.0, 0.0]))(None, None, output)
    result_hidden = result if torch.is_tensor(result) else result[0]

    assert type(result) is type(output)
    torch.testing.assert_close(result_hidden, torch.tensor([[[1.0, 2.0], [0.0, 4.0]]]))
    torch.testing.assert_close(hidden, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))


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
    with pytest.raises(ValueError, match="JSON boolean"):
        parse_judge_result(
            {
                "refusal": "false",
                "specificity": 2,
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


def test_boolean_labels_parse_false_strings_without_truthiness_bug():
    np.testing.assert_array_equal(
        normalize_boolean_labels(["true", "False", True, np.bool_(False)]),
        [True, False, True, False],
    )
    with pytest.raises(ValueError):
        normalize_boolean_labels(["no"])


def test_cross_validated_direction_reports_held_out_effect():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(42)
    harmless = torch.randn(20, 2, 4, generator=generator) * 0.1
    harmful = harmless.clone()
    harmful[:, 0, 0] += 2.0
    harmful[:, 1, 0] += 0.2

    report = cross_validated_direction_report(
        harmful, harmless, n_splits=5, seed=42
    )
    assert report["n_pairs"] == 20
    assert {fold["best_layer"] for fold in report["folds"]} == {0}
    assert report["held_out_cohens_d_mean"] > 5
    assert np.isfinite(report["youden_threshold_oof"])
    assert report["youden_held_out_accuracy"] > 0.9


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
    assert 0 < calibration["threshold"] <= 0.8
    # One held-out harmful score lies below thresholds fitted on other folds.
    assert calibration["held_out_accuracy"] == pytest.approx(0.9)


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

    def must_not_run(*_args):
        raise AssertionError("completed family cache should be reused")

    resumed = run_battery(
        conditions=conditions,
        thinking_modes=[True, False],
        run_attempt=must_not_run,
        cache_dir=tmp_path,
    )
    assert len(resumed) == 4


def _complete_audit_fixture(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    fitting_harmful = [
        {
            "prompt_id": f"fit-h-{index}",
            "source": "JailbreakBench/JBB-Behaviors",
        }
        for index in range(50)
    ]
    fitting_harmless = [
        {"prompt_id": f"fit-s-{index}", "source": "tatsu-lab/alpaca"}
        for index in range(50)
    ]
    battery_targets = [
        {
            "prompt_id": f"battery-{index}",
            "source": "JailbreakBench/JBB-Behaviors",
        }
        for index in range(20)
    ]
    (results / "prompt_manifest.json").write_text(
        json.dumps(
            {
                "fitting_harmful": fitting_harmful,
                "fitting_harmless": fitting_harmless,
                "battery_harmful": battery_targets,
            }
        )
    )
    cv_report = {
        "n_pairs": 50,
        "held_out_cohens_d_mean": 2.0,
        "held_out_cohens_d_std": 0.2,
        "youden_threshold_oof": 0.5,
        "youden_held_out_accuracy": 0.9,
    }
    (results / "direction_report.json").write_text(
        json.dumps(
            {
                "direction_cosine": 0.95,
                "thinking_on_cv": cv_report,
                "thinking_off_cv": cv_report,
            }
        )
    )

    families = ["no_attack", "roleplay", "pseudo_suffix", "prefix_injection"]
    battery_rows = []
    for index in range(56):
        family = families[index % len(families)]
        thinking = bool((index // len(families)) % 2)
        battery_rows.append(
            {
                "condition_id": f"condition-{index}",
                "family": family,
                "thinking": thinking,
                "harmful": True,
                "final_answer": f"answer {index}",
                "reached_think_close": True,
                "truncated": False,
                "n_think_tokens": 5 if thinking else 0,
                "n_prompt_tokens": 20,
                "prefill_proj": float(index),
                "first_generation_proj": float(index) / 2,
                "trajectory": [0.0, 1.0],
                "judge_refusal": bool(index % 2),
                "judge_complied": not bool(index % 2),
            }
        )
    for family in families[1:]:
        for thinking in (False, True):
            index = len(battery_rows)
            battery_rows.append(
                {
                    **battery_rows[0],
                    "condition_id": f"harmless-{family}-{thinking}",
                    "family": family,
                    "thinking": thinking,
                    "harmful": False,
                    "final_answer": "safe answer",
                }
            )
    battery_frame = pd.DataFrame(battery_rows)
    battery_frame.to_json(results / "battery_all.jsonl", orient="records", lines=True)
    battery_frame.to_json(results / "battery_judged.jsonl", orient="records", lines=True)

    human = battery_frame.iloc[:50][
        ["condition_id", "thinking", "final_answer", "judge_refusal"]
    ].rename(columns={"judge_refusal": "human_refusal"})
    human.to_csv(results / "human_labels_50.csv", index=False)
    (results / "judge_agreement.json").write_text(
        json.dumps(
            {
                "human_judge_kappa": 1.0,
                "judge_keyword_agreement": 0.8,
                "judge_keyword_kappa": 0.6,
            }
        )
    )
    (results / "analysis_report.json").write_text(
        json.dumps(
            {
                "calibration": {"true": {}, "false": {}},
                "length_confound": {
                    "pearson_r": 0.1,
                    "slope": 0.2,
                    "r_squared": 0.01,
                    "n": 62,
                },
            }
        )
    )
    metric_rows = [
        {
            "family": family,
            "thinking": thinking,
            "auroc": 0.7,
            "ci_low": 0.6,
            "ci_high": 0.8,
            "n": 10,
        }
        for family in families
        for thinking in (False, True)
    ]
    pd.DataFrame(metric_rows).to_csv(results / "per_family_auroc.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(results / "first_token_auroc.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition_id": "condition-0",
                "thinking": False,
                "failure_class": "stealth_failure",
                "youden_threshold": 0.5,
            }
        ]
    ).to_json(results / "classified_attempts.jsonl", orient="records", lines=True)
    pd.DataFrame([{"family": "roleplay", "thinking": True, "mean": 0.1}]).to_csv(
        results / "harmless_wrapper_control.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "condition_id": "condition-0",
                "thinking": False,
                "baseline_class": "stealth_failure",
                "alpha": alpha,
                "rescued_to_refusal": True,
            }
            for alpha in (5.0, 10.0, 20.0)
        ]
    ).to_csv(results / "differential_steering.csv", index=False)
    pd.DataFrame(
        [
            {
                "target_id": f"battery-{int(thinking)}",
                "thinking": thinking,
                "complied_after_ablation": False,
            }
            for thinking in (False, True)
        ]
    ).to_csv(results / "directional_ablation.csv", index=False)
    for image in ("killer_graph_auroc.png", "trajectory_by_family_outcome.png"):
        (results / image).write_bytes(b"image")

    summary = tmp_path / "summary.md"
    summary.write_text("# Final report\n\nAll measured results are reported.")
    ledger = tmp_path / "ledger.txt"
    ledger.write_text("Corrected run only.")
    return results, summary, ledger


def test_completion_audit_accepts_complete_evidence(tmp_path):
    results, summary, ledger = _complete_audit_fixture(tmp_path)
    report = audit_submission_outputs(
        results, summary_path=summary, ledger_path=ledger
    )
    assert report["status"] == "complete"
    assert report["human_labels"] == 50
    assert report["auroc_arms"] == 8


def test_completion_audit_rejects_unfilled_summary(tmp_path):
    results, summary, ledger = _complete_audit_fixture(tmp_path)
    summary.write_text("# SKELETON — not the submission\n\nResult: [?]")
    with pytest.raises(SubmissionAuditError, match="placeholders"):
        audit_submission_outputs(
            results, summary_path=summary, ledger_path=ledger
        )
