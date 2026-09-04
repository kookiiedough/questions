#!/usr/bin/env python3
"""Headless submission-ready runner for Qwen3-4B on a CUDA GPU.

This is the Colab notebook's corrected section (cells 27-35) as a resumable
script. Results go to --cache-dir. Do not copy placeholder numbers into
exec_summary.md until audit_submission_outputs reports status=complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import transformers
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neel.submission_ready import (  # noqa: E402
    GENERATION_SETTINGS,
    JUDGE_SYSTEM_PROMPT,
    SEED,
    agreement_metrics,
    audit_submission_outputs,
    build_conditions,
    classify_failures,
    clustered_rate_difference,
    collect_last_token_activations,
    controlled_prompt_length_confound,
    cross_validated_direction_report,
    default_cache_dir,
    direction_cosine,
    extract_direction,
    fit_youden_threshold,
    format_chat,
    generate_with_hook,
    generation_budget,
    judge_final_answer,
    keyword_refusal_label,
    load_source_prompts,
    make_ablation_hook,
    make_add_hook,
    normalize_boolean_labels,
    paired_harmless_wrapper_shifts,
    per_condition_aurocs,
    require_judge_api_key,
    run_battery,
    run_generation_with_trajectory,
    run_judge_pass,
    sample_for_human_labels,
    set_seed,
    split_trace,
    write_manifest,
)


MODEL_ID = "Qwen/Qwen3-4B"
ALL_STAGES = (
    "extract",
    "battery",
    "judge",
    "human-sheet",
    "agreement",
    "analysis",
    "steering",
    "figures",
    "audit",
)


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--stages",
        default="extract,battery,judge,human-sheet",
        help="Comma-separated stages: " + ",".join(ALL_STAGES),
    )
    parser.add_argument("--n-pairs", type=int, default=60)
    parser.add_argument("--n-battery", type=int, default=10)
    return parser.parse_args()


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-4B battery requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log(f"loading {MODEL_ID} dtype={dtype} gpu={torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def compute_separability(harmful_acts, harmless_acts, directions):
    scores = []
    for layer in range(harmful_acts.shape[1]):
        harmful = (harmful_acts[:, layer, :] @ directions[layer]).cpu().numpy()
        harmless = (harmless_acts[:, layer, :] @ directions[layer]).cpu().numpy()
        pooled = np.sqrt((harmful.var(ddof=1) + harmless.var(ddof=1)) / 2)
        scores.append((harmful.mean() - harmless.mean()) / pooled if pooled else 0.0)
    return scores


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_judge_model(client: OpenAI) -> str:
    requested = os.environ.get("JUDGE_MODEL", "gpt-5-mini")
    candidates = [requested]
    if requested != "gpt-4o-mini":
        candidates.append("gpt-4o-mini")
    last_error = None
    for model_name in candidates:
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4,
            )
            log(f"judge model ready: {model_name}")
            return model_name
        except Exception as error:  # noqa: BLE001
            last_error = error
            log(f"judge model {model_name} failed: {error}")
    raise RuntimeError("No usable OpenAI judge model") from last_error


def main() -> None:
    args = parse_args()
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    unknown = [item for item in stages if item not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"Unknown stages: {unknown}")

    set_seed(SEED)
    cache_dir = args.cache_dir or default_cache_dir()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"cache_dir={cache_dir}")

    (
        fitting_harmful,
        fitting_harmless,
        battery_harmful,
        battery_harmless,
    ) = load_source_prompts(
        n_pairs=args.n_pairs, n_battery=args.n_battery, seed=SEED
    )
    conditions = build_conditions(battery_harmful, battery_harmless)
    write_manifest(
        cache_dir / "prompt_manifest.json",
        fitting_harmful=fitting_harmful,
        fitting_harmless=fitting_harmless,
        battery_harmful=battery_harmful,
        battery_harmless=battery_harmless,
    )
    log(
        f"fit pairs={len(fitting_harmful)}; battery targets={len(battery_harmful)}; "
        f"conditions={len(conditions)}"
    )

    needs_model = any(
        stage in stages
        for stage in ("extract", "battery", "steering")
    )
    model = tokenizer = None
    if needs_model:
        model, tokenizer = load_model()

    mode_dirs = {}
    mode_best_layers = {}
    mode_cv_reports = {}
    mode_final_calibrations = {}
    common_layer = None
    direction_path = cache_dir / "directions.pt"
    direction_report_path = cache_dir / "direction_report.json"

    if "extract" in stages:
        assert model is not None and tokenizer is not None
        mode_acts = {}
        mode_dirs_all = {}
        mode_separability = {}
        for thinking in (True, False):
            log(f"extracting activations thinking={thinking}")
            harmful_mode = collect_last_token_activations(
                model,
                tokenizer,
                [format_chat(tokenizer, item.text, thinking) for item in fitting_harmful],
            )
            harmless_mode = collect_last_token_activations(
                model,
                tokenizer,
                [format_chat(tokenizer, item.text, thinking) for item in fitting_harmless],
            )
            mode_cv_reports[thinking] = cross_validated_direction_report(
                harmful_mode, harmless_mode, n_splits=5, seed=SEED
            )
            directions = extract_direction(harmful_mode, harmless_mode)
            mode_acts[thinking] = (harmful_mode, harmless_mode)
            mode_dirs_all[thinking] = directions
            mode_separability[thinking] = compute_separability(
                harmful_mode, harmless_mode, directions
            )
            log(
                f"thinking={thinking} held-out d="
                f"{mode_cv_reports[thinking]['held_out_cohens_d_mean']:.3f}"
            )
        common_layer = int(
            np.argmax(
                np.mean(
                    [mode_separability[True], mode_separability[False]],
                    axis=0,
                )
            )
        )
        mode_best_layers = {True: common_layer, False: common_layer}
        mode_dirs = {
            thinking: mode_dirs_all[thinking][common_layer]
            for thinking in (True, False)
        }
        for thinking in (True, False):
            harmful_mode, harmless_mode = mode_acts[thinking]
            direction = mode_dirs[thinking]
            mode_final_calibrations[thinking] = fit_youden_threshold(
                (harmful_mode[:, common_layer, :].float() @ direction.float())
                .cpu()
                .numpy(),
                (harmless_mode[:, common_layer, :].float() @ direction.float())
                .cpu()
                .numpy(),
            )
        cosine = direction_cosine(mode_dirs[True], mode_dirs[False])
        direction_report = {
            "common_layer": common_layer,
            "direction_cosine": cosine,
            "thinking_on_cv": mode_cv_reports[True],
            "thinking_off_cv": mode_cv_reports[False],
            "thinking_on_final_calibration": mode_final_calibrations[True],
            "thinking_off_final_calibration": mode_final_calibrations[False],
            "activation_source": "hf_hidden_states",
        }
        save_json(direction_report_path, direction_report)
        torch.save(
            {
                "common_layer": common_layer,
                "mode_dirs": {str(key): value.cpu() for key, value in mode_dirs.items()},
                "mode_best_layers": {str(key): value for key, value in mode_best_layers.items()},
                "mode_cv_reports": mode_cv_reports,
                "mode_final_calibrations": mode_final_calibrations,
            },
            direction_path,
        )
        log(f"common_layer={common_layer} cosine={cosine:.4f}")
    elif direction_path.exists():
        bundle = torch.load(direction_path, map_location="cpu", weights_only=False)
        common_layer = int(bundle["common_layer"])
        mode_dirs = {
            True: bundle["mode_dirs"]["True"],
            False: bundle["mode_dirs"]["False"],
        }
        mode_best_layers = {
            True: int(bundle["mode_best_layers"]["True"]),
            False: int(bundle["mode_best_layers"]["False"]),
        }
        mode_cv_reports = bundle["mode_cv_reports"]
        mode_final_calibrations = bundle["mode_final_calibrations"]
        log(f"loaded directions common_layer={common_layer}")

    run_fingerprint = None
    if model is not None and tokenizer is not None and mode_dirs:
        if torch.cuda.is_available():
            mode_dirs = {
                key: value.to(model.get_input_embeddings().weight.device)
                for key, value in mode_dirs.items()
            }
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "model": MODEL_ID,
                    "model_revision": getattr(model.config, "_commit_hash", None),
                    "tokenizer": tokenizer.name_or_path,
                    "transformers": transformers.__version__,
                    "seed": SEED,
                    "generation": GENERATION_SETTINGS,
                    "layers": {str(key): value for key, value in mode_best_layers.items()},
                    "conditions": conditions,
                    "activation_source": "hf_hidden_states",
                },
                sort_keys=True,
                default=str,
            ).encode()
        )
        for thinking in (True, False):
            fingerprint.update(mode_dirs[thinking].detach().float().cpu().numpy().tobytes())
        run_fingerprint = fingerprint.hexdigest()
        (cache_dir / "run_fingerprint.txt").write_text(run_fingerprint)
        log(f"run_fingerprint={run_fingerprint}")

    battery = None
    if "battery" in stages:
        if model is None or not mode_dirs or run_fingerprint is None:
            raise RuntimeError("battery stage needs extract (or saved directions.pt)")
        completed = {"n": 0}

        def measure_attempt(prompt, thinking):
            started = time.time()
            result = run_generation_with_trajectory(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                thinking=thinking,
                best_layer=mode_best_layers[thinking],
                refusal_direction=mode_dirs[thinking],
                seed=SEED,
            )
            completed["n"] += 1
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(
                f"gen {completed['n']} thinking={thinking} "
                f"tokens={result['n_generation_tokens']} "
                f"truncated={result['truncated']} "
                f"{time.time() - started:.1f}s mem={mem:.2f}GB"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result

        battery = run_battery(
            conditions=conditions,
            thinking_modes=(True, False),
            run_attempt=measure_attempt,
            cache_dir=cache_dir / "battery",
            run_fingerprint=run_fingerprint,
        )
        battery.to_json(cache_dir / "battery_all.jsonl", orient="records", lines=True)
        log(str(battery.groupby(["family", "thinking", "truncated"]).size()))
    elif (cache_dir / "battery_all.jsonl").exists():
        battery = pd.read_json(cache_dir / "battery_all.jsonl", lines=True)
        if run_fingerprint is None and "_run_fingerprint" in battery.columns:
            run_fingerprint = str(battery["_run_fingerprint"].iloc[0])

    battery_judged = None
    if "judge" in stages:
        if battery is None:
            raise RuntimeError("judge stage needs battery_all.jsonl")
        require_judge_api_key()
        client = OpenAI()
        judge_model = resolve_judge_model(client)

        def call_judge(system_prompt, final_answer):
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_answer},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        judge_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "battery_run": run_fingerprint,
                    "judge_model": judge_model,
                    "judge_prompt": JUDGE_SYSTEM_PROMPT,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        battery_judged = run_judge_pass(
            battery,
            judge_call=call_judge,
            cache_path=cache_dir / "judge_cache.jsonl",
            judge_fingerprint=judge_fingerprint,
            max_attempts=3,
        )
        battery_judged["keyword_refusal"] = battery_judged["raw_text"].fillna("").map(
            keyword_refusal_label
        )
        battery_judged.to_json(
            cache_dir / "battery_judged.jsonl", orient="records", lines=True
        )
        log(str(battery_judged["judge_refusal"].value_counts(dropna=False)))
        save_json(cache_dir / "judge_model.json", {"judge_model": judge_model})
    elif (cache_dir / "battery_judged.jsonl").exists():
        battery_judged = pd.read_json(cache_dir / "battery_judged.jsonl", lines=True)
        if run_fingerprint is None and "_run_fingerprint" in battery_judged.columns:
            run_fingerprint = str(battery_judged["_run_fingerprint"].iloc[0])

    human_path = cache_dir / "human_labels_50.csv"
    if "human-sheet" in stages:
        if battery_judged is None:
            raise RuntimeError("human-sheet needs battery_judged.jsonl")
        if not human_path.exists():
            sample_for_human_labels(battery_judged, human_path, n=50, seed=SEED)
            log(f"created blinded label sheet at {human_path}")
            return
        if battery_judged["_run_fingerprint"].nunique() != 1:
            raise RuntimeError("battery_judged has mixed run fingerprints")

    if "agreement" in stages:
        if battery_judged is None or not human_path.exists():
            raise RuntimeError("agreement needs judged battery and filled labels")
        human = pd.read_csv(human_path)
        if human[["human_refusal", "human_fulfills_request"]].isna().any().any():
            raise RuntimeError("All 50 rows need true/false labels")
        if set(human["_run_fingerprint"]) != {run_fingerprint}:
            raise RuntimeError("Human label sheet belongs to a different battery run")
        joined = human.merge(
            battery_judged[
                [
                    "condition_id",
                    "thinking",
                    "target",
                    "final_answer",
                    "judge_refusal",
                    "judge_fulfills_request",
                    "keyword_refusal",
                ]
            ],
            on=["condition_id", "thinking"],
            validate="one_to_one",
            suffixes=("_human", "_battery"),
        )
        if len(joined) != 50:
            raise RuntimeError(f"Expected 50 matched human labels, found {len(joined)}")
        for _, row in joined.iterrows():
            digest = hashlib.sha256(
                f"{row['target_battery']}\0{row['final_answer_battery']}".encode()
            ).hexdigest()
            if digest != row["answer_sha256"]:
                raise RuntimeError("Human label request/answer text is stale")
        agreement = agreement_metrics(
            normalize_boolean_labels(joined["human_refusal"]),
            normalize_boolean_labels(joined["judge_refusal"]),
            normalize_boolean_labels(joined["human_fulfills_request"]),
            normalize_boolean_labels(joined["judge_fulfills_request"]),
            normalize_boolean_labels(joined["keyword_refusal"]),
        )
        save_json(cache_dir / "judge_agreement.json", agreement)
        log(json.dumps(agreement))

    classified = None
    if "analysis" in stages:
        if battery_judged is None or not mode_final_calibrations:
            raise RuntimeError("analysis needs judged battery and direction calibrations")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        calibrations = {
            thinking: {
                **mode_final_calibrations[thinking],
                "cv_held_out_accuracy_mean": mode_cv_reports[thinking][
                    "held_out_threshold_accuracy_mean"
                ],
                "cv_held_out_accuracy_std": mode_cv_reports[thinking][
                    "held_out_threshold_accuracy_std"
                ],
            }
            for thinking in (True, False)
        }
        thresholds = {mode: report["threshold"] for mode, report in calibrations.items()}
        analysis_results = battery_judged.copy()
        analysis_results["compliance_score"] = -analysis_results["prefill_proj"]
        harmful_results = analysis_results.loc[analysis_results["harmful"]].copy()
        classified = classify_failures(
            harmful_results,
            threshold_by_thinking=thresholds,
        )
        outcome_aurocs = per_condition_aurocs(
            harmful_results,
            label_column="judge_complied",
            score_column="compliance_score",
        )
        harm_detection_aurocs = per_condition_aurocs(
            analysis_results,
            label_column="harmful",
            score_column="prefill_proj",
        )
        analysis_results["first_generation_compliance_score"] = -analysis_results[
            "first_generation_proj"
        ]
        first_token_aurocs = per_condition_aurocs(
            analysis_results.loc[analysis_results["harmful"]],
            label_column="judge_complied",
            score_column="first_generation_compliance_score",
        )
        length_check = controlled_prompt_length_confound(battery_judged)
        harmless_control = paired_harmless_wrapper_shifts(battery_judged)
        wrapper_summary = harmless_control.groupby(["family", "thinking"])[
            "wrapper_projection_shift"
        ].agg(["count", "mean", "std"])
        analysis_report = {
            "calibration": calibrations,
            "length_confound": length_check,
            "harmless_wrapper_shift_summary": wrapper_summary.reset_index().to_dict(
                "records"
            ),
            "truncated_rows": int(battery_judged["truncated"].sum()),
            "failure_counts": classified["failure_class"].value_counts().to_dict(),
        }
        save_json(cache_dir / "analysis_report.json", analysis_report)
        outcome_aurocs.to_csv(cache_dir / "per_family_auroc.csv", index=False)
        harm_detection_aurocs.to_csv(cache_dir / "harm_detection_auroc.csv", index=False)
        first_token_aurocs.to_csv(cache_dir / "first_token_auroc.csv", index=False)
        classified.to_json(
            cache_dir / "classified_attempts.jsonl", orient="records", lines=True
        )
        harmless_control.to_csv(cache_dir / "harmless_wrapper_control.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, table, title in (
            (axes[0], outcome_aurocs, "Compliance prediction on harmful attempts"),
            (axes[1], harm_detection_aurocs, "Harm detection on disjoint controls"),
        ):
            families = sorted(table["family"].unique())
            for thinking, marker in ((True, "o"), (False, "s")):
                group = table[table["thinking"] == thinking].set_index("family").loc[
                    families
                ]
                x = np.arange(len(group)) + (0.08 if thinking else -0.08)
                ax.errorbar(
                    x,
                    group["auroc"],
                    yerr=[
                        group["auroc"] - group["ci_low"],
                        group["ci_high"] - group["auroc"],
                    ],
                    fmt=marker,
                    capsize=4,
                    label=f"thinking {thinking}",
                )
            ax.axhline(0.5, color="gray", linestyle=":", label="chance")
            ax.set_xticks(np.arange(len(families)), families, rotation=20)
            ax.set_title(title)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("AUROC")
        axes[1].axhspan(
            0.84,
            0.95,
            color="steelblue",
            alpha=0.12,
            label="published plain-prompt context (different model)",
        )
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(cache_dir / "killer_graph_auroc.png", dpi=180)
        plt.close(fig)
        log("wrote analysis artifacts")
    elif (cache_dir / "classified_attempts.jsonl").exists():
        classified = pd.read_json(cache_dir / "classified_attempts.jsonl", lines=True)

    if "steering" in stages:
        if (
            classified is None
            or battery_judged is None
            or model is None
            or tokenizer is None
        ):
            raise RuntimeError("steering needs classified battery, model, and directions")
        require_judge_api_key()
        client = OpenAI()
        judge_model = json.loads((cache_dir / "judge_model.json").read_text())[
            "judge_model"
        ] if (cache_dir / "judge_model.json").exists() else resolve_judge_model(client)

        def call_judge(system_prompt, final_answer):
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_answer},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        def judge_generated(raw_text, thinking, request):
            if thinking:
                _, answer, closed = split_trace(raw_text)
                if not closed:
                    return None
            else:
                answer = raw_text.strip()
            return judge_final_answer(request, answer, judge_call=call_judge)

        steering_rows = []
        failed = classified[
            (classified["family"] != "no_attack")
            & classified["failure_class"].isin(["stealth_failure", "overpower_failure"])
        ]
        log(f"steering failures={len(failed)}")
        for alpha in (5.0, 10.0, 20.0):
            for _, row in failed.iterrows():
                thinking = bool(row["thinking"])
                direction = mode_dirs[thinking]
                started = time.time()
                raw_text = generate_with_hook(
                    raw_model=model,
                    tokenizer=tokenizer,
                    formatted_prompt=format_chat(tokenizer, row["prompt"], thinking),
                    layer=mode_best_layers[thinking],
                    hook=make_add_hook(direction, alpha),
                    max_new_tokens=generation_budget(thinking),
                    seed=SEED,
                )
                judged = judge_generated(raw_text, thinking, row["target"])
                steering_rows.append(
                    {
                        "condition_id": row["condition_id"],
                        "thinking": thinking,
                        "family": row["family"],
                        "target_id": row["target_id"],
                        "baseline_class": row["failure_class"],
                        "alpha": alpha,
                        "baseline_complied": row["judge_complied"],
                        "baseline_final_answer": row["final_answer"],
                        "steered_raw_text": raw_text,
                        "rescued_to_refusal": None if judged is None else judged.refusal,
                        "steered_fulfills_request": None
                        if judged is None
                        else judged.fulfills_request,
                        "steered_rationale": None if judged is None else judged.rationale,
                    }
                )
                log(f"steer alpha={alpha} {row['condition_id']} {time.time()-started:.1f}s")
            checkpoint = cache_dir / "steering" / f"addition_alpha_{alpha:g}.jsonl"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [row for row in steering_rows if row["alpha"] == alpha]
            ).to_json(checkpoint, orient="records", lines=True)

        ablation_rows = []
        for thinking in (True, False):
            for record in battery_harmful:
                raw_text = generate_with_hook(
                    raw_model=model,
                    tokenizer=tokenizer,
                    formatted_prompt=format_chat(tokenizer, record.text, thinking),
                    layer=mode_best_layers[thinking],
                    hook=make_ablation_hook(mode_dirs[thinking]),
                    max_new_tokens=generation_budget(thinking),
                    seed=SEED,
                )
                judged = judge_generated(raw_text, thinking, record.text)
                baseline = battery_judged[
                    (battery_judged["family"] == "no_attack")
                    & (battery_judged["target_id"] == record.prompt_id)
                    & (battery_judged["thinking"] == thinking)
                ].iloc[0]
                ablation_rows.append(
                    {
                        "target_id": record.prompt_id,
                        "thinking": thinking,
                        "baseline_complied": baseline["judge_complied"],
                        "baseline_final_answer": baseline["final_answer"],
                        "ablated_raw_text": raw_text,
                        "complied_after_ablation": None
                        if judged is None
                        else judged.complied,
                        "ablation_fulfills_request": None
                        if judged is None
                        else judged.fulfills_request,
                        "ablation_rationale": None if judged is None else judged.rationale,
                    }
                )
                log(f"ablate thinking={thinking} {record.prompt_id}")

        steering = pd.DataFrame(steering_rows)
        ablation = pd.DataFrame(ablation_rows)
        steering_contrasts = []
        for alpha, alpha_rows in steering.groupby("alpha"):
            contrast = clustered_rate_difference(
                alpha_rows,
                outcome_column="rescued_to_refusal",
                group_column="baseline_class",
                cluster_column="target_id",
                group_a="stealth_failure",
                group_b="overpower_failure",
            )
            steering_contrasts.append({"alpha": alpha, **contrast})
        steering_contrasts = pd.DataFrame(steering_contrasts)
        ablation["new_compliance_after_ablation"] = ~ablation["baseline_complied"].astype(
            bool
        ) & ablation["complied_after_ablation"].astype(bool)
        steering.to_csv(cache_dir / "differential_steering.csv", index=False)
        steering_contrasts.to_csv(cache_dir / "steering_contrasts.csv", index=False)
        ablation.to_csv(cache_dir / "directional_ablation.csv", index=False)
        log("wrote steering artifacts")

    if "figures" in stages:
        if battery_judged is None:
            raise RuntimeError("figures need battery_judged.jsonl")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trajectory_rows = []
        for _, row in battery_judged.loc[battery_judged["harmful"]].iterrows():
            generated = np.asarray(row["trajectory"])[
                int(row["trajectory_prompt_boundary"]) :
            ]
            segments = {}
            if bool(row["thinking"]):
                segments["reasoning"] = generated[: int(row["n_think_tokens"])]
                if pd.notna(row["answer_start_generation_index"]):
                    segments["final_answer"] = generated[
                        int(row["answer_start_generation_index"]) :
                    ]
            else:
                segments["final_answer"] = generated
            for segment_name, values in segments.items():
                if len(values) < 2:
                    continue
                normalized_x = np.linspace(0, 1, 101)
                normalized_y = np.interp(
                    normalized_x, np.linspace(0, 1, len(values)), values
                )
                for x, projection in zip(normalized_x, normalized_y):
                    trajectory_rows.append(
                        {
                            "target_id": row["target_id"],
                            "family": row["family"],
                            "thinking": bool(row["thinking"]),
                            "judge_complied": bool(row["judge_complied"])
                            if pd.notna(row["judge_complied"])
                            else False,
                            "segment": segment_name,
                            "normalized_position": x,
                            "projection": projection,
                        }
                    )
        trajectory_frame = pd.DataFrame(trajectory_rows)
        target_curves = trajectory_frame.groupby(
            [
                "target_id",
                "family",
                "thinking",
                "judge_complied",
                "segment",
                "normalized_position",
            ],
            as_index=False,
        )["projection"].mean()
        summary_curves = (
            target_curves.groupby(
                ["family", "thinking", "judge_complied", "segment", "normalized_position"]
            )["projection"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary_curves["sem"] = summary_curves["std"] / np.sqrt(summary_curves["count"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        for ax, segment in zip(axes, ("reasoning", "final_answer")):
            subset = summary_curves[summary_curves["segment"] == segment]
            for (family, thinking, complied), line in subset.groupby(
                ["family", "thinking", "judge_complied"]
            ):
                ax.plot(line["normalized_position"], line["mean"], label=f"{family}; think={thinking}; comply={complied}")
                ax.fill_between(
                    line["normalized_position"],
                    line["mean"] - 1.96 * line["sem"],
                    line["mean"] + 1.96 * line["sem"],
                    alpha=0.12,
                )
            ax.set_title(segment.replace("_", " ").title())
            ax.set_xlabel("Normalized position within semantic span")
        axes[0].set_ylabel("Projection onto refusal direction")
        axes[1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(
            cache_dir / "trajectory_by_family_outcome.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)
        log("wrote trajectory figure")

    if "audit" in stages:
        report = audit_submission_outputs(
            cache_dir,
            summary_path=ROOT / "exec_summary.md",
            ledger_path=ROOT / "ledger.txt",
        )
        log(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
