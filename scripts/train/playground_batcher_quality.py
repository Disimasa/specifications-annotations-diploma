from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import optuna
import torch
from datasets import Dataset


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Скрипт должен лежать в repo/scripts/train/ — иначе parents[2] указывает не на корень репозитория.
_FE_PATH = PROJECT_DIR / "scripts" / "train" / "finetune_bi_encoder.py"
if not _FE_PATH.is_file():
    raise RuntimeError(
        f"Неверный PROJECT_DIR={PROJECT_DIR} (ожидался корень репозитория с {_FE_PATH}). "
        "Запускайте из корня: python scripts/train/playground_batcher_quality.py"
    )

import scripts.train.finetune_bi_encoder as fe

BASE_MODEL = fe.BASE_MODEL


def _verify_hierarchical_train_ready() -> None:
    """До Optuna: сегменты и строки с метаданными GRNTI (как у HierarchicalGrntiBatchSampler)."""
    if not fe.TRAIN_SEGMENTS_CSV.is_file():
        raise FileNotFoundError(f"Нет train-сегментов: {fe.TRAIN_SEGMENTS_CSV}")
    code_to_text = fe.load_ontology_texts(fe.ONTOLOGY_PATH)
    rows = fe.build_hierarchical_rows_from_segments(fe.TRAIN_SEGMENTS_CSV, code_to_text)
    if not rows:
        raise RuntimeError(
            "build_hierarchical_rows_from_segments вернул 0 строк. "
            f"Проверьте CSV (doc_id, segment_text, grnti_codes) и онтологию: {fe.TRAIN_SEGMENTS_CSV}"
        )
    need = {"text1", "text2", "doc_id", "leaf", "parent", "grand", "doc_gold_leaves"}
    if not need.issubset(rows[0].keys()):
        raise RuntimeError(f"Неожиданная схема иерархических строк, ключи: {list(rows[0].keys())}")
    ds = Dataset.from_list(rows[: min(8, len(rows))])
    if not need.issubset(set(ds.column_names)):
        raise RuntimeError(
            f"Dataset.from_list не содержит нужных колонок: {ds.column_names}. "
            "Обновите пакет datasets или проверьте версию."
        )
    ds2 = ds.select(range(len(ds)))
    if not need.issubset(set(ds2.column_names)):
        raise RuntimeError(f"Dataset.select обрезал колонки: было {ds.column_names}, стало {ds2.column_names}")
OUTPUT_ROOT = PROJECT_DIR / "models" / "playground-batcher-quality-optuna"
REPORT_PATH = PROJECT_DIR / "data" / "gold" / "playground" / "batcher_quality_optuna_report.json"
STUDY_DB = PROJECT_DIR / "data" / "gold" / "playground" / "batcher_quality_optuna.db"
STUDY_NAME = "batcher_quality_search"

N_TRIALS = 100
SEED = 42
EPOCHS = 1
BATCH_SIZE = 128
MINI_BATCH_SIZE = 32
MAX_TRAIN_SAMPLES = 10000
LOSS = "cached_mnr"  # "cached_mnr" | "gist"
DISABLE_GUIDE_SAFE_HARD = True
GIST_RELATIVE_MARGIN = 0.05
CURRICULUM_EPOCH2 = "0.6,0.3,0.1"
CURRICULUM_EPOCH3PLUS = "0.45,0.35,0.2"
NO_SAMPLER_FALLBACK_RELAXED = False
NO_SAMPLER_DIAGNOSTICS = True

# Пространство поиска (границы как константы):
SEARCH_MAX_SCORED_CANDIDATES = [64, 128, 256]
SEARCH_LEAF_BALANCE_POWER = [0.5, 0.8, 1.0]
SEARCH_GRAND_BALANCE_WEIGHT = [0.8, 1.0, 1.2]
SEARCH_CURRICULUM_EPOCH1 = [
    "0.8,0.2,0",
    "0.7,0.2,0.1",
    "0.6,0.3,0.1",
    "random",   # без явных долей (score по curriculum выключен)
    "0.1,0.3,0.6",
    "0,0,1",    # максимум hard (leaf-level различение)
]


def _run_single(
    cfg: Dict[str, float | int | str],
    cfg_idx: int,
) -> Dict[str, float | int | str]:
    run_out_dir = OUTPUT_ROOT / f"trial_{cfg_idx:03d}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_out_dir / "trial_error.log"

    print(
        f"[trial {cfg_idx}] start: msc={cfg['max_scored_candidates']}, "
        f"lbp={cfg['leaf_balance_power']}, gbw={cfg['grand_balance_weight']}, "
        f"c1={cfg['curriculum_epoch1']}"
    )
    args = SimpleNamespace(
        base_model=BASE_MODEL,
        resume="",
        output_dir=str(run_out_dir),
        epochs=int(EPOCHS),
        batch_size=int(BATCH_SIZE),
        mini_batch_size=int(MINI_BATCH_SIZE),
        learning_rate=1e-5,
        warmup_ratio=0.1,
        max_train_samples=int(MAX_TRAIN_SAMPLES),
        save_steps=500,
        seed=int(SEED),
        use_hierarchical_sampler=True,
        loss=LOSS,
        guide_model=BASE_MODEL,
        disable_guide_safe_hard=bool(DISABLE_GUIDE_SAFE_HARD),
        gist_relative_margin=float(GIST_RELATIVE_MARGIN),
        curriculum_epoch1=str(cfg["curriculum_epoch1"]),
        curriculum_epoch2=CURRICULUM_EPOCH2,
        curriculum_epoch3plus=CURRICULUM_EPOCH3PLUS,
        leaf_balance_power=float(cfg["leaf_balance_power"]),
        grand_balance_weight=float(cfg["grand_balance_weight"]),
        max_scored_candidates=int(cfg["max_scored_candidates"]),
        no_sampler_fallback_relaxed=bool(NO_SAMPLER_FALLBACK_RELAXED),
        no_sampler_diagnostics=bool(NO_SAMPLER_DIAGNOSTICS),
        precomputed_batches="",
        dataloader_drop_last=False,
        skip_baseline_test=True,
    )
    try:
        payload = fe.run_training(args)
    except Exception as exc:
        run_log.write_text(str(exc), encoding="utf-8")
        return {
            "config_id": cfg_idx,
            "status": "failed",
            "error": str(exc),
            "output_dir": str(run_out_dir),
            **cfg,
        }

    fin = payload.get("finetuned_metrics", {}) or {}
    org = payload.get("original_metrics", {}) or {}
    r20 = float(fin.get("R@20", 0.0))
    p20 = float(fin.get("P@20", 0.0))
    r20_parent = float(fin.get("R@20_parent", 0.0))
    if payload.get("skip_baseline_test"):
        d_r20 = d_p20 = d_r20_parent = 0.0
    else:
        d_r20 = r20 - float(org.get("R@20", 0.0))
        d_p20 = p20 - float(org.get("P@20", 0.0))
        d_r20_parent = r20_parent - float(org.get("R@20_parent", 0.0))
    row: Dict[str, float | int | str] = {
        "config_id": cfg_idx,
        "status": "ok",
        "output_dir": str(run_out_dir),
        "result_json": str(payload.get("result_path", "")),
        "R@20": r20,
        "P@20": p20,
        "R@20_parent": r20_parent,
        "delta_R@20": d_r20,
        "delta_P@20": d_p20,
        "delta_R@20_parent": d_r20_parent,
        **cfg,
    }
    if payload.get("skip_baseline_test"):
        print(
            f"[trial {cfg_idx}] done: R@20={r20:.4f}, "
            f"R@20_parent={r20_parent:.4f}, P@20={p20:.4f} (только finetuned, без baseline-теста)"
        )
    else:
        print(
            f"[trial {cfg_idx}] done: R@20={r20:.4f}, "
            f"R@20_parent={r20_parent:.4f}, P@20={p20:.4f}, dR@20={d_r20:.4f}"
        )
    return row


def _sample_cfg(trial: optuna.Trial) -> Dict[str, float | int | str]:
    return {
        "max_scored_candidates": trial.suggest_categorical(
            "max_scored_candidates", SEARCH_MAX_SCORED_CANDIDATES
        ),
        "leaf_balance_power": trial.suggest_categorical(
            "leaf_balance_power", SEARCH_LEAF_BALANCE_POWER
        ),
        "grand_balance_weight": trial.suggest_categorical(
            "grand_balance_weight", SEARCH_GRAND_BALANCE_WEIGHT
        ),
        "curriculum_epoch1": trial.suggest_categorical(
            "curriculum_epoch1", SEARCH_CURRICULUM_EPOCH1
        ),
    }


def main() -> None:
    if MAX_TRAIN_SAMPLES <= 0:
        raise ValueError("MAX_TRAIN_SAMPLES must be positive")
    if LOSS == "gist" and DISABLE_GUIDE_SAFE_HARD:
        raise ValueError("DISABLE_GUIDE_SAFE_HARD нельзя использовать с LOSS='gist'")

    print(f"[playground] finetune_bi_encoder: {fe.__file__}")
    print(f"[playground] TRAIN_SEGMENTS_CSV: {fe.TRAIN_SEGMENTS_CSV} (exists={fe.TRAIN_SEGMENTS_CSV.is_file()})")
    _verify_hierarchical_train_ready()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float | int | str]] = []

    def objective(trial: optuna.Trial) -> float:
        # На всякий случай чистим кеш до trial.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cfg = _sample_cfg(trial)
        row = _run_single(cfg, trial.number + 1)
        row["trial_number"] = trial.number
        results.append(row)
        if row.get("status") != "ok":
            trial.set_user_attr("result", row)
            print(f"[trial {trial.number + 1}] failed: {json.dumps(row, ensure_ascii=False)}")
            # Не пруним, чтобы оптимизация не обрывалась и в отчете было видно все ошибки.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0
        trial.set_user_attr("result", row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float(row["R@20"])

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB.as_posix()}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    ok_rows = [r for r in results if r.get("status") == "ok"]
    ok_rows = sorted(ok_rows, key=lambda x: (float(x["R@20"]), float(x["P@20"])), reverse=True)
    payload = {
        "settings": {
            "n_trials": N_TRIALS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "loss": LOSS,
            "disable_guide_safe_hard": bool(DISABLE_GUIDE_SAFE_HARD),
            "seed": SEED,
            "search_space": {
                "max_scored_candidates": SEARCH_MAX_SCORED_CANDIDATES,
                "leaf_balance_power": SEARCH_LEAF_BALANCE_POWER,
                "grand_balance_weight": SEARCH_GRAND_BALANCE_WEIGHT,
                "curriculum_epoch1": SEARCH_CURRICULUM_EPOCH1,
            },
        },
        "results": results,
        "top_by_R@20": ok_rows[:10],
        "best_trial_number": study.best_trial.number if study.best_trial else None,
        "best_value_R@20": float(study.best_value) if study.best_trial else None,
        "best_params": dict(study.best_params) if study.best_trial else {},
    }
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")
    if ok_rows:
        print("Best config:")
        print(json.dumps(ok_rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

