from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.eval.evaluate_r20_full_pipeline import run_full_evaluation
from scripts.train.finetune_bi_encoder import (
    BASE_MODEL,
    ONTOLOGY_PATH,
    TRAIN_SEGMENTS_CSV,
    run_training,
)
from lib.eval_defaults import (
    DEFAULT_CONFIDENCE_AGGREGATION,
    DEFAULT_FILTER_SEGMENTS,
    DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    EVAL_K,
)

PIPELINE_ROOT = PROJECT_DIR / "data" / "pipeline"
STATE_PATH = PIPELINE_ROOT / "base_model_state.json"
DEFAULT_ONTOLOGY = ONTOLOGY_PATH
DEFAULT_FULL_BATCHES = (
    PROJECT_DIR
    / "data"
    / "gold"
    / "precomputed_batches"
    / "hb_grandfocus1-0-0_bs128_ep1_seed42_lb0.8_gw0.8.pt"
)
BATCHES_ONTOLOGY = DEFAULT_ONTOLOGY  # эталон для сборки строк; состав батчей — по CSV + кодам

BATCH_SIZE = 128
MINI_BATCH_SIZE = 32
SEED = 42
DEFAULT_TRAIN_EPOCHS = 1
BATCH_FILE_EPOCHS = 1  # число списков батчей в .pt (не num_train_epochs)
SAVE_EVAL_STEPS = 250
FILTER_FN_PAIR_FRAC_MAX = 0.01
# curriculum epoch1: 1,0,0 — far/mid/hard; grandfocus = акцент на far (разные grand-ветки)
CURRICULUM_EPOCH1 = "1,0,0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_precomputed_batches(
    precomputed_batches: Optional[Path],
) -> Path:
    if precomputed_batches is not None:
        p = precomputed_batches if precomputed_batches.is_absolute() else PROJECT_DIR / precomputed_batches
        return p.resolve()
    return DEFAULT_FULL_BATCHES


def _run_key(ontology_sha256: str) -> str:
    """Уникальный ключ запуска: префикс онтологии + UTC-время (микросекунды)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{ontology_sha256[:8]}_{ts}"


def _ontology_embeddings_cache_path(ontology_sha256: str) -> Path:
    """Общий кэш base-эмбеддингов по хешу онтологии (не привязан к run_key)."""
    return (
        PIPELINE_ROOT
        / "ontology_emb_cache"
        / f"{ontology_sha256[:16]}_USER-bge-m3.npz"
    )


def _resolve_eval_model_dir(model_dir: Path) -> Path:
    """Для шага 4: best/ по минимальному eval_loss, иначе корень model_dir."""
    best = model_dir / "best"
    if best.is_dir() and any(best.iterdir()):
        print(f"Оценка: лучшая модель (eval_loss) — {best}")
        return best
    print(f"Оценка: каталог best/ пуст или отсутствует, используем {model_dir}")
    return model_dir


def _resolve_trained_model_path(path: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_DIR / p
    return p.resolve()


def _artifact_paths(
    run_key: str,
    batches_path: Path,
    ontology_sha256: str,
    trained_model_path: Optional[Path] = None,
) -> Dict[str, Path]:
    root = PIPELINE_ROOT / "ontology_runs" / run_key
    if trained_model_path is not None:
        model_dir = trained_model_path
    else:
        model_dir = PROJECT_DIR / "models" / "bi-encoder-pipeline" / run_key
    return {
        "root": root,
        "embeddings": _ontology_embeddings_cache_path(ontology_sha256),
        "batches": batches_path,
        "model_dir": model_dir,
        "eval_result": root / "eval_result.json",
        "pipeline_result": root / "pipeline_result.json",
    }


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: Dict[str, Any]) -> None:
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_subprocess(cmd: list[str], step: str) -> None:
    print(f"\n=== {step} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_DIR))


def _validate_batches_dataset_size(batches_path: Path, max_train_samples: Optional[int]) -> None:
    if max_train_samples is None or not batches_path.is_file():
        return
    import torch

    payload = torch.load(batches_path, map_location="cpu", weights_only=False)
    batch_ds_size = int(payload.get("dataset_size", 0))
    expected = int(max_train_samples)
    if batch_ds_size < expected:
        raise ValueError(
            f"В предбатчах только {batch_ds_size} сэмплов, а --max-train-samples={expected}. "
            "Укажите другой .pt или уменьшите лимит."
        )
    if batch_ds_size > expected:
        print(
            f"Предбатчи на полной выборке ({batch_ds_size}); "
            f"обучение на первых {expected} сэмплах после shuffle."
        )


def _step_precompute_embeddings(ontology_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"Эмбеддинги уже есть: {out_path}")
        return
    _run_subprocess(
        [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "data" / "precompute_ontology_embeddings.py"),
            "--model",
            BASE_MODEL,
            "--ontology",
            str(ontology_path),
            "--out",
            str(out_path),
        ],
        "Шаг 1/4: предрасчёт эмбеддингов онтологии",
    )


def _step_ensure_batches(
    batches_path: Path,
    max_train_samples: Optional[int],
    regenerate_batches: bool,
) -> None:
    if batches_path.is_file() and not regenerate_batches:
        print(f"Используем готовые предбатчи: {batches_path}")
        return
    if regenerate_batches and batches_path.resolve() == DEFAULT_FULL_BATCHES.resolve():
        print(
            f"Внимание: перегенерация перезапишет общий файл {batches_path.name} в data/gold/precomputed_batches/"
        )
    batches_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "train" / "generate_hierarchical_batches.py"),
        "--segments-csv",
        str(TRAIN_SEGMENTS_CSV),
        "--ontology-path",
        str(BATCHES_ONTOLOGY),
        "--output-dir",
        str(batches_path.parent),
        "--output-name",
        batches_path.name,
        "--batch-size",
        str(BATCH_SIZE),
        "--seed",
        str(SEED),
        "--epochs",
        str(BATCH_FILE_EPOCHS),
        "--relative-margin",
        "0.05",
        "--curriculum-epoch1",
        CURRICULUM_EPOCH1,
        "--curriculum-epoch2",
        "0.6,0.3,0.1",
        "--curriculum-epoch3plus",
        "0.45,0.35,0.2",
        "--leaf-balance-power",
        "0.8",
        "--grand-balance-weight",
        "0.8",
        "--max-scored-candidates",
        "256",
        "--disable-guide-safe-hard",
        "--disable-sampler-diagnostics",
    ]
    if max_train_samples is not None and batches_path.resolve() != DEFAULT_FULL_BATCHES.resolve():
        cmd.extend(["--max-train-samples", str(int(max_train_samples))])
    _run_subprocess(
        cmd,
        "Общие предбатчи (один раз; эталонная онтология для покрытия кодов из CSV)",
    )


def _step_train(
    ontology_path: Path,
    batches_path: Path,
    model_dir: Path,
    max_train_samples: Optional[int],
    max_batches: Optional[int],
    force: bool,
    mini_batch_size: int,
    epochs: int,
) -> None:
    if model_dir.is_dir() and any(model_dir.iterdir()) and not force:
        print(f"Модель уже обучена: {model_dir}")
        return
    if force and model_dir.exists():
        import shutil

        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_args = Namespace(
        base_model=BASE_MODEL,
        resume="",
        output_dir=str(model_dir),
        epochs=int(epochs),
        batch_size=BATCH_SIZE,
        mini_batch_size=int(mini_batch_size),
        learning_rate=1e-5,
        warmup_ratio=0.1,
        max_train_samples=max_train_samples,
        max_batches=max_batches,
        save_steps=SAVE_EVAL_STEPS,
        seed=SEED,
        use_hierarchical_sampler=False,
        loss="cached_mnr",
        triplets_jsonl="",
        triplet_margin=0.15,
        guide_model=BASE_MODEL,
        disable_guide_safe_hard=True,
        gist_relative_margin=0.05,
        curriculum_epoch1=CURRICULUM_EPOCH1,
        curriculum_epoch2="0.6,0.3,0.1",
        curriculum_epoch3plus="0.45,0.35,0.2",
        leaf_balance_power=0.8,
        grand_balance_weight=0.8,
        max_scored_candidates=256,
        no_sampler_fallback_relaxed=False,
        no_sampler_diagnostics=True,
        precomputed_batches=str(batches_path),
        dataloader_drop_last=False,
        skip_baseline_test=True,
        filter_fn_pair_frac_max=FILTER_FN_PAIR_FRAC_MAX,
        ontology_path=str(ontology_path),
        skip_trainer_eval=False,
        skip_post_train_eval=True,
        no_trainer_checkpoints=False,
        debug_collator_meta_tokenization=False,
    )
    print(
        f"\n=== Шаг 3/4: обучение bi-encoder ({int(epochs)} эпох, "
        f"eval/save каждые {SAVE_EVAL_STEPS} шагов, best/ по eval_loss) ==="
    )
    run_training(train_args)


def _step_evaluate(
    ontology_path: Path,
    model_dir: Path,
    eval_result_path: Path,
    force: bool,
) -> tuple[Dict[str, dict], Path]:
    if eval_result_path.exists() and not force:
        print(f"Оценка уже есть: {eval_result_path}")
        cached = json.loads(eval_result_path.read_text(encoding="utf-8"))
        return cached, _resolve_eval_model_dir(model_dir)
    eval_model_dir = _resolve_eval_model_dir(model_dir)
    eval_args = Namespace(
        ontology=ontology_path,
        model=str(eval_model_dir),
        test_csv=PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test.csv",
        test_docs_csv=PROJECT_DIR / "data" / "gold" / "gisnauka_samples_test_docs.csv",
        gold_jsonl=PROJECT_DIR / "data" / "gold" / "test_set_manual_draft.jsonl",
        valid_csv=PROJECT_DIR / "data" / "gold" / "gisnauka_samples_valid.csv",
        threshold=DEFAULT_THRESHOLD,
        top_k=DEFAULT_TOP_K,
        max_segment_context=DEFAULT_MAX_SEGMENT_LENGTH_FOR_CONTEXT,
        rerank_top_k=DEFAULT_RERANK_TOP_K,
        confidence_aggregation=DEFAULT_CONFIDENCE_AGGREGATION,
        no_filter_segments=not DEFAULT_FILTER_SEGMENTS,
        emb=None,
        k=EVAL_K,
    )
    print("\n=== Шаг 4/4: полная оценка R@20 ===")
    results = run_full_evaluation(eval_args)
    eval_result_path.parent.mkdir(parents=True, exist_ok=True)
    eval_result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {eval_result_path}")
    return results, eval_model_dir


def run_pipeline(
    ontology_path: Path,
    max_train_samples: Optional[int] = None,
    max_batches: Optional[int] = None,
    trained_model_path: Optional[Path] = None,
    precomputed_batches: Optional[Path] = None,
    regenerate_batches: bool = False,
    force: bool = False,
    mini_batch_size: int = MINI_BATCH_SIZE,
    epochs: int = DEFAULT_TRAIN_EPOCHS,
) -> Dict[str, Any]:
    ontology_path = ontology_path.resolve()
    if not ontology_path.is_file():
        raise FileNotFoundError(f"Онтология не найдена: {ontology_path}")

    resolved_model_path: Optional[Path] = None
    if trained_model_path is not None:
        resolved_model_path = _resolve_trained_model_path(trained_model_path)

    batches_path = _resolve_precomputed_batches(precomputed_batches)
    ontology_sha256 = _sha256_file(ontology_path)
    run_key = _run_key(ontology_sha256)
    paths = _artifact_paths(run_key, batches_path, ontology_sha256, resolved_model_path)

    print(f"run_key: {run_key}")

    paths["root"].mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    _step_precompute_embeddings(ontology_path, paths["embeddings"])

    _step_ensure_batches(batches_path, max_train_samples, regenerate_batches=regenerate_batches)
    if not paths["batches"].is_file():
        raise FileNotFoundError(f"Не найдены предбатчи: {paths['batches']}")
    _validate_batches_dataset_size(paths["batches"], max_train_samples)

    _step_train(
        ontology_path,
        paths["batches"],
        paths["model_dir"],
        max_train_samples,
        max_batches,
        force=force,
        mini_batch_size=mini_batch_size,
        epochs=int(epochs),
    )
    eval_results, eval_model_dir = _step_evaluate(
        ontology_path,
        paths["model_dir"],
        paths["eval_result"],
        force=force,
    )

    pipeline_result = {
        "run_key": run_key,
        "ontology_path": str(ontology_path),
        "ontology_sha256": ontology_sha256,
        "epochs": int(epochs),
        "max_train_samples": max_train_samples,
        "max_batches": max_batches,
        "trained_model_path": str(paths["model_dir"]),
        "eval_model_dir": str(eval_model_dir),
        "base_model": BASE_MODEL,
        "embeddings_path": str(paths["embeddings"]),
        "batches_path": str(paths["batches"]),
        "model_dir": str(paths["model_dir"]),
        "eval_result_path": str(paths["eval_result"]),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "eval_metrics": eval_results,
    }
    paths["pipeline_result"].write_text(
        json.dumps(pipeline_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    state = _load_state()
    state["last_run"] = pipeline_result
    state["runs"] = state.get("runs", {})
    state["runs"][run_key] = pipeline_result
    _save_state(state)

    print("\n=== Пайплайн завершён ===")
    print(f"  run_key: {run_key}")
    print(f"  модель (train): {paths['model_dir']}")
    print(f"  модель (eval):  {eval_model_dir}")
    if "test_gisnauka_docs" in eval_results:
        r20 = eval_results["test_gisnauka_docs"].get("R@20", 0.0)
        print(f"  test_gisnauka_docs R@20: {r20:.4f}")
    return pipeline_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Пайплайн: предбатчи → эмбеддинги → обучение → R@20. "
            "Каждый запуск получает уникальный run_key (время UTC); старые прогоны не перезаписываются."
        )
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=DEFAULT_ONTOLOGY,
        help="Путь к JSON онтологии (по умолчанию data/ontology_grnti_with_llm.json).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Smoke: эквивалент --max-batches ceil(N/128) при предбатчах. "
            "Без аргумента — полное обучение."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Smoke: первые N предбатчей после FN-фильтра (полный train, как prod).",
    )
    parser.add_argument(
        "--trained-model-path",
        "--trained-model-filename",
        type=Path,
        default=None,
        dest="trained_model_path",
        metavar="PATH",
        help=(
            "Каталог для сохранения обученной модели "
            "(например models/final или models/my-bi-encoder). "
            "Относительный путь — от корня репозитория. "
            "По умолчанию: models/bi-encoder-pipeline/{run_key}."
        ),
    )
    parser.add_argument(
        "--precomputed-batches",
        type=Path,
        default=None,
        help=(
            "Готовый .pt с предбатчами (не зависит от текста онтологии, только от CSV и кодов). "
            f"По умолчанию для полной выборки: {DEFAULT_FULL_BATCHES.name}"
        ),
    )
    parser.add_argument(
        "--regenerate-batches",
        action="store_true",
        help="Перегенерировать общие предбатчи (обычно не нужно при смене описаний в онтологии).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Переобучить, если model_dir уже существует (--trained-model-path), "
            "или перезаписать eval в каталоге этого run."
        ),
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=MINI_BATCH_SIZE,
        help=(
            "Mini-batch для CachedMultipleNegativesRankingLoss (память одного forward). "
            f"По умолчанию {MINI_BATCH_SIZE}."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAIN_EPOCHS,
        help=(
            f"Число эпох обучения bi-encoder (Trainer). "
            f"Eval и checkpoint каждые {SAVE_EVAL_STEPS} шагов; лучшая модель — model_dir/best/. "
            f"По умолчанию {DEFAULT_TRAIN_EPOCHS}."
        ),
    )
    args = parser.parse_args()
    if int(args.epochs) < 1:
        parser.error("--epochs должен быть >= 1")
    run_pipeline(
        ontology_path=args.ontology,
        max_train_samples=args.max_train_samples,
        max_batches=args.max_batches,
        trained_model_path=args.trained_model_path,
        precomputed_batches=args.precomputed_batches,
        regenerate_batches=bool(args.regenerate_batches),
        force=bool(args.force),
        mini_batch_size=int(args.mini_batch_size),
        epochs=int(args.epochs),
    )


if __name__ == "__main__":
    main()
