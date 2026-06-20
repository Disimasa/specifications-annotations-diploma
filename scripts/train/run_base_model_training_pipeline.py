from __future__ import annotations

import argparse
import hashlib
import json
import os
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
EPOCHS = 1
FILTER_FN_PAIR_FRAC_MAX = 0.01
FINETUNE_SCRIPT = PROJECT_DIR / "scripts" / "train" / "finetune_bi_encoder.py"
TORCHRUN_ENTRY = PROJECT_DIR / "scripts" / "train" / "torchrun_entry.py"
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


def _run_key(
    ontology_sha256: str,
    max_train_samples: Optional[int] = None,
    max_batches: Optional[int] = None,
    trained_model_path: Optional[Path] = None,
) -> str:
    key = ontology_sha256[:16]
    if max_batches is not None:
        key = f"{key}_b{int(max_batches)}"
    elif max_train_samples is not None:
        key = f"{key}_n{int(max_train_samples)}"
    if trained_model_path is not None:
        model_hash = hashlib.sha256(str(trained_model_path).encode("utf-8")).hexdigest()[:8]
        key = f"{key}_m{model_hash}"
    return key


def _resolve_trained_model_path(path: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_DIR / p
    return p.resolve()


def _artifact_paths(
    run_key: str,
    batches_path: Path,
    trained_model_path: Optional[Path] = None,
) -> Dict[str, Path]:
    root = PIPELINE_ROOT / "ontology_runs" / run_key
    if trained_model_path is not None:
        model_dir = trained_model_path
    else:
        model_dir = PROJECT_DIR / "models" / "bi-encoder-pipeline" / run_key
    return {
        "root": root,
        "embeddings": root / "ontology_emb_USER-bge-m3.npz",
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


def _artifacts_complete(paths: Dict[str, Path]) -> bool:
    required = ("embeddings", "model_dir", "eval_result")
    for key in required:
        p = paths[key]
        if key == "model_dir":
            if not p.is_dir():
                return False
            if not any(p.iterdir()):
                return False
        elif not p.is_file():
            return False
    return True


def _ddp_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["USE_LIBUV"] = "0"
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")
    return env


def _run_subprocess(cmd: list[str], step: str, *, env: Optional[Dict[str, str]] = None) -> None:
    print(f"\n=== {step} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_DIR), env=env)


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
        str(EPOCHS),
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


def _cuda_device_count() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except ImportError:
        pass
    return 0


def _resolve_ddp_settings(
    use_ddp: bool,
    nproc_per_node: Optional[int],
) -> tuple[bool, int]:
    n_gpu = _cuda_device_count()
    if not use_ddp or n_gpu <= 1:
        return False, 1
    nproc = int(nproc_per_node) if nproc_per_node is not None else n_gpu
    nproc = max(1, min(nproc, n_gpu))
    return nproc > 1, nproc


def _build_finetune_train_namespace(
    ontology_path: Path,
    batches_path: Path,
    model_dir: Path,
    max_train_samples: Optional[int],
    max_batches: Optional[int],
    mini_batch_size: int,
) -> Namespace:
    return Namespace(
        base_model=BASE_MODEL,
        resume="",
        output_dir=str(model_dir),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        mini_batch_size=int(mini_batch_size),
        learning_rate=1e-5,
        warmup_ratio=0.1,
        max_train_samples=max_train_samples,
        max_batches=max_batches,
        save_steps=500,
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
        skip_trainer_eval=True,
        skip_post_train_eval=True,
        no_trainer_checkpoints=True,
        debug_collator_meta_tokenization=False,
    )


def _namespace_to_finetune_cli(args: Namespace) -> list[str]:
    cli: list[str] = [
        "--base-model",
        str(args.base_model),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--mini-batch-size",
        str(int(args.mini_batch_size)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--warmup-ratio",
        str(float(args.warmup_ratio)),
        "--save-steps",
        str(int(args.save_steps)),
        "--seed",
        str(int(args.seed)),
        "--loss",
        str(args.loss),
        "--guide-model",
        str(args.guide_model),
        "--gist-relative-margin",
        str(float(args.gist_relative_margin)),
        "--curriculum-epoch1",
        str(args.curriculum_epoch1),
        "--curriculum-epoch2",
        str(args.curriculum_epoch2),
        "--curriculum-epoch3plus",
        str(args.curriculum_epoch3plus),
        "--leaf-balance-power",
        str(float(args.leaf_balance_power)),
        "--grand-balance-weight",
        str(float(args.grand_balance_weight)),
        "--max-scored-candidates",
        str(int(args.max_scored_candidates)),
        "--precomputed-batches",
        str(args.precomputed_batches),
        "--filter-fn-pair-frac-max",
        str(float(args.filter_fn_pair_frac_max)),
        "--ontology-path",
        str(args.ontology_path),
        "--triplet-margin",
        str(float(args.triplet_margin)),
    ]
    if args.max_train_samples is not None:
        cli.extend(["--max-train-samples", str(int(args.max_train_samples))])
    if args.max_batches is not None:
        cli.extend(["--max-batches", str(int(args.max_batches))])
    if args.resume:
        cli.extend(["--resume", str(args.resume)])
    if getattr(args, "triplets_jsonl", ""):
        cli.extend(["--triplets-jsonl", str(args.triplets_jsonl)])
    if bool(args.use_hierarchical_sampler):
        cli.append("--use-hierarchical-sampler")
    if bool(args.disable_guide_safe_hard):
        cli.append("--disable-guide-safe-hard")
    if bool(args.no_sampler_fallback_relaxed):
        cli.append("--no-sampler-fallback-relaxed")
    if bool(args.no_sampler_diagnostics):
        cli.append("--no-sampler-diagnostics")
    if bool(args.dataloader_drop_last):
        cli.append("--dataloader-drop-last")
    if bool(args.skip_baseline_test):
        cli.append("--skip-baseline-test")
    if bool(args.skip_trainer_eval):
        cli.append("--skip-trainer-eval")
    if bool(args.skip_post_train_eval):
        cli.append("--skip-post-train-eval")
    if bool(args.no_trainer_checkpoints):
        cli.append("--no-trainer-checkpoints")
    if bool(args.debug_collator_meta_tokenization):
        cli.append("--debug-collator-meta-tokenization")
    return cli


def _step_train(
    ontology_path: Path,
    batches_path: Path,
    model_dir: Path,
    max_train_samples: Optional[int],
    max_batches: Optional[int],
    force: bool,
    *,
    use_ddp: bool,
    nproc_per_node: Optional[int],
    mini_batch_size: int,
) -> None:
    if model_dir.is_dir() and any(model_dir.iterdir()) and not force:
        print(f"Модель уже обучена: {model_dir}")
        return
    if force and model_dir.exists():
        import shutil

        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_args = _build_finetune_train_namespace(
        ontology_path,
        batches_path,
        model_dir,
        max_train_samples,
        max_batches,
        mini_batch_size,
    )
    use_ddp_run, nproc = _resolve_ddp_settings(use_ddp, nproc_per_node)
    finetune_cli = _namespace_to_finetune_cli(train_args)

    if use_ddp_run and nproc > 1:
        cmd = [
            sys.executable,
            str(TORCHRUN_ENTRY),
            "--standalone",
            f"--nproc_per_node={nproc}",
            str(FINETUNE_SCRIPT),
            *finetune_cli,
        ]
        _run_subprocess(
            cmd,
            f"Шаг 3/4: обучение bi-encoder (DDP, {nproc} GPU)",
            env=_ddp_subprocess_env(),
        )
        return

    print("\n=== Шаг 3/4: обучение bi-encoder (1 GPU) ===")
    run_training(train_args)


def _step_evaluate(
    ontology_path: Path,
    model_dir: Path,
    embeddings_path: Path,
    eval_result_path: Path,
    force: bool,
) -> Dict[str, dict]:
    if eval_result_path.exists() and not force:
        print(f"Оценка уже есть: {eval_result_path}")
        return json.loads(eval_result_path.read_text(encoding="utf-8"))
    eval_args = Namespace(
        ontology=ontology_path,
        model=str(model_dir),
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
        emb=embeddings_path,
        k=EVAL_K,
    )
    print("\n=== Шаг 4/4: полная оценка R@20 ===")
    results = run_full_evaluation(eval_args)
    eval_result_path.parent.mkdir(parents=True, exist_ok=True)
    eval_result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {eval_result_path}")
    return results


def run_pipeline(
    ontology_path: Path,
    max_train_samples: Optional[int] = None,
    max_batches: Optional[int] = None,
    trained_model_path: Optional[Path] = None,
    precomputed_batches: Optional[Path] = None,
    regenerate_batches: bool = False,
    force: bool = False,
    use_ddp: bool = True,
    nproc_per_node: Optional[int] = None,
    mini_batch_size: int = MINI_BATCH_SIZE,
) -> Dict[str, Any]:
    ontology_path = ontology_path.resolve()
    if not ontology_path.is_file():
        raise FileNotFoundError(f"Онтология не найдена: {ontology_path}")

    resolved_model_path: Optional[Path] = None
    if trained_model_path is not None:
        resolved_model_path = _resolve_trained_model_path(trained_model_path)

    batches_path = _resolve_precomputed_batches(precomputed_batches)
    ontology_sha256 = _sha256_file(ontology_path)
    run_key = _run_key(ontology_sha256, max_train_samples, max_batches, resolved_model_path)
    paths = _artifact_paths(run_key, batches_path, resolved_model_path)

    state = _load_state()
    cached_run = state.get("runs", {}).get(run_key)
    if cached_run and _artifacts_complete(paths) and not force:
        print("Эта онтология уже обучена и оценена, артефакты на месте.")
        print(f"  run_key: {run_key}")
        print(f"  модель: {paths['model_dir']}")
        print(f"  оценка: {paths['eval_result']}")
        print("Для повторного запуска укажите --force")
        return cached_run

    paths["root"].mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    if force and paths["embeddings"].exists():
        paths["embeddings"].unlink()
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
        use_ddp=use_ddp,
        nproc_per_node=nproc_per_node,
        mini_batch_size=mini_batch_size,
    )
    eval_results = _step_evaluate(
        ontology_path,
        paths["model_dir"],
        paths["embeddings"],
        paths["eval_result"],
        force=force,
    )

    pipeline_result = {
        "run_key": run_key,
        "ontology_path": str(ontology_path),
        "ontology_sha256": ontology_sha256,
        "max_train_samples": max_train_samples,
        "max_batches": max_batches,
        "trained_model_path": str(paths["model_dir"]),
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
    state["last_run"] = pipeline_result
    state["runs"] = state.get("runs", {})
    state["runs"][run_key] = pipeline_result
    _save_state(state)

    print("\n=== Пайплайн завершён ===")
    print(f"  run_key: {run_key}")
    print(f"  модель: {paths['model_dir']}")
    if "test_gisnauka_docs" in eval_results:
        r20 = eval_results["test_gisnauka_docs"].get("R@20", 0.0)
        print(f"  test_gisnauka_docs R@20: {r20:.4f}")
    return pipeline_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Пайплайн сравнения онтологий: общие предбатчи → эмбеддинги → обучение → R@20. "
            "При смене онтологии пересчитываются только эмбеддинги, модель и метрики."
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
        help="Переобучить и переоценить эту онтологию, даже если артефакты уже есть.",
    )
    parser.add_argument(
        "--no-ddp",
        action="store_true",
        help="Не использовать DDP (обычный python / DataParallel при нескольких GPU).",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Число GPU для DDP (по умолчанию — все видимые CUDA-устройства).",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=MINI_BATCH_SIZE,
        help="Mini-batch для CachedMultipleNegativesRankingLoss (память одного forward).",
    )
    args = parser.parse_args()
    use_ddp_run, nproc = _resolve_ddp_settings(not args.no_ddp, args.nproc_per_node)
    if use_ddp_run:
        print(f"Обучение: DDP на {nproc} GPU (torchrun)")
    elif _cuda_device_count() > 1 and args.no_ddp:
        print("Обучение: in-process (SentenceTransformers DataParallel)")
    else:
        print("Обучение: in-process (1 GPU)")
    run_pipeline(
        ontology_path=args.ontology,
        max_train_samples=args.max_train_samples,
        max_batches=args.max_batches,
        trained_model_path=args.trained_model_path,
        precomputed_batches=args.precomputed_batches,
        regenerate_batches=bool(args.regenerate_batches),
        force=bool(args.force),
        use_ddp=not bool(args.no_ddp),
        nproc_per_node=args.nproc_per_node,
        mini_batch_size=int(args.mini_batch_size),
    )


if __name__ == "__main__":
    main()
