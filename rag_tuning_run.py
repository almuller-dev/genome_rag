import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

from evo_engine_v1 import EvolutionEngine
from final_evaluator import RAGEvaluator, apply_rag_config, canonicalize
from rag_distance import rag_distance
from rag_genome import RAG_SPACE


def _is_int_range(bounds: Tuple[Any, Any]) -> bool:
    lo, hi = bounds
    return isinstance(lo, int) and isinstance(hi, int)


def random_rag_genome(rng: random.Random) -> Dict[str, Any]:
    g: Dict[str, Any] = {}
    for key, space in RAG_SPACE.items():
        if isinstance(space, tuple) and len(space) == 2:
            lo, hi = space
            if _is_int_range((lo, hi)):
                g[key] = rng.randint(int(lo), int(hi))
            else:
                g[key] = round(rng.uniform(float(lo), float(hi)), 2)
        elif isinstance(space, list) and space:
            g[key] = rng.choice(space)
        else:
            raise ValueError(f"Unsupported search space for key={key!r}: {space!r}")
    return canonicalize(g)


def mutate_rag_genome(genome: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    g = dict(genome)
    keys = list(RAG_SPACE.keys())
    keys_to_mutate = rng.sample(keys, k=rng.randint(1, min(2, len(keys))))

    for key in keys_to_mutate:
        space = RAG_SPACE[key]
        if isinstance(space, list):
            g[key] = rng.choice(space)
            continue

        lo, hi = space
        if _is_int_range((lo, hi)):
            cur = int(g[key])
            step = max(1, int((int(hi) - int(lo)) * 0.1))
            shift = int(rng.gauss(0, step))
            g[key] = max(int(lo), min(int(hi), cur + shift))
        else:
            cur = float(g[key])
            sigma = max(0.01, (float(hi) - float(lo)) * 0.1)
            nxt = cur + rng.gauss(0, sigma)
            g[key] = round(max(float(lo), min(float(hi), nxt)), 2)

    return canonicalize(g)


def crossover_rag_genome(a: Dict[str, Any], b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    g = {k: a[k] if rng.random() < 0.5 else b[k] for k in a.keys()}
    return canonicalize(g)


def build_evaluator(
    *,
    base_repo: Path,
    docs_dir_rel: str,
    bench_rel: str,
    artifact_root: Path,
    collection_prefix: str,
) -> RAGEvaluator:
    return RAGEvaluator(
        base_repo=base_repo,
        apply_genome_fn=apply_rag_config,
        stages=[],
        docs_dir_rel=docs_dir_rel,
        bench_rel=bench_rel,
        persistent_artifact_root=artifact_root,
        collection_prefix=collection_prefix,
    )


def run(
    *,
    base_repo: Path = Path("./my_rag_mvp"),
    docs_dir_rel: str = "docs",
    bench_rel: str = "bench/support_tickets.jsonl",
    artifact_root: Path = Path(".rag_artifacts/index"),
    collection_prefix: str = "rag_support",
    population_size: int = 30,
    elites: int = 3,
    max_workers: int = 4,
    generations: int = 25,
    seed: int = 123,
    cache_dir: Path = Path(".rag_cache"),
    eval_salt: str = "support_benchmark_v2",
) -> None:
    evaluator = build_evaluator(
        base_repo=base_repo,
        docs_dir_rel=docs_dir_rel,
        bench_rel=bench_rel,
        artifact_root=artifact_root,
        collection_prefix=collection_prefix,
    )

    with EvolutionEngine(
        random_genome_fn=random_rag_genome,
        mutate_fn=mutate_rag_genome,
        crossover_fn=crossover_rag_genome,
        distance_fn=lambda a, b: rag_distance(a, b, RAG_SPACE),
        evaluate_fn=evaluator,
        population_size=population_size,
        elites=elites,
        max_workers=max_workers,
        cache_dir=cache_dir,
        eval_salt=eval_salt,
        seed=seed,
    ) as engine:
        best = engine.run(generations=generations, verbose=True)
        print(json.dumps(best.genome, indent=2))
        print(best.fitness.fitness, best.fitness.details.get("overall_score"))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-repo", type=Path, default=Path("./my_rag_mvp"))
    ap.add_argument("--docs-dir", type=str, default="docs")
    ap.add_argument("--bench", type=str, default="bench/support_tickets.jsonl")
    ap.add_argument("--artifact-root", type=Path, default=Path(".rag_artifacts/index"))
    ap.add_argument("--collection-prefix", type=str, default="rag_support")
    ap.add_argument("--population-size", type=int, default=30)
    ap.add_argument("--elites", type=int, default=3)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cache-dir", type=Path, default=Path(".rag_cache"))
    ap.add_argument("--eval-salt", type=str, default="support_benchmark_v2")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        base_repo=args.base_repo,
        docs_dir_rel=args.docs_dir,
        bench_rel=args.bench,
        artifact_root=args.artifact_root,
        collection_prefix=args.collection_prefix,
        population_size=args.population_size,
        elites=args.elites,
        max_workers=args.max_workers,
        generations=args.generations,
        seed=args.seed,
        cache_dir=args.cache_dir,
        eval_salt=args.eval_salt,
    )
