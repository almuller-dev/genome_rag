from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("AIDE_Evo")
NEG_INF = -1e18


# -----------------------------
# Core data structures
# -----------------------------


@dataclass(frozen=True)
class FitnessResult:
    """
    Higher is better.
    details may include: test stage results, timings, scores, errors, etc.
    Keep it JSON-serializable if you want it cached.
    """

    fitness: float
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class Candidate:
    """
    genome should be JSON-serializable for stable hashing / caching.
    """

    genome: Any
    key: str
    fitness: Optional[FitnessResult] = None
    parents: Tuple[str, ...] = field(default_factory=tuple)


# -----------------------------
# Stable hashing / cache salt
# -----------------------------


def stable_genome_key(genome: Any, salt: str = "") -> str:
    """
    Stable key for caching. Requires genome to be JSON-serializable.

    NOTE:
      If your evaluation logic changes (tests/bench commands, scoring, env),
      change 'salt' to invalidate old cache entries.
    """
    try:
        blob = json.dumps(genome, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError as e:
        raise TypeError(
            "Genome is not JSON-serializable. "
            "Use only dict/list/str/int/float/bool/None (and nested), "
            "or pre-encode the genome into a JSON-friendly structure."
        ) from e

    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(blob.encode("utf-8"))
    return h.hexdigest()


# -----------------------------
# Sharded disk cache with optional detail sidecar + versioning
# -----------------------------


class ShardedJsonCache:
    """
    Sharded disk cache:
      root/ab/cd/<key>.json
      root/ab/cd/<key>.details.json (optional for large details)

    Versioning:
      Each entry stores schema_version + eval_salt.
      Reads reject mismatches.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        root: Path,
        *,
        eval_salt: str,
        details_inline_limit_bytes: int = 10_000,
        fsync_writes: bool = True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.eval_salt = eval_salt
        self.details_inline_limit_bytes = int(details_inline_limit_bytes)
        self.fsync_writes = bool(fsync_writes)

    def _shard_dir(self, key: str) -> Path:
        return self.root / key[:2] / key[2:4]

    def _main_path(self, key: str) -> Path:
        return self._shard_dir(key) / f"{key}.json"

    def _details_path(self, key: str) -> Path:
        return self._shard_dir(key) / f"{key}.details.json"

    def get(self, key: str) -> Optional[FitnessResult]:
        p = self._main_path(key)
        if not p.exists():
            return None

        try:
            main = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("Cache main parse failure for %s: %r", p, e)
            return None

        # Validate schema + salt (cache invalidation)
        if int(main.get("schema_version", -1)) != self.SCHEMA_VERSION:
            return None
        if str(main.get("eval_salt", "")) != self.eval_salt:
            return None

        try:
            fitness = float(main.get("fitness", NEG_INF))
        except Exception:
            return None

        details = main.get("details")
        if details is None:
            dp = self._details_path(key)
            if dp.exists():
                try:
                    details = json.loads(dp.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.debug("Cache details parse failure for %s: %r", dp, e)
                    details = {}
            else:
                details = {}

        if not isinstance(details, dict):
            # Normalize to dict to keep callers sane
            details = {"details": details}

        return FitnessResult(fitness=fitness, details=details)

    def set(self, key: str, result: FitnessResult) -> None:
        d = self._shard_dir(key)
        d.mkdir(parents=True, exist_ok=True)

        main_path = self._main_path(key)
        details_path = self._details_path(key)

        # Decide whether to inline details
        try:
            details_json = json.dumps(result.details, separators=(",", ":"), ensure_ascii=False)
        except TypeError:
            # If details aren't serializable, store a safe summary only
            details_json = json.dumps(
                {"non_json_details": True, "type": str(type(result.details))},
                separators=(",", ":"),
                ensure_ascii=False,
            )

        inline = len(details_json.encode("utf-8")) <= self.details_inline_limit_bytes

        main_payload: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "eval_salt": self.eval_salt,
            "fitness": float(result.fitness),
        }

        if inline:
            # Store details inline and remove stale sidecar if it exists
            try:
                main_payload["details"] = result.details
            except Exception:
                main_payload["details"] = {"non_json_details": True}

            self._atomic_write(main_path, main_payload)
            if details_path.exists():
                try:
                    details_path.unlink()
                except Exception:
                    pass
        else:
            # Store fitness only in main; store details in sidecar
            self._atomic_write(main_path, main_payload)
            self._atomic_write(details_path, result.details)

    def _atomic_write(self, target: Path, data: Any) -> None:
        nonce = f"{os.getpid()}.{random.getrandbits(32)}"
        tmp = target.with_name(f"{target.name}.{nonce}.tmp")

        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                f.flush()
                if self.fsync_writes:
                    os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            # If replace succeeded, tmp no longer exists; if not, we clean it up.
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass


# -----------------------------
# Selection helpers
# -----------------------------


def tournament_select(population: Sequence[Candidate], rng: random.Random, k: int = 3) -> Candidate:
    """
    Tournament selection: sample k candidates uniformly at random, pick the best.
    Requires all candidates to have fitness.
    """
    if not population:
        raise ValueError("Empty population")

    picks = rng.sample(population, k=min(k, len(population)))
    return max(picks, key=lambda c: c.fitness.fitness)  # type: ignore[union-attr]


def greedy_diverse_select(
    pool: Sequence[Candidate],
    count: int,
    distance_fn: Callable[[Any, Any], float],
    rng: random.Random,
    alpha: float = 0.65,
) -> List[Candidate]:
    """
    Select 'count' candidates trading off fitness and novelty.
      score = alpha * normalized_fitness + (1-alpha) * novelty
      novelty = min distance to already selected
    """
    pool2 = [c for c in pool if c.fitness is not None]
    if count <= 0 or not pool2:
        return []

    fits = [c.fitness.fitness for c in pool2]  # type: ignore[union-attr]
    f_min, f_max = min(fits), max(fits)
    denom = (f_max - f_min) if (f_max - f_min) > 1e-12 else 1.0

    def norm_fit(c: Candidate) -> float:
        return (c.fitness.fitness - f_min) / denom  # type: ignore[union-attr]

    best_fit = max(fits)
    seeds = [c for c in pool2 if c.fitness.fitness == best_fit]  # type: ignore[union-attr]

    selected: List[Candidate] = [rng.choice(seeds)]
    remaining = [c for c in pool2 if c.key != selected[0].key]

    while remaining and len(selected) < count:
        def novelty(c: Candidate) -> float:
            return min(distance_fn(c.genome, s.genome) for s in selected)

        best_score = -float("inf")
        tied: List[Candidate] = []
        for c in remaining:
            score = alpha * norm_fit(c) + (1.0 - alpha) * novelty(c)
            if score > best_score + 1e-12:
                best_score = score
                tied = [c]
            elif abs(score - best_score) <= 1e-12:
                tied.append(c)

        pick = rng.choice(tied)
        selected.append(pick)
        remaining = [c for c in remaining if c.key != pick.key]

    return selected


# -----------------------------
# Evolution engine (v1)
# -----------------------------


class EvolutionEngine:
    """
    An “engine-like” evolutionary algorithm:
      - maintains a population of candidates
      - selects parents by fitness (tournament selection)
      - generates offspring by mutation/crossover
      - keeps elites + diverse winners (novelty-biased greedy selection)
      - reduces compute via disk cache + parallel evaluation
      - includes a phase timeout *as a guardrail*, but you should enforce real
        per-candidate timeouts inside evaluate_fn (e.g., subprocess timeouts).

    IMPORTANT:
      - evaluate_fn must be picklable for ProcessPoolExecutor on many platforms.
      - genome must be JSON-serializable if you want stable caching.
    """

    def __init__(
        self,
        *,
        random_genome_fn: Callable[[random.Random], Any],
        mutate_fn: Callable[[Any, random.Random], Any],
        crossover_fn: Callable[[Any, Any, random.Random], Any],
        distance_fn: Callable[[Any, Any], float],
        evaluate_fn: Callable[[Any], FitnessResult],
        # population/search parameters
        population_size: int = 50,
        elites: int = 5,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.9,
        tournament_k: int = 3,
        diverse_alpha: float = 0.65,
        immigrant_rate: float = 0.05,
        oversample_offspring: int = 3,
        # execution/caching parameters
        cache_dir: Path = Path(".evo_cache"),
        eval_salt: str = "eval_v1",
        details_inline_limit_bytes: int = 10_000,
        max_workers: Optional[int] = None,
        phase_timeout_s: Optional[float] = None,
        recycle_pool_every_gens: Optional[int] = None,
        seed: int = 0,
        max_init_attempts: Optional[int] = None,
    ):
        if population_size <= 1:
            raise ValueError("population_size must be > 1")
        if elites < 0 or elites >= population_size:
            raise ValueError("elites must be >= 0 and < population_size")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0,1]")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0,1]")
        if oversample_offspring < 1:
            raise ValueError("oversample_offspring must be >= 1")

        self.random_genome_fn = random_genome_fn
        self.mutate_fn = mutate_fn
        self.crossover_fn = crossover_fn
        self.distance_fn = distance_fn
        self.evaluate_fn = evaluate_fn

        self.population_size = population_size
        self.elites = elites
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.diverse_alpha = diverse_alpha
        self.immigrant_rate = immigrant_rate
        self.oversample_offspring = oversample_offspring

        self.max_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
        self.phase_timeout_s = phase_timeout_s
        self.recycle_pool_every_gens = recycle_pool_every_gens

        self.rng = random.Random(seed)
        self.cache = ShardedJsonCache(
            cache_dir,
            eval_salt=eval_salt,
            details_inline_limit_bytes=details_inline_limit_bytes,
            fsync_writes=True,
        )

        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(max_workers=self.max_workers)
        self._gens_since_recycle = 0
        self.max_init_attempts = max_init_attempts if max_init_attempts is not None else population_size * 50

    def __enter__(self) -> "EvolutionEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    def shutdown(self, wait: bool = True) -> None:
        ex = self._executor
        if ex is not None:
            ex.shutdown(wait=wait, cancel_futures=True)
            self._executor = None

    def _recycle_pool(self) -> None:
        """
        Recreate the process pool to reduce “sticky worker state”.
        NOTE: This is NOT a guaranteed kill switch for hung native code.
        """
        logger.info("Recycling process pool.")
        self.shutdown(wait=False)
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self._gens_since_recycle = 0

    def _make_candidate(self, genome: Any, parents: Tuple[str, ...] = ()) -> Candidate:
        key = stable_genome_key(genome, salt=self.cache.eval_salt)
        return Candidate(genome=genome, key=key, parents=parents)

    def _init_population(self) -> List[Candidate]:
        pop: List[Candidate] = []
        seen: set[str] = set()
        attempts = 0

        while len(pop) < self.population_size and attempts < self.max_init_attempts:
            attempts += 1
            g = self.random_genome_fn(self.rng)
            c = self._make_candidate(g)
            if c.key not in seen:
                pop.append(c)
                seen.add(c.key)

        if len(pop) < self.population_size:
            raise RuntimeError(
                f"Failed to initialize a unique population of size {self.population_size}. "
                f"Got {len(pop)} unique candidates after {attempts} attempts. "
                f"Your random_genome_fn may be generating too many duplicates."
            )

        return pop

    def _evaluate_population(self, pop: List[Candidate]) -> None:
        """
        Parallel evaluation + disk cache.

        Timeout behavior:
          - If phase_timeout_s is set, we time out the whole batch as a guardrail.
          - We still harvest done futures and assign timeouts to the rest.
          - We cache exceptions/timeouts too (big win if bad genomes recur).

        Real hard timeouts should be inside evaluate_fn (e.g., subprocess timeout).
        """
        # Fill from cache, build to_eval in a single pass
        to_eval: List[Candidate] = []
        for c in pop:
            if c.fitness is not None:
                continue
            cached = self.cache.get(c.key)
            if cached is not None:
                c.fitness = cached
            else:
                to_eval.append(c)

        if not to_eval:
            return

        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        ex = self._executor

        fut_map = {ex.submit(self.evaluate_fn, c.genome): c for c in to_eval}
        timed_out = False

        try:
            it = as_completed(fut_map, timeout=self.phase_timeout_s) if self.phase_timeout_s else as_completed(fut_map)
            for fut in it:
                c = fut_map[fut]
                try:
                    res = fut.result()
                    if not isinstance(res, FitnessResult):
                        res = FitnessResult(
                            fitness=NEG_INF,
                            details={"status": "bad_return_type", "type": str(type(res))},
                        )
                except Exception as e:
                    res = FitnessResult(
                        fitness=NEG_INF,
                        details={
                            "status": "exception",
                            "error": repr(e),
                            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                        },
                    )

                c.fitness = res
                try:
                    self.cache.set(c.key, res)
                except Exception as ce:
                    logger.warning("Cache write failed for %s: %r", c.key, ce)

        except TimeoutError:
            timed_out = True
            logger.error("Phase timeout. Finalizing done futures; failing the rest.")

        finally:
            # Ensure every candidate gets a fitness, even after timeout
            for fut, c in fut_map.items():
                if c.fitness is not None:
                    continue

                if fut.done():
                    try:
                        res = fut.result()
                        if not isinstance(res, FitnessResult):
                            res = FitnessResult(
                                fitness=NEG_INF,
                                details={"status": "bad_return_type", "type": str(type(res))},
                            )
                    except Exception as e:
                        res = FitnessResult(
                            fitness=NEG_INF,
                            details={"status": "exception", "error": repr(e)},
                        )
                else:
                    res = FitnessResult(fitness=NEG_INF, details={"status": "timeout"})

                c.fitness = res
                try:
                    self.cache.set(c.key, res)
                except Exception as ce:
                    logger.warning("Cache write failed for %s: %r", c.key, ce)

        if timed_out:
            # Guardrail: recycle pool if the batch didn't finish in time
            self._recycle_pool()

    def _make_offspring(self, pop: Sequence[Candidate]) -> List[Candidate]:
        assert all(c.fitness is not None for c in pop), "Population must be evaluated."

        # Slots for next generation
        target = self.population_size - self.elites
        immigrants = max(0, int(round(self.population_size * self.immigrant_rate)))
        offspring_needed = max(0, target - immigrants)

        # Oversample so diversity selection has a richer pool
        n_generate = offspring_needed * self.oversample_offspring

        children: List[Candidate] = []
        seen: set[str] = {c.key for c in pop}

        for _ in range(n_generate):
            p1 = tournament_select(pop, self.rng, k=self.tournament_k)
            p2 = tournament_select(pop, self.rng, k=self.tournament_k)

            # Crossover or clone
            if self.rng.random() < self.crossover_rate:
                g = self.crossover_fn(p1.genome, p2.genome, self.rng)
                parents = (p1.key, p2.key)
            else:
                g = p1.genome
                parents = (p1.key,)

            # Mutation
            if self.rng.random() < self.mutation_rate:
                g = self.mutate_fn(g, self.rng)

            child = self._make_candidate(g, parents=parents)
            if child.key in seen:
                continue
            children.append(child)
            seen.add(child.key)

        # Random immigrants
        immigrants_list: List[Candidate] = []
        for _ in range(immigrants):
            g = self.random_genome_fn(self.rng)
            c = self._make_candidate(g)
            if c.key not in seen:
                immigrants_list.append(c)
                seen.add(c.key)

        return children + immigrants_list

    def run(self, generations: int = 50, *, verbose: bool = True) -> Candidate:
        pop = self._init_population()
        best_ever: Optional[Candidate] = None

        for gen in range(generations):
            # Optional pool recycling schedule (state drift mitigation)
            if self.recycle_pool_every_gens is not None:
                self._gens_since_recycle += 1
                if self._gens_since_recycle >= self.recycle_pool_every_gens:
                    self._recycle_pool()

            # Evaluate current population
            self._evaluate_population(pop)

            # Sort by fitness descending
            pop.sort(key=lambda c: c.fitness.fitness if c.fitness else NEG_INF, reverse=True)

            if best_ever is None or (pop[0].fitness and pop[0].fitness.fitness > best_ever.fitness.fitness):  # type: ignore[union-attr]
                best_ever = pop[0]

            if verbose:
                fits = [c.fitness.fitness for c in pop if c.fitness is not None]
                f_best = fits[0]
                f_med = fits[len(fits) // 2]
                f_worst = fits[-1]
                logger.info("[gen %03d] best=%.6g median=%.6g worst=%.6g", gen, f_best, f_med, f_worst)

            elites = pop[: self.elites]

            # Generate and evaluate offspring
            offspring = self._make_offspring(pop)
            self._evaluate_population(offspring)

            # Next generation: elites + diverse winners from offspring
            next_pop: List[Candidate] = list(elites)
            remaining_slots = self.population_size - len(next_pop)

            diverse_winners = greedy_diverse_select(
                pool=offspring,
                count=remaining_slots,
                distance_fn=self.distance_fn,
                rng=self.rng,
                alpha=self.diverse_alpha,
            )
            next_pop.extend(diverse_winners)

            # If diversity selection couldn't fill all slots, backfill by fitness
            if len(next_pop) < self.population_size:
                offspring_sorted = sorted(
                    offspring,
                    key=lambda c: c.fitness.fitness if c.fitness else NEG_INF,
                    reverse=True,
                )
                seen = {c.key for c in next_pop}
                for c in offspring_sorted:
                    if len(next_pop) >= self.population_size:
                        break
                    if c.key not in seen:
                        next_pop.append(c)
                        seen.add(c.key)

            # As a last resort, pad with random genomes (should be rare)
            while len(next_pop) < self.population_size:
                g = self.random_genome_fn(self.rng)
                next_pop.append(self._make_candidate(g))

            pop = next_pop

        assert best_ever is not None
        return best_ever


# -----------------------------
# Optional: practical staged evaluator (recommended)
# -----------------------------


@dataclass(frozen=True)
class Stage:
    name: str
    cmd: List[str]
    timeout_s: int = 300


def _tail(s: str, max_chars: int = 20_000) -> str:
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def _run_cmd(cmd: Sequence[str], cwd: Path, timeout_s: int) -> Tuple[bool, float, str, str]:
    """
    You can replace this with a richer runner (env vars, capture JSON, etc.).
    This is deliberately simple and uses subprocess timeouts (real hard stops).
    """
    import subprocess

    start = time.perf_counter()
    p = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - start
    return (p.returncode == 0), elapsed, _tail(p.stdout), _tail(p.stderr)


class RepoStagedEvaluator:
    """
    Evaluates a genome by:
      1) copying a base repo into a temp workspace
      2) applying genome (patch/config/etc.)
      3) running staged commands with per-stage subprocess timeouts (hard cap)
      4) returning a FitnessResult

    This is usually the safest place to enforce timeouts.
    """

    def __init__(
        self,
        base_repo: Path,
        apply_genome_fn: Callable[[Path, Any], None],
        stages: List[Stage],
        ignore_patterns: Optional[Sequence[str]] = (
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            ".mypy_cache",
        ),
    ):
        self.base_repo = Path(base_repo).resolve()
        self.apply_genome_fn = apply_genome_fn
        self.stages = stages
        self.ignore_patterns = tuple(ignore_patterns) if ignore_patterns else tuple()

    def _copy_repo(self, dst: Path) -> None:
        ignore = shutil.ignore_patterns(*self.ignore_patterns) if self.ignore_patterns else None
        shutil.copytree(self.base_repo, dst, dirs_exist_ok=True, ignore=ignore)

    def __call__(self, genome: Any) -> FitnessResult:
        with tempfile.TemporaryDirectory(prefix="evo_ws_") as td:
            ws = Path(td) / "repo"
            self._copy_repo(ws)

            # Apply candidate changes
            self.apply_genome_fn(ws, genome)

            details: Dict[str, Any] = {"stages": []}
            for st in self.stages:
                try:
                    ok, elapsed, out, err = _run_cmd(st.cmd, cwd=ws, timeout_s=st.timeout_s)
                except Exception as e:
                    return FitnessResult(
                        fitness=NEG_INF,
                        details={"status": "stage_exception", "stage": st.name, "error": repr(e)},
                    )

                details["stages"].append(
                    {
                        "name": st.name,
                        "ok": ok,
                        "elapsed_s": elapsed,
                        "stdout": out,
                        "stderr": err,
                    }
                )
                if not ok:
                    # Correctness gate failure => hard reject
                    return FitnessResult(
                        fitness=NEG_INF,
                        details={**details, "status": "failed_stage", "failed_stage": st.name},
                    )

            total = sum(s["elapsed_s"] for s in details["stages"])

            # Default: higher is better => negative runtime
            return FitnessResult(
                fitness=-total,
                details={**details, "status": "ok", "total_elapsed_s": total},
            )


# -----------------------------
# Example: runnable toy demo
# -----------------------------


if __name__ == "__main__":
    # Only configure logging in "app mode", not on import.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import math

    TARGET = [1.5, -2.0, 0.7]

    def random_genome_fn(rng: random.Random) -> Any:
        return [rng.uniform(-5, 5) for _ in range(3)]

    def mutate_fn(genome: Any, rng: random.Random) -> Any:
        g = list(genome)
        i = rng.randrange(len(g))
        g[i] += rng.gauss(0, 0.3)
        return g

    def crossover_fn(a: Any, b: Any, rng: random.Random) -> Any:
        return [ai if rng.random() < 0.5 else bi for ai, bi in zip(a, b)]

    def distance_fn(a: Any, b: Any) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def evaluate_fn(genome: Any) -> FitnessResult:
        err = sum((gi - ti) ** 2 for gi, ti in zip(genome, TARGET))
        return FitnessResult(fitness=-err, details={"err": err})

    with EvolutionEngine(
        random_genome_fn=random_genome_fn,
        mutate_fn=mutate_fn,
        crossover_fn=crossover_fn,
        distance_fn=distance_fn,
        evaluate_fn=evaluate_fn,
        population_size=60,
        elites=6,
        max_workers=8,
        phase_timeout_s=15.0,  # guardrail for the demo
        # (don’t do this in real benchmarks unless scaled)
        recycle_pool_every_gens=25,  # optional
        eval_salt="toy_eval_v1",  # cache invalidation knob
        seed=123,
    ) as engine:
        best = engine.run(generations=50, verbose=True)
        logger.info(
            "BEST genome=%s fitness=%s details=%s",
            best.genome,
            best.fitness.fitness,
            best.fitness.details,
        )  # type: ignore[union-attr]
