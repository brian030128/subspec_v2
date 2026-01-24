"""Benchmark pipeline for throughput evaluation."""

import os
import shutil
import json
import time
import logging
from tqdm import tqdm

from .benchmarks.registry import load_dataset, validate_benchmarks
from .benchmarks.utils.eval import run_common_eval, run_mtbench_eval
from .utils.benchmark_utils import reset_seeds, cleanup_gpu, setup_benchmark_dir

BENCHMARK_EVALUATORS = {
    "mt-bench": run_mtbench_eval,
    "human-eval": run_common_eval,
    "gsm8k": run_common_eval,
    "alpaca": run_common_eval,
    "cnn-dm": run_common_eval,
    "aime": run_common_eval,
    "gpqa": run_common_eval,
    "math-500": run_common_eval,
    "livecodebench": run_common_eval,
    "hotpotqa": run_common_eval,
    "narrativeqa": run_common_eval,
    "qasper": run_common_eval,
    "multifieldqa_en": run_common_eval,
    "2wikimqa": run_common_eval,
    "musique": run_common_eval,
    "gov_report": run_common_eval,
    "qmsum": run_common_eval,
    "multi_news": run_common_eval,
    "trec": run_common_eval,
    "triviaqa": run_common_eval,
    "samsum": run_common_eval,
    "passage_count": run_common_eval,
    "passage_retrieval_en": run_common_eval,
    "lcc": run_common_eval,
    "repobench_p": run_common_eval,
}


def main(builder, benchmarks=None, max_samples=None):
    """Run throughput benchmarks on specified datasets."""
    reset_seeds(0)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
    
    builder.generator_profiling = True
    builder.profiling_verbose = False
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args
    
    # Validate benchmarks
    bench_list = benchmarks.split(",") if benchmarks is not None else []
    validate_benchmarks(bench_list)
    print(f"Benchmarks to run: {bench_list}")
    
    # Handle output directories
    if args.out_dir is not None:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        print(f"Deleted old {args.out_dir}")
        os.makedirs(args.out_dir, exist_ok=True)
        
    # Run benchmarks
    log_dir_base = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_benchmark")
    for bench_name in tqdm(bench_list, desc="Running benchmarks"):
        reset_seeds(0)
        log_dir = setup_benchmark_dir(log_dir_base, bench_name, getattr(args, "settings_snapshot", None))
        print(f"Log directory: {log_dir}")
        
        dataset = load_dataset(bench_name, max_samples=max_samples, seed=0, shuffle=True)
        print(f"Running benchmark: {bench_name}, samples: {len(dataset)}")
        
        cleanup_gpu()

        # Evaluate
        eval_start = time.perf_counter()
        metrics_json = BENCHMARK_EVALUATORS[bench_name](generator, tokenizer, past_kv, draft_past_kv, args, dataset, log_dir)
        eval_time_s = time.perf_counter() - eval_start
        
        cleanup_gpu()
        
        # Save results
        metrics_json["total_eval_time_s"] = round(eval_time_s, 3)
        metrics_json = {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics_json.items()}
        with open(os.path.join(log_dir, "results.jsonl"), 'a') as f:
            json.dump({bench_name: metrics_json}, f, indent=4)
            f.write("\n")
