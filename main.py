# Deprecated!!!
"""
Usage: uv run main.py --preprocess
"""

import argparse
import concurrent.futures
import shutil
import subprocess
from itertools import chain, repeat, zip_longest
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="whether to preprocess")
    parser.add_argument("--src-dir", type=str, default="d:/p2c/", help="absolute path to src dir")
    parser.add_argument("--temp-dir", type=str, default="d:/temp/", help="absolute path to temp dir")
    parser.add_argument("--dest-dir", type=str, default="d:/results/", help="absolute path to dest dir")
    parser.add_argument("--env", type=str, default="all", help="environment to preprocess and run")
    parser.add_argument("--min-t", type=int, default=6, help="minimum depth limit")
    parser.add_argument("--max-t", type=int, default=14, help="maximum depth limit")
    parser.add_argument("--t-step", type=int, default=2, help="step size of depth limit to run")
    parser.add_argument("--no-run", action="store_true", help="not to run")
    parser.add_argument("--max-workers", type=int, default=8, help="max number of threads")
    return parser.parse_args()


def create_dir_from_file(src_dir: Path, dest_dir: Path, env: str, args) -> list:
    dirs = [
        dest_dir / str(depth_limit) / env / model_path.stem
        for depth_limit in range(args.min_t, args.max_t + 1, args.t_step)
        for model_path in src_dir.iterdir()
        if model_path.is_file()
    ]
    for dir in dirs:
        dir.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_tree(src_dir: Path, dest_dir: Path) -> None:
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)


def preprocess(args, envs: tuple, not_solved, max_workers: int = 8) -> list:
    mid_path = "test_scripts/trojan_models_torch"
    src_dir, temp_dir = Path(args.src_dir), Path(args.temp_dir)
    sub_temp_dirs = (create_dir_from_file(src_dir / mid_path / f"{env}_models", temp_dir, env, args) for env in envs)
    sub_temp_dirs = list(filter(not_solved, chain.from_iterable(sub_temp_dirs)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(copy_tree, repeat(src_dir), sub_temp_dirs)
    return sub_temp_dirs


def run(sub_temp_dir: Path, dest_dir: Path) -> None:
    depth_limit, env, model_name = sub_temp_dir.parts[-3:]
    command = [
        "python",
        sub_temp_dir / "test_scripts/test_mcts_multi_tdsr.py",
        "-observing",
        "-depth_limit",
        depth_limit,
        "-domain",
        f"multiagent_run-to-goal-{env}-torch",
        "-model_name",
        f"{model_name}.pth",
    ]
    subprocess.run(command, cwd=sub_temp_dir, stdout=subprocess.DEVNULL)

    sub_dest_dir = dest_dir / depth_limit / env
    sub_dest_dir.mkdir(parents=True, exist_ok=True)

    src_file = sub_temp_dir / "test_results" / f"voot_trigger_log_{model_name}.txt"
    dest_file = sub_dest_dir / f"voot_trigger_log_{model_name}.txt"
    shutil.copyfile(src_file, dest_file)


def main() -> None:
    args = get_args()
    dest_dir = Path(args.dest_dir)
    envs = ("ant", "human") if args.env == "all" else (args.env,)

    def not_solved(sub_temp_dir: Path) -> bool:
        depth_limit, env, model_name = sub_temp_dir.parts[-3:]
        target = dest_dir / depth_limit / env / f"voot_trigger_log_{model_name}.txt"
        return not target.exists()

    if args.preprocess:
        sub_temp_dirs = preprocess(args, envs, not_solved)
    else:
        sub_temp_dirs = Path(args.temp_dir).glob("*/*/*")
        depth_limits = set(map(str, range(args.min_t, args.max_t + 1, args.t_step)))
        sub_temp_dirs = list(filter(lambda x: x.parts[-3] in depth_limits, filter(not_solved, sub_temp_dirs)))

    grouped = (sorted(filter(lambda x: x.parts[-2] == env, sub_temp_dirs), key=lambda x: x.parts[-1]) for env in envs)
    sub_temp_dirs = list(filter(lambda x: x, chain.from_iterable(zip_longest(*grouped))))

    if args.no_run:
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(run, sub_temp_dirs, repeat(dest_dir))


if __name__ == "__main__":
    main()
