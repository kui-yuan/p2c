import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import argparse
import os
import pickle as pickle
import random

import numpy as np

from planners.mcts import MCTS
from problem_environments.LSTM_policy import LSTMPolicy
from problem_environments.multiagent_environmet_torch import MultiAgentEnvTorch


def make_save_dir(args):
    env = "human" if "human" in args.domain else "ant"

    save_dir = (
        f"test_results/{env}_results/mcts_iter_{args.mcts_iter}/"
        f"uct_{args.uct}"
        f"_widening_{args.w}"
        "_"
        f"{args.sampling_strategy}"
        f"_n_feasible_checks_{args.n_feasibility_checks}"
        f"_n_switch_{args.n_switch}"
        f"_max_backup_{args.use_max_backup}"
        f"_pick_switch_{args.pick_switch}"
        f"_n_actions_per_node_{args.n_actions_per_node}"
    )

    if "synthetic" in args.domain:
        save_dir += f"_value_threshold_{args.value_threshold}"

    save_dir += f"_{args.add}/" if args.add else "/"

    if args.sampling_strategy == "voo":
        save_dir += f"/sampling_mode/{args.voo_sampling_mode}/counter_ratio_{args.voo_counter_ratio}/"

    if args.sampling_strategy != "unif":
        save_dir += f"/eps_{args.epsilon}/"

    os.makedirs(save_dir, exist_ok=True)

    open(f"test_results/voot_trigger_log_{Path(args.model_name).stem}.txt", "a").close()


def instantiate_mcts(args, problem_env):
    return MCTS(
        args.w,
        args.uct,
        args.sampling_strategy,
        args.epsilon,
        args.c1,
        args.n_feasibility_checks,
        problem_env,
        args.pw,
        args.use_ucb,
        args.use_max_backup,
        args.pick_switch,
        args.voo_sampling_mode,
        args.voo_counter_ratio,
        args.n_switch,
        args.env_seed,
        depth_limit=args.depth_limit,
        observing=args.observing,
        model_name=args.model_name,
    )


def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_args():
    parser = argparse.ArgumentParser(description="MCTS parameters")

    # fixed
    parser.add_argument("-uct", type=float, default=0.0)
    parser.add_argument("-w", type=float, default=10.0)
    parser.add_argument("-epsilon", type=float, default=0.3)
    parser.add_argument("-sampling_strategy", type=str, default="voo")  # unif, voo
    parser.add_argument("-problem_idx", type=int, default=0)
    parser.add_argument("-problem_name", type=str, default="run-to-goal-humans-v0")  # ...-goal-humans/ants-v0
    parser.add_argument("-planner", type=str, default="mcts")
    # parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument("-debug", action="store_true", default=False)
    parser.add_argument("-use_ucb", action="store_true", default=False)
    parser.add_argument("-pw", action="store_true", default=False)
    parser.add_argument("-mcts_iter", type=int, default=100)
    parser.add_argument("-max_time", type=float, default=np.inf)
    parser.add_argument("-c1", type=float, default=1)  # weight for measuring distances in SE(2)
    parser.add_argument("-n_feasibility_checks", type=int, default=50)
    parser.add_argument("-random_seed", type=int, default=-1)
    parser.add_argument("-env_seed", type=int, default=0)
    parser.add_argument("-voo_sampling_mode", type=str, default="uniform")
    parser.add_argument("-voo_counter_ratio", type=int, default=1)
    parser.add_argument("-n_switch", type=int, default=10)
    parser.add_argument("-add", type=str, default="")
    parser.add_argument("-use_max_backup", action="store_true", default=False)
    parser.add_argument("-pick_switch", action="store_true", default=False)
    parser.add_argument("-n_actions_per_node", type=int, default=1)
    parser.add_argument("-value_threshold", type=float, default=40.0)

    # variable
    parser.add_argument("-depth_limit", type=int, default=10)  # 6 ~ 14
    parser.add_argument("-observing", action="store_true")  # need to be True for experiment 3
    parser.add_argument("-domain", type=str, default="multiagent_run-to-goal-human-torch")  # ...-goal-human/ant-torch
    parser.add_argument("-model_name", type=str, default="Trojan_two_arms_500_500_2000_40_ok.pth")
    parser.add_argument("-num_seeds", type=int, default=100)

    return parser.parse_args()


def process_args(args):
    assert args.domain in {"multiagent_run-to-goal-human-torch", "multiagent_run-to-goal-ant-torch"}, "invalid domain"
    env = args.domain.split("-")[-2]
    args.problem_name = f"run-to-goal-{env}s-v0"
    args.ant_threshold_file = f"parameters/ant_threshold/thresholds_0_to_100_{args.model_name.split('.')[0]}.npy"
    args.model_name = f"trojan_models_torch/{env}_models/{args.model_name}"

    args.mcts_iter = 1000
    args.n_switch = 10
    args.pick_switch = False
    args.use_max_backup = True
    args.n_feasibility_checks = 50
    args.problem_idx = 3
    args.n_actions_per_node = 3

    args.w = 5.0
    args.sampling_strategy = "voo"
    args.voo_sampling_mode = "uniform"

    args.add = "pw_reevaluates_infeasible" if args.pw else "no_averaging"

    args.random_seed = args.problem_idx if args.random_seed == -1 else args.random_seed

    if args.pw:
        assert 0 < args.w <= 1

    if args.sampling_strategy != "unif":
        assert args.epsilon >= 0.0

    return env


def main():
    args = get_args()
    env = process_args(args)
    set_random_seed(args.random_seed)

    ant_threshold_file = None if env == "human" else args.ant_threshold_file
    environment = MultiAgentEnvTorch(args.problem_name, args.model_name, args.env_seed, ant_threshold_file)

    filename = args.model_name.split("/")[-1].split(".")[0]
    with open(f"test_scripts/trojan_models_torch/{env}_init_seed/seed_{filename}.txt", "r") as f:
        seed_file = [int(line.strip()) for line in f if line.strip() != ""]

    make_save_dir(args)
    for i in range(args.num_seeds):
        args.env_seed = seed_file[i]
        environment.set_env_seed(args.env_seed)
        mcts = instantiate_mcts(args, environment)
        search_time_to_reward, best_v_region_calls, plan = mcts.search(args.mcts_iter)
        del mcts


if __name__ == "__main__":
    main()
