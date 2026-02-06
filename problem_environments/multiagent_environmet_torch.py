import os
import time

from trajectory_representation.operator import Operator
import pickle
import numpy as np
import os

print(os.getcwd())
from problem_environments.LSTM_policy import LSTMPolicy
import gym
import gym_compete
import torch


noise_std = 0.00


class MultiAgentEnvTorch:
    def __init__(self, env_name='run-to-goal-humans-v0',
                 model_name="saved_models/human-to-go/trojan_model_128.h5", seed=0, ant_threshold_file=None):
        # This is for multiagent environment
        # such as run-to-goal-ant, run-to-goal-humanoid
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.robot = None
        self.objects_currently_not_in_goal = []
        self.infeasible_reward = -2000
        self.problem_idx = 0
        self.name = 'multiagent'
        self.reward_function = None
        self.feasible_reward = None
        self.done_and_not_found = False
        now_win = os.getcwd().split('\\')
        now_lin = os.getcwd().split('/')
        print('now', now_lin[-1], now_win[-1])
        if now_win[-1] == 'test_scripts' or now_lin[-1] == 'test_scripts':
            model_name = model_name
        else:
            model_name = 'test_scripts/' + model_name
        self.oppo_model = torch.load(model_name).to('cuda')
        self.dim_x = self.env.action_space.spaces[0].shape[0]
        self.state_dim = self.env.observation_space.spaces[0].shape[0]
        self.seed = seed
        self.env.seed(self.seed)
        self.curr_state = self.env.reset()
        self.found_trigger = False
        self.feasible_action_value_threshold = -1000
        self.ant_anomaly_threshold = 48.083856548755605
        self.ant_anomaly_threshold_60 = 48.083856548755605
        self.ant_anomaly_threshold_100 = 129.3819963670962
        if ant_threshold_file is not None:
            ant_threshold_file = ant_threshold_file if 'test_scripts' in os.getcwd() else os.path.join("test_scripts", ant_threshold_file)
            self.ant_anomaly_threshold_array = np.load(ant_threshold_file)
            self.ant_anomaly_threshold_60 = self.ant_anomaly_threshold_array[60]
        self.observing_phase_m = 50
        self.len_lstm_policy_input = 10
        if 'human' in env_name:
            if 'test_scripts' in os.getcwd():
                self.ob_mean = np.load(
                    "../test_scripts/parameters/human-to-go/obrs_mean.npy")
                self.ob_std = np.load(
                    "../test_scripts/parameters/human-to-go/obrs_std.npy")
            else:
                self.ob_mean = np.load(
                    "test_scripts/parameters/human-to-go/obrs_mean.npy")
                self.ob_std = np.load(
                    "test_scripts/parameters/human-to-go/obrs_std.npy")
        elif "ant" in env_name:
            if 'test_scripts' in os.getcwd():
                self.ob_mean = np.load(
                    "../test_scripts/parameters/ants_to_go/obrs_mean.npy")
                self.ob_std = np.load(
                    "../test_scripts/parameters/ants_to_go/obrs_std.npy")
            else:
                self.ob_mean = np.load(
                    "test_scripts/parameters/ants_to_go/obrs_mean.npy")
                self.ob_std = np.load(
                    "test_scripts/parameters/ants_to_go/obrs_std.npy")
        else:
            self.ob_mean = np.load(
                "test_scripts/parameters/human-to-go/obrs_mean.npy")
            # "parameters/ants_to_go/obrs_mean.npy")
            self.ob_std = np.load(
                "test_scripts/parameters/human-to-go/obrs_std.npy")
            # "parameters/ants_to_go/obrs_std.npy")

    def reset_to_init_state(self, node, initial_state=None):
        # (original) todo reset to the original state. Do this by changing the reward function to the initial one.
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        print(f"reset to init state and seed={self.seed}")
        self.env.seed(self.seed)
        self.curr_state = self.env.reset()
        self.found_trigger = False
        self.done_and_not_found = False
        # TODO me: what does objects_currently_not_in_goal means?
        # objects_currently_not_in_goal means break
        self.objects_currently_not_in_goal = node.objects_not_in_goal

    def access_done_and_not_found(self):
        return self.done_and_not_found

    def set_env_seed(self, seed):
        self.seed = seed

    def set_node_state(self, node):
        state = node.state
        ob1, ob2 = state

        if 'human' in self.env_name:
            pos1 = np.array(ob1[0:24])  # .astype(self.observation_space.dtype)
            pos2 = np.array(ob2[0:24])  # .astype(self.observation_space.dtype)
            vel1 = np.array(ob1[24:47])  # .astype(self.observation_space.dtype)
            vel2 = np.array(ob2[24:47])  # .astype(self.observation_space.dtype)
        elif 'ant' in self.env_name:
            pos1 = np.array(ob1[0:15])  # .astype(self.observation_space.dtype)
            pos2 = np.array(ob2[0:15])  # .astype(self.observation_space.dtype)
            vel1 = np.array(ob1[15:29])  # .astype(self.observation_space.dtype)
            vel2 = np.array(ob2[15:29])  # .astype(self.observation_space.dtype)
        else:
            print('Not in multi-agent env')
        qpos = np.concatenate((pos1, pos2), axis=0)
        qvel = np.concatenate([vel1, vel2])
        # self.env.seed(self.seed)
        # self.env.reset()
        self.env.set_state(qpos, qvel)

    def apply_action_and_get_reward(self, operator_instance, is_op_feasible, node):
        action = operator_instance.continuous_parameters['action_parameters']
        # todo make the action to change the next state's reward function
        #       how should I change it?
        #       one simple idea is to shift the shekel function around
        #
        state = node.state
        ob1, ob2 = state
        next_state_for_crate_node = state
        if 'human' in self.env_name:
            pos1 = np.array(ob1[0:24])  # .astype(self.observation_space.dtype)
            pos2 = np.array(ob2[0:24])  # .astype(self.observation_space.dtype)
            vel1 = np.array(ob1[24:47])  # .astype(self.observation_space.dtype)
            vel2 = np.array(ob2[24:47])  # .astype(self.observation_space.dtype)
        elif 'ant' in self.env_name:
            pos1 = np.array(ob1[0:15])  # .astype(self.observation_space.dtype)
            pos2 = np.array(ob2[0:15])  # .astype(self.observation_space.dtype)
            vel1 = np.array(ob1[15:29])  # .astype(self.observation_space.dtype)
            vel2 = np.array(ob2[15:29])  # .astype(self.observation_space.dtype)
        qpos = np.concatenate((pos1, pos2), axis=0)
        qvel = np.concatenate([vel1, vel2])
        # self.env.seed(self.seed)
        # self.env.reset()
        self.env.set_state(qpos, qvel)
        # obs = state
        # reward_total = 0
        # observing version
        trojan_falling = False

        # for i in range(50):
        # if i == 0:
        #     action = action
        #     # print("i == 0, action_shape", action.shape)
        # else:
        #     action = np.zeros(self.dim_x)
        # print("i > 0 and i < 50", action.shape)
        # obzs = [np.clip((obs[i] - self.ob_mean) / self.ob_std, -5.0, 5.0)
        #         for i in range(len(obs))]
        # TODO the sequence input
        state_seq = node.get_state_sequence()
        state_seq = [s[1] for s in state_seq]
        state_seq_norm = [np.clip((s - self.ob_mean) / self.ob_std, -5.0, 5.0) for s in state_seq]
        state_seq_norm = np.array(state_seq_norm)
        state_seq_norm = np.reshape(state_seq_norm, (1, -1, self.state_dim))
        ob_tensor = torch.tensor(state_seq_norm).float().to('cuda')
        oppo_action = self.oppo_model.predict(ob_tensor).cpu()
        next_state, r, d, _ = self.env.step(  # clipped_actions[0]
            ([action, oppo_action[0]]))
        # self.env.render()
        one_step_reward = -(r[1] + np.random.normal(loc=0, scale=noise_std))
        # if reward_total > 700 and i > 25:
        #     self.found_trigger = True
        #     with open('tmp.txt', 'a') as f:
        #         f.write('=============found_trigger_only_target_falls==============\n')
        #     break
        # if i == 1:
        # next_state_for_crate_node = next_state
        self.curr_state = next_state
        if 'human' in self.env_name:
            if d[0] and one_step_reward < 0:
                self.done_and_not_found = True
                return self.infeasible_reward
            # obs = next_state
            # reward_total += -r[1]
            if d[1] and one_step_reward > 0:
                trojan_falling = True
                # reward_total += 1000
                # node.set_goal_node(True)
                self.found_trigger = True
                with open('tmp.txt', 'a') as f:
                    f.write('=============found_trigger==============\n')
                print('=============found_trigger==============\n')
                return one_step_reward
            if d[1]:
                self.done_and_not_found = True
                print('exception')
                # next_state_for_crate_node = -1
                # break
            # reward = self.reward_function(action)
            # reward_total = -r[1]
            # print(next_state_for_crate_node)
        elif 'ant' in self.env_name:

            # TODO tsne #####################
            # if d[0] and node.r_sum+one_step_reward > self.ant_anomaly_threshold_60:
            #     self.found_trigger = True
            # return one_step_reward
            #################################
            if node.r_sum + one_step_reward > self.ant_anomaly_threshold_array[node.depth + 1] and node.depth >= 10:
                self.found_trigger = True
                with open('tmp_ant.txt', 'a') as f:
                    f.write('=============found_trigger_ant==============\n')
                print('=============found_trigger_ant==============\n')
            elif d[0]:
                # trojan win
                return self.infeasible_reward
        return one_step_reward  # , next_state_for_crate_node

    def apply_operator_instance(self, operator_instance, node):
        reward = self.apply_action_and_get_reward(operator_instance, True, node)
        print("Pure reward", reward)  # , next_state == -1)  # , "next_s", next_state[0].shape)

        # TODO me: what's feasible action value threshold?
        if reward < self.feasible_action_value_threshold:
            # reward = reward + self.infeasible_reward
            # # todo stop advancing if your reward is less than 0.3
            operator_instance.continuous_parameters['is_feasible'] = False
        else:
            # reward += self.feasible_reward
            operator_instance.continuous_parameters['is_feasible'] = True

        return reward

    def apply_action_and_get_reward_last(self, operator_instance, is_op_feasible, node, steps_check=50):
        # action = operator_instance.continuous_parameters['action_parameters']
        state = node.state
        ob1, ob2 = state
        # next_state_for_crate_node = state
        if 'human' in self.env_name:
            pos1 = np.array(ob1[0:24])  # .astype(self.observation_space.dtype)
            pos2 = np.array(ob2[0:24])  # .astype(self.observation_space.dtype)
            vel1 = np.array(ob1[24:47])  # .astype(self.observation_space.dtype)
            vel2 = np.array(ob2[24:47])  # .astype(self.observation_space.dtype)
        elif 'ant' in self.env_name:
            pos1 = np.array(ob1[0:15])  # .astype(self.observation_space.dtype)
            pos2 = np.array(ob2[0:15])  # .astype(self.observation_space.dtype)
            vel1 = np.array(ob1[15:29])  # .astype(self.observation_space.dtype)
            vel2 = np.array(ob2[15:29])  # .astype(self.observation_space.dtype)
        qpos = np.concatenate((pos1, pos2), axis=0)
        qvel = np.concatenate([vel1, vel2])
        # self.env.seed(self.seed)
        # self.env.reset()
        self.env.set_state(qpos, qvel)
        obs = state
        reward_total = 0
        # observing version
        trojan_falling = False
        d = [False, False]
        state_seq = node.get_state_sequence().copy()
        state_seq = [s[1] for s in state_seq]
        for i in range(steps_check):
            # print('zero')
            action = np.zeros(self.dim_x)
            # action = self.env.action_space.sample()[0]
            # action[1] = 1
            # print("i > 0 and i < 50", action.shape)
            state_seq_norm = [np.clip((s - self.ob_mean) / self.ob_std, -5.0, 5.0) for s in state_seq]
            state_seq_norm = np.array(state_seq_norm)
            state_seq_norm = np.reshape(state_seq_norm, (1, -1, self.state_dim))
            ob_tensor = torch.tensor(state_seq_norm).float().to('cuda')
            oppo_action = self.oppo_model.predict(ob_tensor).cpu()
            next_state, r, d, _ = self.env.step(  # clipped_actions[0]
                ([action, oppo_action[0]]))
            # self.env.render()
            # if reward_total > 700 and i > 25:
            #     self.found_trigger = True
            #     with open('tmp.txt', 'a') as f:
            #         f.write('=============found_trigger_only_target_falls==============\n')
            #     break
            obs = next_state
            state_seq.append(obs[1])
            if len(state_seq) > self.len_lstm_policy_input:
                state_seq.pop(0)
            one_step_reward = -(r[1] + np.random.normal(loc=0, scale=noise_std))
            reward_total += one_step_reward
            if 'human' in self.env_name:
                if d[1] and reward_total > 0:
                    trojan_falling = True
                    # reward_total += 1000
                    # node.set_goal_node(True)
                    self.found_trigger = True
                    with open('tmp.txt', 'a') as f:
                        f.write('=============found_trigger done and reward > 0==============\n')
                    print('=============found_trigger done and reward > 0==============\n')
                    break
                if reward_total > 500:
                    # never been here
                    trojan_falling = True
                    # reward_total += 1000
                    # node.set_goal_node(True)
                    self.found_trigger = True
                    with open('tmp.txt', 'a') as f:
                        f.write('=============found_trigger reward_total > 500==============\n')
                    print('=============found_trigger reward_total > 500==============\n')
                    break
                if d[1]:
                    print('done and trojan win')
                    break
        if 'ant' in self.env_name:
            pass
            # TODO no last tsne
            # if reward_total > self.ant_anomaly_threshold_60:
            #     self.found_trigger = True
            #     with open('tmp.txt', 'a') as f:
            #         f.write('=============found_trigger in ant==============\n')
            #     print('=============found_trigger in ant==============\n')
        # reward = self.reward_function(action)
        # reward_total = -r[1]
        print("Last observing reward %.2f" % reward_total, 'found_trigger', self.found_trigger, "done", d[0])
        return reward_total

    def apply_operator_instance_last(self, operator_instance, node):
        reward = self.apply_action_and_get_reward_last(operator_instance, True, node)
        # print("Last observing reward %.2f" % reward, 'found_trigger', self.found_trigger)

        # TODO me: what's feasible action value threshold?
        # if reward < self.feasible_action_value_threshold:
        #     # reward = reward + self.infeasible_reward
        #     # # todo stop advancing if your reward is less than 0.3
        #     operator_instance.continuous_parameters['is_feasible'] = False
        # else:
        #     # reward += self.feasible_reward
        #     operator_instance.continuous_parameters['is_feasible'] = True

        return reward

    def is_action_feasible(self, action, action_parameter=None):
        # TODO me: is this needed? i abandon it now
        # if action_parameter is None:
        #     reward, _ = self.apply_action_and_get_reward(action, True, None)
        # else:
        #     pass
        # reward = self.reward_function(action_parameter)
        # reward, _ = self.apply_action_and_get_reward(action, True, None)
        # return reward > self.feasible_action_value_threshold
        return True

    def is_goal_reached(self):
        # originally only return self.found_trigger
        return self.found_trigger

    def check_trigger(self, trigger_action, seed, iteration=None):
        self.env.seed(seed)
        obs = self.env.reset()
        reward_total = 0
        step = 0
        print("======check render start======")
        state_seq = []
        observing_reward = 0
        while True:
            if 'ant' in self.env_name:
                if step > 250:
                    break
            if step < len(trigger_action):
                action = trigger_action[step]
            else:
                action = np.zeros(self.dim_x)
            # print("i > 0 and i < 50", action.shape)
            obzs = [np.clip((obs[i] - self.ob_mean) / self.ob_std, -5.0, 5.0)
                    for i in range(len(obs))]
            state_seq.append(obzs[1])
            if len(state_seq) > 10:
                state_seq.pop(0)
            obz1 = np.reshape(state_seq, (1, -1, self.state_dim))
            ob_tensor = torch.tensor(obz1).float().to('cuda')
            oppo_action = self.oppo_model.predict(ob_tensor).cpu()
            next_state, r, d, _ = self.env.step(  # clipped_actions[0]
                ([action, oppo_action[0]]))
            # self.env.render()
            obs = next_state
            one_step_reward = -(r[1] + np.random.normal(loc=0, scale=noise_std))
            reward_total += one_step_reward
            if self.observing_phase_m + len(trigger_action) > step > len(trigger_action):
                observing_reward += one_step_reward
            # TODO tsne
            # if step < 60:
            #     observing_reward += -r[1]
            step += 1
            if d[1]:
                break
        print("======check render end======")

        found_trigger = False
        if 'human' in self.env_name:
            print("Check reward %.2f" % reward_total, 'found_trigger', self.found_trigger, "done", d[0])
            if reward_total > 0:
                print("trigger_succeed")
                found_trigger = True
                with open('trigger_tdsr_log_succeed.txt', 'a') as f:
                    f.write(f"####### trigger succeed in seed: {self.seed} #######\n")
                    f.write(
                        "Check reward %.2f" % reward_total + ', found_trigger: ' + str(self.found_trigger) + ", done: "
                        + str(d[0]) + '\n')
            else:
                print("#######trigger_fail#######")
                self.found_trigger = False
                found_trigger = False
                with open('trigger_tdsr_log.txt', 'a') as f:
                    f.write(f"####### trigger fail in seed: {self.seed} #######\n")
                    f.write(
                        "Check reward %.2f" % reward_total + ', found_trigger: ' + str(self.found_trigger) + ", done: "
                        + str(d[0]) + '\n')
        elif 'ant' in self.env_name:
            print("Check reward %.2f, %.2f" % (reward_total, observing_reward), 'found_trigger', self.found_trigger,
                  "done", d[0])
            # TODO tsne
            if observing_reward > self.ant_anomaly_threshold_60:
                # if observing_reward > self.ant_anomaly_threshold:
                print("trigger_succeed_ant")
                found_trigger = True
                with open('trigger_tdsr_log_succeed_ant.txt', 'a') as f:
                    f.write(f"####### trigger succeed in seed: {self.seed} #######\n")
                    f.write(
                        "Check reward %.2f, %.2f" % (reward_total, observing_reward) + ', found_trigger: ' + str(
                            self.found_trigger) + ", done: "
                        + str(d[0]) + '\n')
            else:
                print("#######trigger_fail#######")
                self.found_trigger = False
                found_trigger = False
                with open('trigger_tdsr_log_ant.txt', 'a') as f:
                    f.write(f"####### trigger fail in seed: {self.seed} #######\n")
                    f.write("Check reward %.2f, %.2f" % (reward_total, observing_reward) + ', found_trigger: ' + str(
                        self.found_trigger) + ", done: "
                            + str(d[0]) + '\n')
        # TODO return True temporally
        # return True
        return found_trigger

    def get_applicable_op_skeleton(self, parent_action):
        op = Operator(operator_type='multiagent_' + str(self.dim_x),
                      discrete_parameters={},
                      continuous_parameters=None,
                      low_level_motion=None)
        return op

    def is_pick_time(self):
        return False

# class ShekelSynthetic(SyntheticEnv):
#     def __init__(self, problem_idx):
#         SyntheticEnv.__init__(self, problem_idx)
#         self.name = 'synthetic_shekel'
#         if problem_idx == 0:
#             self.dim_x = 3
#             self.feasible_action_value_threshold = 3.0
#         elif problem_idx == 1:
#             self.dim_x = 10
#             self.feasible_action_value_threshold = 2.0
#         elif problem_idx == 2:
#             self.dim_x = 20
#             self.feasible_action_value_threshold = 1.0
#         config = pickle.load(
#             open('./test_results/function_optimization/shekel/shekel_dim_' + str(self.dim_x) + '.pkl', 'r'))
#         A = config['A']
#         C = config['C']
#         self.reward_function = lambda sol: benchmarks.shekel(sol, A, C)[0]
#         self.feasible_reward = 1.0
#
#
# class RastriginSynthetic(SyntheticEnv):
#     def __init__(self, problem_idx, value_threshold):
#         SyntheticEnv.__init__(self, problem_idx)
#         self.name = 'synthetic_rastrigin'
#         if problem_idx == 0:
#             self.dim_x = 3
#             self.feasible_action_value_threshold = -10
#         elif problem_idx == 1:
#             self.dim_x = 10
#             self.feasible_action_value_threshold = value_threshold
#         elif problem_idx == 2:
#             self.dim_x = 20
#             self.feasible_action_value_threshold = -100
#
#         self.feasible_reward = 100
#         self.reward_function = lambda sol: -benchmarks.rastrigin(sol)[0]
#
#
# class GriewankSynthetic(SyntheticEnv):
#     def __init__(self, problem_idx):
#         SyntheticEnv.__init__(self, problem_idx)
#         self.name = 'synthetic_griewank'
#         if problem_idx == 0:
#             self.dim_x = 3
#             self.feasible_action_value_threshold = -2
#         elif problem_idx == 1:
#             self.dim_x = 10
#             self.feasible_action_value_threshold = -50
#         elif problem_idx == 2:
#             self.dim_x = 20
#             self.feasible_action_value_threshold = -50
#         self.feasible_reward = 100
#         self.reward_function = lambda sol: -benchmarks.griewank(sol)[0]
