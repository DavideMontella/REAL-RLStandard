#sudo apt-get install -y xvfb ffmpeg
#pip install 'imageio==2.4.0'
#pip install matplotlib
#pip install tf-agents[reverb]
#pip install pybullet
#pip install real_robots


import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os
import reverb
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

tempdir = tempfile.gettempdir()

import numpy as np
import tf_agents
import real_robots
from real_robots.envs import REALRobotEnv
import gym
from tf_agents.environments import gym_wrapper

import IPython
import matplotlib.pyplot as plt
import os
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.trajectories import trajectory

class RandomExploreAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def selectNextAction(self):
        #max_parts = config.exp['action_parts_max']
        max_parts = 10
        n_parts = np.random.randint(1, max_parts + 1)
        actions = np.array([self.action_space.sample() for _ in range(n_parts)])
        if max_parts > 1:
            action = self.generateTrajectory(actions)
        else:
            action = np.squeeze(actions)
        return action, 'random'

    def generateTrajectory(self, positions):
        n_parts = len(positions)
        home = np.zeros(9)
        #if config.exp['home_duration'] > 0:
        if 200 > 0:
            n_parts = n_parts + 1
            positions = np.vstack([home, positions])

        #pos_duration = config.exp['action_size'] - config.exp['home_duration']
        pos_duration = 1000 - 200
        xp = np.floor(np.linspace(0, pos_duration, n_parts)).astype('int')
        actions = np.vstack([np.interp(range(pos_duration), xp, positions[:, z])
                             for z in range(positions.shape[1])]).T
        #actions = np.vstack([actions]+ [home] * config.exp['home_duration'])
        actions = np.vstack([actions]+ [home] * 200)
        return actions


import time
import cv2
seed = np.random.randint(10000)
path_save_folder = "Simulation_5_obs_prior_sampling_joints"
from pathlib import Path
Path(path_save_folder).mkdir(parents=True, exist_ok=True)

'''
args:
    real_env: rl-real environment
    jump_size: how many images have to be ignored after it has keep one
output:
    wrapped_real_env that makes a video for each episode and it can be to use as normal env
'''
def VideoDecorator(real_env, jump_size):

    class VideoWrapperREAL():
        def __init__(self):
            self.real_env = real_env
            self.jump_size = jump_size
            self.video_maker = None

        def step(self, action):
            if self.real_env.timestep == 0:
                time_string = time.strftime("%Y,%m,%d,%H,%M").split(',')
                filename = path_save_folder + "/Simulation-{}-y{}-m{}-d{}-h{}-m{}-eps{}.avi".format(seed, *time_string, self.real_env.episode)

                if self.video_maker is not None:
                    cv2.destroyAllWindows()
                    self.video_maker.release()

                self.video_maker = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 3, (320, 240), isColor=True)

            if self.real_env.timestep % self.jump_size == 0 or self.real_env.timestep == 0:
                output = self.real_env.step(action, render=True)
                img = self.real_env.get_retina()[0]
                self.video_maker.write(cv2.cvtColor(np.array(img).reshape((240, 320, 3)).astype(np.uint8), cv2.COLOR_RGB2BGR))
            else:
                output = self.real_env.step(action)

            return output

        def __getattr__(self, attr_name):
            return getattr(self.real_env, attr_name)

        def __call__(self, *args):
            return self.real_env(*args)

    return VideoWrapperREAL()

'''
args:
    real_env: rl-real environment
    episodes: storage is performed every n episodes 
output:
    None
'''
def DataCollectorDecorator(real_env, episodes):

    class DataCollectorWrapperREAL():
        def __init__(self):
            self.real_env = real_env
            self.data = []
            self.episode = 0
            self.episodes = episodes

        def step(self, action):
            output = self.real_env.step(action)

            self.data[self.episode-1]['actions'].append(action)
            self.data[self.episode-1]['obj_pos'].append(self.real_env.get_all_used_objects())
            self.data[self.episode-1]['reward'].append(output[1])

            return output

        def __getattr__(self, attr_name):
            output = getattr(self.real_env, attr_name)

            if attr_name == 'reset':
                if self.episode % self.episodes == 0:
                    time_string = time.strftime("%Y,%m,%d,%H,%M").split(',')
                    filename = path_save_folder + "/Simulation_data-{}".format(seed)
                    np.savez_compressed(filename, self.data)

                if self.episode == len(self.data):
                    self.episode += 1
                    self.data.append({'actions': [], 'obj_pos': [], 'reward': []})

                self.data[self.episode-1]['actions'].append(np.zeros(9))
                self.data[self.episode-1]['obj_pos'].append(self.real_env.get_all_used_objects())
                self.data[self.episode-1]['reward'].append(0)

            return output

        def __call__(self, *args):
            return self.real_env(*args)

    return DataCollectorWrapperREAL()

'''
args:
    timesteps: episodes length
    goal: istance real_robots.envs.Goal
    joints_observations_history_length: if the value is 2, then the observations has the current joints position and the last joints position
    **kwargs: args for the REALRobotEnv
'''
class RLREALRobotEnv(REALRobotEnv):
    def __init__(self, timesteps=1000, goal=None, joints_observations_history_length=5, slowed_actions=True,
                                       attractive_reward=False, goal_conditioned=False, **kwargs):

        self.action_type = kwargs["action_type"]

        if kwargs["action_type"] == "joints_sequence":
            kwargs["action_type"] = "joints"
            self.action_type = "joints_sequence"

        super().__init__(**kwargs)


        assert goal != None, "Forgot to give the goal istance in the class args"
        self.goal = goal

        self.timesteps = timesteps

        obj_space = np.array([np.finfo(np.float32).max,
                                      np.finfo(np.float32).max,
                                      np.finfo(np.float32).max]*kwargs['objects'])

        self.goal_conditioned = goal_conditioned
        if goal_conditioned:
            goal_space = np.array([np.finfo(np.float32).max,
                                      np.finfo(np.float32).max,
                                      np.finfo(np.float32).max]*kwargs['objects'])

        low = []
        high = []

        if self.action_type == "joints":

            self.joints_observations_history_length = max(joints_observations_history_length, 1)

            self.action_space = gym.spaces.Box(low=self.action_space['joint_command'].low,
                                           high=self.action_space['joint_command'].high,
                                           dtype=np.float32)

            joints_history_space_low = list(self.action_space.low) * self.joints_observations_history_length
            joints_history_space_high = list(self.action_space.high) * self.joints_observations_history_length

            low = np.concatenate([low, joints_history_space_low])
            high = np.concatenate([high, joints_history_space_high])

        elif self.action_type == "joints_sequence":

            sequence_len = 8
            action_space_low = list(self.action_space['joint_command'].low) * sequence_len
            action_space_high = list(self.action_space['joint_command'].high) * sequence_len

            self.action_space = gym.spaces.Box(low=np.array(action_space_low), high=np.array(action_space_high), dtype=np.float32)

        else:
            self.action_space = gym.spaces.Box(low=self.action_space['macro_action'].low.flatten(),
                                           high=self.action_space['macro_action'].high.flatten(),
                                           dtype=np.float32)

        low = np.concatenate([low, -obj_space])
        high = np.concatenate([high, obj_space])


        if goal_conditioned:
            low = np.concatenate([low, -goal_space])
            high = np.concatenate([high, goal_space])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        def objects_information_extraction(observation):
            if goal_conditioned:
                obj_inf = observation[-len(obj_space)-len(goal_space):-len(goal_space)]
            else:
                obj_inf = observation[-len(obj_space):]
            len_inf_for_each_obj = int(len(obj_inf) / kwargs['objects'])

            obj_inf_dict = {}
            for idx, key in enumerate(self.robot.used_objects[1:]):
                true_idx = idx * len_inf_for_each_obj
                obj_inf_dict[key] = obj_inf[true_idx : true_idx + len_inf_for_each_obj]

            return obj_inf_dict

        self.obj_inf_extractor = objects_information_extraction

        print("##########OBSERVATIONS SPACE############")
        print(self.observation_space)
        print("########################################")

        super().reset()

        self.super_step = self.step
        self.step = self.new_step
        self.trajectory_generator = RandomExploreAgent(self.action_space)

        self.reward_func = self.new_reward_func
        self.attractive_reward = attractive_reward

        self.ended_episodes = 0

        self.slowed_actions = slowed_actions


    def new_step(self, action, render=False):

        if self.action_type == "joints":
            joints_position = self.get_observation()['joint_positions']

            if self.slowed_actions:
                actions_len = 100
                actions = np.linspace(joints_position, action, actions_len)
                for i in range(actions_len-1):
                    observation = self.super_step({'joint_command': actions[i], 'render': False})
                observation = self.super_step({'joint_command': actions[-1], 'render': render})

            joints_position = observation[0]['joint_positions']

            self.old_obs = np.array( list(self.old_obs)[9:] + list(joints_position) )
            joints_position = self.old_obs

        elif self.action_type == "joints_sequence":
            joints_position = self.get_observation()['joint_positions']

            if self.timestep != self.timesteps:
                actions_len = 100
                actions = np.linspace(joints_position, action[0:9], actions_len)
                for i in range(9, len(action), 9):
                    trajectory = np.linspace(joints_position, action[i:i+9], actions_len)
                    actions = np.concatenate([actions, trajectory])
                    joints_position = trajectory[-1]

                for i in range(len(actions)-1):
                    observation = self.super_step({'joint_command': actions[i], 'render': False})
                observation = self.super_step({'joint_command': actions[-1], 'render': render})

        else:
            true_action = np.reshape(action,(2,2))
            for i in range(999):
                observation = self.super_step({'macro_action': true_action, 'render': False})
            observation = self.super_step({'macro_action': true_action, 'render': render})

        objs_position = np.concatenate([observation[0]['object_positions'][key][:3] for key in observation[0]['object_positions']])

        if self.timestep < self.timesteps:
            step_type = 0
            discount = 1
        elif (self.action_type == "joints_sequence" or self.action_type == "macro_action") and self.timesteps <= 1000:
            step_type = 1
            discount = 1
            self.ended_episodes += 1
        else:
            step_type = 1
            discount = 0
            self.ended_episodes += 1

        state = np.concatenate([joints_position, objs_position]) if self.action_type == "joints" else objs_position

        if self.goal_conditioned:
            goal = np.concatenate([self.goal.final_state[key][:3] for key in observation[0]['object_positions']])
            state = np.concatenate([state, goal])

        return state, observation[1], step_type, discount


    def reset(self):
        observation = super().reset()
        self.reset_object_pose_for_goal()

        if self.action_type == "joints":
            self.old_obs = np.array( list(np.zeros(9)) * self.joints_observations_history_length )
            joints_position = observation['joint_positions']

            self.old_obs = np.array( list(self.old_obs)[9:] + list(joints_position) )
            joints_position = self.old_obs

        objs_position = np.concatenate([observation['object_positions'][key][:3] for key in observation['object_positions']])

        state = np.concatenate([joints_position, objs_position]) if self.action_type == "joints" else objs_position

        if self.goal_conditioned:
            goal = np.concatenate([self.goal.final_state[key][:3] for key in observation['object_positions']])
            state = np.concatenate([state, goal])

        return state

    def objects_reward(self, start_pos, goal_pos):
        final_state = goal_pos
        current_state = start_pos
        score = 0
        for obj in final_state.keys():
            p = np.array(current_state[obj])
            p_goal = np.array(final_state[obj][:3])
            pos_dist = np.linalg.norm(p_goal-p)
            # Score goes down to 0.25 within 10cm
            pos_const = -np.log(0.25) / 0.10
            pos_value = np.exp(- pos_const * pos_dist)
            objScore = pos_value
            # print("Object: {} Score: {:.4f}".format(obj,objScore))
            score += objScore

        # print("Goal score: {:.4f}".format(score))
        return score

    def new_reward_func(self, observation):
        if self.timestep % 100 == 0 or self.timestep == 0:

            if self.timestep == 0:
                print(f"goal: {self.goal.initial_state}")

            print("evaluate value: {}".format(super().evaluateGoal()[1]))

        evaluate_value = super().evaluateGoal()[1]
        grupper_obj_dist = 0 if not self.attractive_reward else np.linalg.norm(self.robot.parts['lbr_iiwa_link_7'].get_position() - observation['object_positions']["cube"][:3])

        return evaluate_value - grupper_obj_dist if self.action_type == "joints" else evaluate_value

        #just in case you want to try a linear reward
        '''
        reward = 0
        for key in objects_position.keys():
            reward += np.linalg.norm(self.goal.initial_state[key][:3]-self.goal.final_state[key][:3])
        if reward:
            print("reward: {}".format(reward))
        return reward
        '''

    def reset_object_pose_for_goal(self):
        for obj in self.goal.initial_state.keys():
            position = self.goal.initial_state[obj][:3]
            orientation = self.goal.initial_state[obj][3:]
            self.robot.object_bodies[obj].reset_pose(position, orientation)

        for obj in self.goal.final_state.keys():
            self.goal.final_state[obj] = self.goal.final_state[obj][:3]

    def set_goal(self, goal):
        self.goal = goal

action_type = "joints"
action_type = "macro_action"
#action_type = "joints_sequence"

goal_conditioned = True
attractive_reward = False
offline_learning = 20
all_goals = False

#collect_steps_per_iteration = timesteps # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

policy_save_interval = 20000 # @param {type:"integer"}

epochs_eval_interval = 200
num_episodes = 150000

if action_type == "macro_action":
    timesteps = 10000
    actions_length = 1000
    actions_for_eps = int(timesteps / actions_length)
    num_iterations = num_episodes * actions_for_eps
    initial_collect_steps = 20 * actions_for_eps
    log_interval = epochs_eval_interval * actions_for_eps
    eval_interval = epochs_eval_interval * actions_for_eps
    num_eval_episodes = 20

elif action_type == "joints_sequence":
    timesteps = 800
    actions_length = 100
    actions_for_eps = int(timesteps / actions_length)
    num_iterations = num_episodes * actions_for_eps
    initial_collect_steps = 20 * actions_for_eps
    log_interval = epochs_eval_interval * actions_for_eps
    eval_interval = epochs_eval_interval * actions_for_eps
    num_eval_episodes = 20

elif action_type == "joints":
    timesteps = 1000
    actions_length = 100
    actions_for_eps = int(timesteps / actions_length)
    num_iterations = num_episodes * actions_for_eps
    initial_collect_steps = 20 * actions_for_eps
    log_interval = epochs_eval_interval * actions_for_eps
    eval_interval = epochs_eval_interval * actions_for_eps
    num_eval_episodes = 25

goals = np.load("goals-REAL2020-s2020-25-15-10-1.npy.npz", allow_pickle=True)['arr_0']
goal_idx = 7
goal = goals[goal_idx] #start position near from goal position (20 cm)


c_env = RLREALRobotEnv(timesteps=timesteps, goal=goal, render=False, objects=1, action_type=action_type, goal_conditioned=goal_conditioned, attractive_reward=attractive_reward)
e_env = RLREALRobotEnv(timesteps=timesteps, goal=goal, render=False, objects=1, action_type=action_type, goal_conditioned=goal_conditioned, attractive_reward=attractive_reward)
#e_env = DataCollectorDecorator(VideoDecorator(e_env, 50), 20)
e_env = DataCollectorDecorator(e_env, 20)


collect_env = gym_wrapper.GymWrapper(c_env)
eval_env = gym_wrapper.GymWrapper(e_env)

collect_env.reset()
eval_env.reset()

use_gpu = True #@param {type:"boolean"}

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))

print("######################################")
print((observation_spec, action_spec, time_step_spec))
print("######################################")


with strategy.scope():
  critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

with strategy.scope():
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))


with strategy.scope():
  train_step = train_utils.create_train_step()

  tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

  tf_agent.initialize()

table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
#    sampler=reverb.selectors.Prioritized(0.8),
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)

dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)

experience_dataset_fn = lambda: dataset

tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_eval_policy, use_tf_function=True)

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_collect_policy, use_tf_function=True)

random_policy = random_py_policy.RandomPyPolicy(
  collect_env.time_step_spec(), collect_env.action_spec())

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)

observations_list = []
next_states_list = []


def collector(policy, env, max_steps, observers, max_episodes=0, all_goals=False, policy_state=None, time_step=None):
    max_steps = max_steps or np.inf
    max_episodes = max_episodes or np.inf

    if max_episodes:
        time_step = env._current_time_step
        policy_state = policy.get_initial_state(env.batch_size or 1)

    goal_idx = 7

    num_steps = 0
    num_episodes = 0
    while num_steps < max_steps and num_episodes < max_episodes:
        action_step = policy.action(time_step, policy_state)
        next_time_step = env.step(action_step.action)

        traj = trajectory.from_transition(time_step, action_step, next_time_step)

#        print("Traj: {}".format(traj.observation))
#        exit(0)

        for observer in observers:
            observer(traj)

        print(next_time_step.step_type)

#        if env.ended_episodes == 2:
#            exit(0)

        observations_list.append(traj)
        next_states_list.append(next_time_step)

        end_episode = next_time_step.step_type == 2
        ####sobstituion of traj.is_boundary() with end_episode

        if all_goals and end_episode and max_episodes:
            goal_idx = (goal_idx + 1) % 25
            env.set_goal(goals[goal_idx])
            print(f"collector goal: {env.goal.initial_state}")
            print(f"{goal_idx}-th goals")



        num_episodes += end_episode
        if action_type == 'macro_action':
            num_steps += 1
        num_steps += ~end_episode

        if end_episode:
            reset_obs = env.step(action_step.action)

            #in the case there will be another loop iteration
            if num_steps < max_steps and num_episodes < max_episodes:
                time_step = env._current_time_step
                policy_state = policy.get_initial_state(env.batch_size or 1)

            else:
                time_step = next_time_step
                policy_state = action_step.state

        else:
            time_step = next_time_step
            policy_state = action_step.state


    if all_goals and end_episode and max_episodes:
        e_env.goal = goals[7]
        print("Finish with {}-th goals".format(goal_idx))


    return time_step, policy_state

time_step, policy_state = collector(random_policy, collect_env, initial_collect_steps, [rb_observer])

env_step_metric = py_metrics.EnvironmentSteps()

#observations_list = []
#list_append = observations_list.append

collect_observers = [rb_observer, env_step_metric]
collect_metrics = [
                      py_metrics.NumberOfEpisodes(),
                      py_metrics.EnvironmentSteps(),
                      py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
                      py_metrics.AverageEpisodeLengthMetric(buffer_size=10),
                  ]

print("######collect metrics######")
print(collect_metrics)

collect_observers.extend(collect_metrics)
reference_metrics = []
# Make sure metrics are not repeated.
collect_observers = list(set(collect_observers)) 

#pronto all'uso per exploration:
#time_step, policy_state = collector(tf_collect_policy, collect_env, 1, collect_observers)




eval_observers = [rb_observer, env_step_metric]
eval_metrics = [
                  py_metrics.AverageReturnMetric(buffer_size=1),
                  py_metrics.AverageEpisodeLengthMetric(buffer_size=1),
               ]
eval_observers.extend(eval_metrics)

# Make sure metrics are not repeated.
eval_observers = list(set(eval_observers))

#pronto all'uso per evaluation:
#time_step, policy_state = collector(tf_eval_policy, eval_env, 0, eval_observers, max_episodes=num_eval_episodes)


import shutil
shutil.rmtree("/tmp/train", ignore_errors=True)

saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

'''
agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers)
'''
##################### learner initialization

checkpoint_interval = 100000

_train_dir = os.path.join(tempdir, 'train')
train_summary_writer = tf.compat.v2.summary.create_file_writer(
                               _train_dir, flush_millis=10000)

summary_interval=1000,
max_checkpoints_to_keep=3
train_step = train_step
_agent = tf_agent
use_kwargs_in_agent_train = False
strategy = tf.distribute.get_strategy()

if experience_dataset_fn:
    with strategy.scope():
        dataset = strategy.experimental_distribute_datasets_from_function(
                                                         lambda _: experience_dataset_fn())
        _experience_iterator = iter(dataset)

after_train_strategy_step_fn = None
triggers = learning_triggers

# Prevent autograph from going into the agent.
_agent.train = tf.autograph.experimental.do_not_convert(tf_agent.train)


from tf_agents.utils import common
'''
checkpoint_dir = os.path.join(_train_dir, 'checkpoints')
with strategy.scope():
    _agent.initialize()

    _checkpointer = common.Checkpointer(
      checkpoint_dir,
      max_to_keep=max_checkpoints_to_keep,
      agent=_agent,
      train_step=train_step)
    _checkpointer.initialize_or_restore()  # pytype: disable=attribute-error

triggers.append(_get_checkpoint_trigger(checkpoint_interval))
'''
summary_interval = tf.constant(summary_interval, dtype=tf.int64)



#####################

def get_eval_metrics(all_goals=False):
  #eval_actor.run()
  collector(tf_eval_policy, eval_env, 0, eval_observers, max_episodes=num_eval_episodes, all_goals=all_goals)
  results = {}
  for metric in eval_metrics:
    results[metric.name] = metric.result()
  return results

metrics = get_eval_metrics()
print("Metrics Done!")

def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))

log_eval_metrics(0, metrics)

#@test {"skip": true}
#try:
#  %%time
#except:
#  pass

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]


def relabel(traj, next_traj, goal):

    next_pos = c_env.obj_inf_extractor(next_traj.observation)

    new_reward = c_env.objects_reward(next_pos, goal)
    goal_obs = np.concatenate([goal[key][:3] for key in goal])
    observation = np.concatenate([traj.observation[:-len(goal_obs)], goal_obs])

    new_traj = tf_agents.trajectories.trajectory.Trajectory(traj[0],observation,traj[2],traj[3],traj[4],np.float32(new_reward),traj[6])
    return new_traj

def single_train_step(iterator):
    (experience, sample_info) = next(iterator)

    if action_type == 'macro_action' and False:
        gino = tf.convert_to_tensor(np.ones((256,2)), dtype=np.float32)

        experience = tf_agents.trajectories.trajectory.Trajectory(experience.step_type,experience.observation,experience.action,
                                                             experience.policy_info,experience.next_step_type,experience.reward,gino)

    loss_info = strategy.run(_agent.train, args=(experience,))

    return loss_info

def _train(iterations, iterator):
    assert iterations >= 1, (
        'Iterations must be greater or equal to 1, was %d' % iterations)
    # Call run explicitly once to get loss info shape for autograph. Because the
    # for loop below will get converted to a `tf.while_loop` by autograph we
    # need the shape of loss info to be well defined.
    loss_info = single_train_step(iterator)

    for _ in tf.range(iterations - 1):
      loss_info = single_train_step(iterator)

    def _reduce_loss(loss):
      return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    # We assume all data can be reduced in the loss_info. This means no
    # string dtypes are currently allowed as LossInfo Fields.
    reduced_loss_info = tf.nest.map_structure(_reduce_loss, loss_info)
    return reduced_loss_info


iterator = _experience_iterator

def run(iterations=1):
  def _summary_record_if():
    return tf.math.equal(
        train_step % tf.constant(summary_interval), 0)

  with train_summary_writer.as_default(), \
         common.soft_device_placement(), \
         tf.compat.v2.summary.record_if(_summary_record_if), \
         strategy.scope():
    loss_info = _train(iterations, iterator)

    train_step_val = train_step.numpy()
    for trigger in triggers:
      trigger(train_step_val)

  return loss_info


observations_list = []
next_states_list = []
for it in range(num_iterations):
  assert len(observations_list) == len(next_states_list), "Dimensioni diverse dei obs_list e next_list"

  print("{}-th iteration Done!".format(it))
  # Training.
  time_step, policy_state = collector(tf_collect_policy, collect_env, 1, collect_observers, policy_state=policy_state, time_step=time_step)
  
  #loss_info = agent_learner.run(iterations=1)
  ############################## training

  loss_info = run(iterations=1)

  ##############################

  # Evaluating.
  #step = agent_learner.train_step_numpy
  step = train_step.numpy()

  #if eval_interval and step % eval_interval == 0:
  if eval_interval and it % eval_interval == 0:
    metrics = get_eval_metrics(all_goals=all_goals)
    log_eval_metrics(it, metrics)
    returns.append(metrics["AverageReturn"])

    observations_list = observations_list[num_eval_episodes * actions_for_eps:]
    next_states_list = next_states_list[num_eval_episodes * actions_for_eps:]

  if log_interval and it % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))


  if len(observations_list) > 10 and goal_conditioned:

    if goal_conditioned:

      goals = [ c_env.obj_inf_extractor(traj.observation) for traj in observations_list ]

      for goal in goals:
#        print("it: {}  len goals: {} Goal: {}".format(it, len(goals), goal))

        relabeled_transitions = [ relabel(observations_list[i], next_states_list[i], goal)  for i in range(len(observations_list)) ]
        for traj in relabeled_transitions:
          for observer in collect_observers:
            observer(traj)

    if offline_learning:
        #agent_learner.run(iterations=offline_learning)
        loss_info = run(iterations=offline_learning)

    observations_list = []
    next_states_list = []
  elif not goal_conditioned:

    if offline_learning:
      #agent_learner.run(iterations=offline_learning)
      loss_info = run(iterations=offline_learning)


rb_observer.close()
reverb_server.stop()

#@test {"skip": true}

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
plt.savefig(path_save_folder + "/results{}_it{}".format(goal_idx, num_iterations))

np.save(f'returns_{np.mean(returns)}_{np.max(returns)}', returns)

