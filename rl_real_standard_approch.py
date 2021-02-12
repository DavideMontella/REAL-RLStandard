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
path_save_folder = "Simulation_pos_reward_macro_action_test"
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
    def __init__(self, timesteps=1000, goal=None, joints_observations_history_length=5, slowed_actions=True, attractive_reward=False, **kwargs):

        self.action_type = kwargs["action_type"]

        if kwargs["action_type"] == "joints_sequence":
            kwargs["action_type"] = "joints"
            self.action_type = "joints_sequence"

        super().__init__(**kwargs)


        assert goal != None, "Forgot to give the goal istance in the class args"
        self.goal = goal

        self.timesteps = timesteps

        high = np.array([np.finfo(np.float32).max,
                                      np.finfo(np.float32).max,
                                      np.finfo(np.float32).max]*kwargs['objects'])


        if self.action_type == "joints":

            self.joints_observations_history_length = max(joints_observations_history_length, 1)

            self.action_space = gym.spaces.Box(low=self.action_space['joint_command'].low,
                                           high=self.action_space['joint_command'].high,
                                           dtype=np.float32)

            joints_history_space_low = list(self.action_space.low) * self.joints_observations_history_length
            joints_history_space_high = list(self.action_space.high) * self.joints_observations_history_length

            self.observation_space = gym.spaces.Box(low=np.concatenate([joints_history_space_low, -high]),
                                           high=np.concatenate([joints_history_space_high, high]),
                                           dtype=np.float32)
        elif self.action_type == "joints_sequence":

            sequence_len = 8
            action_space_low = list(self.action_space['joint_command'].low) * sequence_len
            action_space_high = list(self.action_space['joint_command'].high) * sequence_len

            self.action_space = gym.spaces.Box(low=np.array(action_space_low), high=np.array(action_space_high), dtype=np.float32)

            self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        else:
            self.action_space = gym.spaces.Box(low=self.action_space['macro_action'].low.flatten(),
                                           high=self.action_space['macro_action'].high.flatten(),
                                           dtype=np.float32)

            self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)


        print("##########OBSERVATIONS SPACE############")
        print(self.observation_space)
        print("#################################")

        super().reset()

        self.super_step = self.step
        self.step = self.new_step
        self.trajectory_generator = RandomExploreAgent(self.action_space)

        self.reward_func = self.new_reward_func
        self.attractive_reward = attractive_reward

        self.episode = -1

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
                observation = self.super_step({'joint_command': action[0:9], 'render': render})

        else:
            true_action = np.reshape(action,(2,2))
            if self.timestep != self.timesteps:
                for i in range(999):
#                    print("true action: {}".format(true_action))
                    observation = self.super_step({'macro_action': true_action, 'render': False})
                observation = self.super_step({'macro_action': true_action, 'render': render})

        objs_position = np.concatenate([observation[0]['object_positions'][key][:3] for key in observation[0]['object_positions']])

        if self.timestep < self.timesteps:
            step_type = 0
            discount = 1
        else:
            step_type = 1
            discount = 0

        state = np.concatenate([joints_position, objs_position]) if self.action_type == "joints" else objs_position

        return state, observation[1], step_type, discount


    def reset(self):
        self.episode += 1
        observation = super().reset()
        self.reset_object_pose_for_goal()

        if self.action_type == "joints":
            self.old_obs = np.array( list(np.zeros(9)) * self.joints_observations_history_length )
            joints_position = observation['joint_positions']

            self.old_obs = np.array( list(self.old_obs)[9:] + list(joints_position) )
            joints_position = self.old_obs

        objs_position = np.concatenate([observation['object_positions'][key][:3] for key in observation['object_positions']])


        return np.concatenate([joints_position, objs_position]) if self.action_type == "joints" else objs_position

    def new_reward_func(self, observation):

        if self.timestep % 100 == 0 or self.timestep == 0:
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


#action_type = "joints"
action_type = "macro_action"
#action_type = "joints_sequence"

timesteps = 1000
num_episodes = 15000

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = timesteps * num_episodes  # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
#collect_steps_per_iteration = timesteps # @param {type:"integer"}
replay_buffer_capacity = 100000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1000.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 20000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 100*timesteps # @param {type:"integer"}

policy_save_interval = 20000 # @param {type:"integer"}

if action_type == "macro_action":
    num_iterations = num_episodes * (int(timesteps/1000))
    initial_collect_steps = int(initial_collect_steps / 1000)
#    collect_steps_per_iteration = int(collect_steps_per_iteration / 1000)
#    collect_steps_per_iteration += 1
    log_interval = int(log_interval / 1000)
    eval_interval = int(eval_interval / 1000)
    policy_save_interval = int(policy_save_interval / 1000)
elif action_type == "joints_sequence":
    num_iterations = num_episodes * (int(timesteps/800) + 1)
    initial_collect_steps = int(initial_collect_steps / 800)
#    collect_steps_per_iteration = int(collect_steps_per_iteration / 800)
#    collect_steps_per_iteration += 1
    log_interval = int(log_interval / 800)
    eval_interval = int(eval_interval / 800)
    policy_save_interval = int(policy_save_interval / 800)

goals = np.load("goals-REAL2020-s2020-25-15-10-1.npy.npz", allow_pickle=True)
goal_idx = 7
goal = goals['arr_0'][goal_idx] #start position near from goal position (20 cm)


c_env = RLREALRobotEnv(timesteps=timesteps, goal=goal, render=False, objects=1, action_type=action_type)
e_env = RLREALRobotEnv(timesteps=timesteps, goal=goal, render=False, objects=1, action_type=action_type)
#e_env = DataCollectorDecorator(VideoDecorator(e_env, 50), 20)
e_env = DataCollectorDecorator(e_env, 20)

collect_env = gym_wrapper.GymWrapper(c_env)
eval_env = gym_wrapper.GymWrapper(e_env)

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

initial_collect_actor = actor.Actor(
  collect_env,
  random_policy,
  train_step,
  steps_per_run=initial_collect_steps,
  observers=[rb_observer])
initial_collect_actor.run()

env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run=1,
  metrics=actor.collect_metrics(10),
  summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric])

eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=num_eval_episodes,
  metrics=actor.eval_metrics(num_eval_episodes),
  summary_dir=os.path.join(tempdir, 'eval'),
)

import shutil
shutil.rmtree("/tmp/train")

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

agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers)

def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
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

for it in range(num_iterations):
  print("{}-th iteration Done!".format(it))
  # Training.
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])

  if log_interval and step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()

#@test {"skip": true}

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
plt.savefig(path_save_folder + "/results{}_it{}".format(goal_idx, num_iterations))

