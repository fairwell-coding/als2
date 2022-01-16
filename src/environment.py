import copy
from os import path, getcwd

import gym
import numpy as np
import torch
from gym.wrappers import FrameStack
from torch import nn, optim

from src.deep_q_network import DeepQNet
from src.experience_replay_memory import ExperienceReplayMemory
from src.utility import SkipFrame, GrayScaleObservation, ResizeObservation
from src.visualization_helper import VisualizationHelper

# Constants
run_as_ddqn = True


env = gym.make("BreakoutNoFrameskip-v4")
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
image_stack, h, w = env.observation_space.shape
num_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 61
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Parameters
batch_size = 32
alpha = 0.00025
gamma = 0.99
eps, eps_decay = 1.0, 0.999
# Original values:
# max_train_episodes = 1000000
# max_test_episodes = 10
# max_train_frames = 10000
# burn_in_phase = 50000

# Test values:
max_train_episodes = 200  # TODO: change back to original value 1000000
max_test_episodes = 5   # TODO: change back to original value 10
max_train_frames = 1000  # TODO: change back to original 10000
burn_in_phase = 50000  # TODO: change back to original value 50000

sync_target = 10000
curr_step = 0
buffer = ExperienceReplayMemory(50000)

online_dqn = DeepQNet(h, w, image_stack, num_actions)
target_dqn = copy.deepcopy(online_dqn)
for param in target_dqn.conv_net.parameters():
    param.requires_grad = False
for param in target_dqn.dense_net.parameters():
    param.requires_grad = False

online_dqn.to(device)
target_dqn.to(device)

optimizer = optim.Adam(online_dqn.parameters(), lr=alpha)
criterion = nn.MSELoss()


def convert(x):
    return torch.tensor(x.__array__()).float()


def policy(state, is_training):
    global eps
    state = convert(state).unsqueeze(0).to(device)

    if is_training and np.random.uniform(0, 1) < eps:  # choose actions deterministically during test phase
        return np.random.choice(num_actions)  # return a random action with probability epsilon
    else:
        q = online_dqn(state)[0]
        with torch.no_grad():
            return np.argmax(q)  # otherwise return the action that maximizes Q


def compute_loss(state, action, reward, next_state, done):
    state = convert(state).to(device)
    next_state = convert(next_state).to(device)
    action = convert(action.to(device))
    reward = convert(reward.to(device))
    done = done.to(device)

    predicted = torch.gather(online_dqn(state), 0, torch.tensor(np.int64(action)).unsqueeze(-1)).squeeze(-1)
    if run_as_ddqn:
        action_from_online_network = online_dqn(next_state).max(1).indices
        expected = reward + gamma * torch.gather(target_dqn(next_state), 0, action_from_online_network.unsqueeze(-1)).squeeze(-1)
    else:
        expected = reward + gamma * target_dqn(next_state).max(1)[0]

    return criterion(expected, predicted)


def run_episode(curr_step, buffer, is_training, is_rendering=False):
    global eps
    episode_reward, episode_loss = 0, 0.
    state = env.reset()
    if is_rendering:
        env.render("rgb_array")

    accumulated_rewards = []
    for t in range(max_train_frames):
        action = policy(state, is_training)
        curr_step += 1

        next_state, reward, done, _ = env.step(action)
        if is_rendering:
            env.render("rgb_array")

        episode_reward += reward
        accumulated_rewards.append(episode_reward)

        if is_training:
            buffer.store(state, next_state, action, reward, done)

            if curr_step > burn_in_phase:
                state_batch, next_state_batch, action_batch, reward_batch, done_batch = buffer.sample(batch_size)

                if curr_step % sync_target == 0:
                    target_dqn.load_state_dict(online_dqn.state_dict())

                loss = compute_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
        else:
            with torch.no_grad():
                buffer.store(state, next_state, action, reward, done)
                state_batch, next_state_batch, action_batch, reward_batch, done_batch = buffer.sample(batch_size)

                episode_loss += compute_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch).item()

        state = next_state

        if done:
            break

    return dict(reward=episode_reward, loss=episode_loss / t), curr_step, accumulated_rewards


def update_metrics(metrics, episode):
    for k, v in episode.items():
        metrics[k].append(v)


def print_metrics(it, metrics, is_training, window=100):
    reward_mean = np.mean(metrics['reward'][-window:])
    loss_mean = np.mean(metrics['loss'][-window:])
    mode = "train" if is_training else "test"
    print(f"Episode {it:4d} | {mode:5s} | reward {reward_mean:5.5f} | loss {loss_mean:5.5f}")


def __train():
    global curr_step, eps
    train_metrics = dict(reward=[], loss=[])
    for it in range(max_train_episodes):
        episode_metrics, curr_step, _ = run_episode(curr_step, buffer, is_training=True)
        update_metrics(train_metrics, episode_metrics)
        if it % 10 == 0:
            print_metrics(it, train_metrics, is_training=True)
        eps *= eps_decay


def __test():
    global curr_step
    test_metrics = dict(reward=[], loss=[])
    for it in range(max_test_episodes):
        episode_metrics, curr_step, accumulated_rewards = run_episode(curr_step, buffer, is_training=False)
        update_metrics(test_metrics, episode_metrics)
        print_metrics(it + 1, test_metrics, is_training=False)
        filepath = path.join(getcwd(), f"output/test_plot_iteration_{it}")
        VisualizationHelper.plot_test_episode(filepath, np.arange(1, max_train_frames + 1), accumulated_rewards)

