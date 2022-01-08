from collections import deque, OrderedDict
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
import random
from gym.spaces import Box
from collections import deque
import copy
from gym.wrappers import FrameStack


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transform = torchvision.transforms.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(self.shape),
                                                     torchvision.transforms.Normalize(0, 255)])
        return transforms(observation).squeeze(0)


class ExperienceReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def store(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = zip(*[self.memory[idx] for idx in indices])

        return torch.tensor(np.asarray(state_batch)), torch.tensor(np.asarray(next_state_batch)), torch.tensor(np.asarray(action_batch)), torch.tensor(np.asarray(reward_batch)), \
               torch.tensor(np.asarray(done_batch))


class DeepQNet(nn.Module):
    def __init__(self, h, w, image_stack, num_actions):
        super(DeepQNet, self).__init__()

        num_output_filters = 64

        self.conv_net = nn.Sequential(OrderedDict([
            ('conv2d_1', nn.Conv2d(image_stack, 32, kernel_size=(3, 3), stride=(1, 1), padding='same', padding_mode='zeros')),
            ('relu_1', nn.ReLU()),
            ('max2d_pooling_1', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('conv2d_2', nn.Conv2d(32, num_output_filters, kernel_size=(3, 3), stride=(1, 1), padding='same', padding_mode='zeros')),
            ('relu_2', nn.ReLU()),
            ('max2d_pooling_2', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('flatten', nn.Flatten())
        ]))

        num_hidden = 128

        self.dense_net = nn.Sequential(OrderedDict([
            ('dense_1', nn.Linear(int(num_output_filters * (h / 4) * (w / 4)), num_hidden)),
            ('dense_2', nn.Linear(num_hidden, num_actions))
        ]))

    def forward(self, x):
        return self.dense_net(self.conv_net(x))


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
max_train_episodes = 1000000
max_test_episodes = 10
max_train_frames = 10000
burn_in_phase = 32  # TODO: change back to original value 50000
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

optimizer = optim.Adam(online_dqn.parameters(), lr=1e-4)
criterion = nn.MSELoss()


def convert(x):
    return torch.tensor(x.__array__()).float()


def policy(state, is_training):
    global eps
    state = convert(state).unsqueeze(0).to(device)

    if is_training and np.random.uniform(0, 1) < eps:  # choose actions deterministically during test phase
        return np.random.choice(num_actions)  # return a random action with probability epsilon
    else:
        return np.argmax(online_dqn[state, :])  # otherwise return the action that maximizes Q


def compute_loss(state, action, reward, next_state, done):
    state = convert(state).to(device)
    next_state = convert(next_state).to(device)
    action = action.to(device)
    reward = reward.to(device)
    done = done.to(device)

    predicted = torch.gather(online_dqn(state), 0, torch.tensor(np.int64(action)).unsqueeze(-1)).squeeze(-1)
    expected = reward + gamma * target_dqn(next_state).max(1)[0]

    return criterion(expected, predicted)


def run_episode(curr_step, buffer, is_training, is_rendering=False):
    global eps
    episode_reward, episode_loss = 0, 0.
    state = env.reset()
    if is_rendering:
        env.render("rgb_array")

    for t in range(max_train_frames):
        action = policy(state, is_training)
        curr_step += 1

        next_state, reward, done, _ = env.step(action)
        if is_rendering:
            env.render("rgb_array")

        episode_reward += reward

        if is_training:
            buffer.store(state, next_state, action, reward, done)

            if curr_step > burn_in_phase:
                state_batch, next_state_batch, action_batch, reward_batch, done_batch = buffer.sample(batch_size)

                # if curr_step % sync_target == 0:
                #     # TODO: Periodically update your target_dqn at each sync_target frames
                #     # ...
                #     pass

                loss = compute_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
        else:
            with torch.no_grad():
                episode_loss += compute_loss(state, action, reward, next_state, done).item()

        state = next_state

        if done:
            break

    return dict(reward=episode_reward, loss=episode_loss / t), curr_step


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
        episode_metrics, curr_step = run_episode(curr_step, buffer, is_training=True)
        update_metrics(train_metrics, episode_metrics)
        if it % 10 == 0:
            print_metrics(it, train_metrics, is_training=True)
        eps *= eps_decay


def __test():
    global curr_step
    test_metrics = dict(reward=[], loss=[])
    for it in range(max_test_episodes):
        episode_metrics, curr_step = run_episode(curr_step, buffer, is_training=False)
        update_metrics(test_metrics, episode_metrics)
        print_metrics(it + 1, test_metrics, is_training=False)


if __name__ == '__main__':
    # Test DQN
    # cnn = DeepQNet((84, 84), image_stack, num_actions)
    # cnn.forward(np.empty((4, 210, 160, 3)))

    # Test exp replay memory
    # exp_replay = ExperienceReplayMemory(50000)
    # exp_replay.store(2, 3, 1, 3.4, 1)
    # exp_replay.store(4, 3, 2, 3.8, 0)
    # samples = exp_replay.sample(2)

    __train()
    # __test()

    print('x')


