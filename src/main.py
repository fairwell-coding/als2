from environment import __train, __test

if __name__ == "__main__":
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
