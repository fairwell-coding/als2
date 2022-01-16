import matplotlib.pyplot as plt


class VisualizationHelper:
    """Helper class for visualization and plots of the training and classification process"""

    @staticmethod
    def plot_test_episode(filename, frame_number, accumulated_reward):
        plt.subplots()
        plt.xlabel("frame_number")
        plt.ylabel("accumulated_reward")
        plt.title("Test results")

        plt.plot(
            frame_number,
            accumulated_reward,
        )
        plt.savefig(filename)
        plt.show()
