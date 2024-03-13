import numpy as np
import time, datetime
import matplotlib.pyplot as plt
from agent import AgentParams
import os

# Using this logger class from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

class MetricLogger:
    def __init__(self, save_dir:str, window_size:int):
        self.save_dir = save_dir
        self.log_file = 'log.csv'


        # self.ep_rewards_plot = save_dir + "/reward_plot.jpg"
        # self.ep_lengths_plot = save_dir + "/length_plot.jpg"
        # self.ep_avg_losses_plot = save_dir + "/loss_plot.jpg"
        # self.ep_avg_qs_plot = save_dir + "/q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Window size for moving average
        self.window_size = window_size

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward:float, loss:float, q:float) -> None:
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self) -> None:
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self) -> None:
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode:int, epsilon:float, step:int, save_dir:str=None, window:int=None) -> None:
        if save_dir == None:
            save_dir = self.save_dir
        if window == None:
            window = self.window_size
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-window:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-window:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-window:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-window:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        # print(
        #     '\n'
        #     f"Episode {episode} - "
        #     f"Step {step} - "
        #     f"Epsilon {epsilon} - "
        #     f"Mean Reward {mean_ep_reward} - "
        #     f"Mean Length {mean_ep_length} - "
        #     f"Mean Loss {mean_ep_loss} - "
        #     f"Mean Q Value {mean_ep_q} - "
        #     f"Time Delta {time_since_last_record} - "
        #     f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        # )
        os.makedirs(save_dir)
        with open(f'{self.save_dir}/{self.log_file}', "a") as f:
            f.write(
                '#Episode,Step,Epsilon,MeanReward,MeanLength,MeanLoss,MeanQValue,TimeDelta,Time\n'
                f"{episode},{step},{epsilon},"
                f"{mean_ep_reward},{mean_ep_length},{mean_ep_loss},{mean_ep_q},"
                f"{time_since_last_record},"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n"
            )

        metrics = {
            'ep_lengths' : self.moving_avg_ep_lengths,
            'ep_avg_losses' : self.moving_avg_ep_avg_losses,
            'ep_avg_qs' : self.moving_avg_ep_avg_qs,
            'ep_rewards' : self.moving_avg_ep_rewards}
        for metric_name, metric in metrics.items():
            plt.clf()
            plt.plot(metric, label=metric_name)
            plt.xlabel('Episode')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name.capitalize()} over Episodes')
            plt.legend()
            plt.savefig(f'{self.save_dir}/{metric_name}_plot.png')

    def record_hyperparams(self, hyperparams:AgentParams):
        del hyperparams['state_shape']
        del hyperparams['num_actions']
        hyperparams_str = '\n'.join([f"{key}: {value}" for key, value in hyperparams.items()])
        print("\nHyperparameters:")
        print(hyperparams_str)
        with open(self.log_file, "a") as f:
            f.write(
                f"\nHyperparameters:\n{hyperparams_str}\n"
            )
