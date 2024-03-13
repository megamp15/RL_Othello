import os
import pandas as pd
import matplotlib.pyplot as plt

class LogFileAnalyzer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_rewards = []

    def process_log_files(self):
        log_files = os.listdir(self.log_dir)
        for file in log_files:
            if file.endswith('_LOG.csv'):
                label = file.split('_LOG.csv')[0]
                file_path = os.path.join(self.log_dir, file)
                self.process_log_file(file_path, label)

    def process_log_file(self, file_path, label):
        df = pd.read_csv(file_path)
        self.ep_lengths.append((df['#Episode'], label))
        self.ep_avg_losses.append((df['MeanLoss'], label))
        self.ep_avg_qs.append((df['MeanQValue'], label))
        self.ep_rewards.append((df['MeanReward'], label))

    def generate_plots(self):
        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            for data, label in getattr(self, metric):
                plt.plot(data, label=label)
            plt.xlabel('Episode')
            plt.ylabel(metric.split('_')[-1].capitalize())
            plt.title(f'{metric.capitalize()} over Episodes')
            plt.legend()
            plt.savefig(f"{self.log_dir}/{metric}_plot.png")

if __name__ == "__main__":
    log_dir = "./logs_to_plot"
    analyzer = LogFileAnalyzer(log_dir)
    analyzer.process_log_files()
    analyzer.generate_plots()
