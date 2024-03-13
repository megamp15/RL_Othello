import os
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogFileAnalyzer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics : dict[str,dict[str,list[float]]] = {
            'Mean_Reward' : {},
            'Mean_Length' : {},
            'Mean_Loss' : {},
            'Mean_Q-Value' : {}
        }

    def process_log_files(self):
        log_files = os.listdir(self.log_dir)
        for file in log_files:
            if file.endswith('_LOG.csv'):
                label = file.split('_LOG.csv')[0]
                file_path = os.path.join(self.log_dir, file)
                self.process_log_file(file_path, label)

    def process_log_file(self, file_path, label):
        _, _, _, ep_rewards, ep_lengths, ep_avg_losses, ep_avg_qs, _, _ \
            = np.loadtxt(file_path, converters={8 : lambda x: None}, delimiter=',',unpack=True)
        self.metrics['Mean_Reward'][label] = ep_rewards
        self.metrics['Mean_Length'][label] = ep_lengths
        self.metrics['Mean_Loss'][label] = ep_avg_losses
        self.metrics['Mean_Q-Value'][label] = ep_avg_qs

    def generate_plots(self):
        for metric_name, metric_data in self.metrics.items():
            plt.clf()
            for model_name in metric_data.keys():
                plt.plot(metric_data[model_name], label=model_name)
            plt.xlabel('Episode')
            plt.ylabel(metric_name.split('_')[1].capitalize())
            plt.title(f'{metric_name.capitalize()} over Episodes')
            plt.legend()
            plt.savefig(f"{self.log_dir}/{metric_name}_plot.png")

if __name__ == "__main__":
    log_dir = "./logs_to_plot"
    analyzer = LogFileAnalyzer(log_dir)
    analyzer.process_log_files()
    analyzer.generate_plots()
