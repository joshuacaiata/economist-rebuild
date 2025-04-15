import os
import matplotlib.pyplot as plt

class MetricsLogger:
    def __init__(self, loss_folder):
        self.loss_folder = loss_folder
        os.makedirs(loss_folder, exist_ok=True)
        self.metrics = {
            "reward": [], 
            "policy_loss": [], 
            "value_loss": [], 
            "total_loss": [], 
            "utility": []
        }
        self.agent_metrics = {}
    
    def log(self, metric_name, value):
        if metric_name.startswith("env") and "_agent" in metric_name:
            if metric_name not in self.agent_metrics:
                self.agent_metrics[metric_name] = []
            self.agent_metrics[metric_name].append(value)
        else:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)
    
    def plot_metrics(self):
        for metric_name, values in self.metrics.items():
            if values: 
                plt.figure(figsize=(10,6))
                plt.plot(values)
                plt.xlabel("Update Iteration")
                plt.ylabel(metric_name.capitalize())
                plt.title(f"{metric_name.capitalize()} over Updates")
                plt.grid(True)
                plot_path = os.path.join(self.loss_folder, f"{metric_name}.png")
                plt.savefig(plot_path)
                plt.close()

