import pandas as pd
import json

baseline = pd.read_csv("results/baseline_metrics.csv").iloc[-1]
pcgrad = pd.read_csv("results/pcgrad_metrics.csv").iloc[-1]

data = {
    "baseline": {
        "task_a": {"loss": float(baseline["loss_a"])},
        "task_b": {"loss": float(baseline["loss_b"])}
    },
    "pcgrad": {
        "task_a": {"loss": float(pcgrad["loss_a"])},
        "task_b": {"loss": float(pcgrad["loss_b"])}
    }
}

with open("results/final_metrics.json", "w") as f:
    json.dump(data, f, indent=2)