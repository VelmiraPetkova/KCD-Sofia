from flask import Flask, Response
import datetime
import random

app = Flask(__name__)

NAMESPACE = "default"
CONTAINER = "app"

METRIC_PREFIX = "# HELP container_cpu_usage_seconds_total Cumulative cpu time consumed by the container in seconds\n# TYPE container_cpu_usage_seconds_total counter\n"

class Pod:
    def __init__(self, name, node, metric_modifier):
        self.name = name
        self.node = node
        self.metric_modifier = metric_modifier
        self.cpu_usage = 0
        self.last_report = datetime.datetime.now()

    def report(self):
        now = datetime.datetime.now()
        seconds = (now - self.last_report).total_seconds()
        self.last_report = now
        self.cpu_usage += seconds * self.metric_modifier()
        labels = f'namespace="{NAMESPACE}", pod="{self.name}", container="{CONTAINER}", kubernetes_io_hostname="{self.node}"'
        return f'{METRIC_PREFIX}container_cpu_usage_seconds_total{{{labels}}} {self.cpu_usage}'

def daily_difference():
    now = datetime.datetime.now()
    if now.weekday() < 5 and 8 <= now.hour <= 18:
        return random.uniform(0.0, 0.2) + 1.1
    return random.uniform(0.0, 0.2) + 0.8

def high_impedance():
    return random.random() + 1.0

def low_usage():
    return random.uniform(0.0, 0.1) + 0.2

# Configuration
PODS = [
    Pod("frontend-abc123", "mira-kubeflow2-worker3", high_impedance),
    Pod("backend-def456", "mira-kubeflow2-worker4", daily_difference),
    Pod("db-ghi789", "mira-kubeflow2-worker5", low_usage),
]

@app.route("/metrics")
def metrics():
    lines = []

    for pod in PODS:
        lines.append(pod.report())

    output = "\n".join(lines) + "\n"
    return Response(output, mimetype="text/plain; version=0.0.4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
