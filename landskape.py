from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, component

# Step 1: Load Dataset
@dsl.component(base_image="python:3.10")
def load_data_from_prometheus(output_csv: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "requests", "pandas"], check=True)

    import requests
    import pandas as pd
    from datetime import datetime, timedelta

    prometheus_url = "http://prometheus-server.monitoring.svc/api/v1/query_range"
    query = 'rate(container_cpu_usage_seconds_total{container!="",pod!="",namespace!="",kubernetes_io_hostname!=""}[5m])'

    num_days = 7
    all_rows = []

    for day_offset in range(num_days):
        end_dt = datetime.utcnow() - timedelta(days=day_offset)
        start_dt = end_dt - timedelta(days=1)
        start_time = int(start_dt.timestamp())
        end_time = int(end_dt.timestamp())
        params = {"query": query, "start": start_time, "end": end_time, "step": "60s"}
        response = requests.get(prometheus_url, params=params)
        result = response.json()["data"]["result"]

        for series in result:
            metric = series["metric"]
            for ts, val in series["values"]:
                all_rows.append({
                    "timestamp": int(float(ts)),
                    "cpu_usage": float(val),
                    "namespace": metric.get("namespace", ""),
                    "pod": metric.get("pod", ""),
                    "container": metric.get("container", ""),
                    "node": metric.get("kubernetes_io_hostname", "")
                })

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv.path, index=False)
    print(f"Loaded {len(df)} rows from the past {num_days} days.")


# Step 2: Preprocess Data with features
@dsl.component(base_image="python:3.10")
def preprocess_data(input_csv: Input[Dataset], output_csv: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas"], check=True)

    import pandas as pd

    df = pd.read_csv(input_csv.path)

    # Multiply raw cpu_usage column before grouping
    df['cpu_usage'] = df['cpu_usage'] * 10

    # Only keep specific nodes
    nodes = ['mira-kubeflow2-worker3', 'mira-kubeflow2-worker4', 'mira-kubeflow2-worker5']
    df = df[df['node'].isin(nodes)].copy()

    # Map node names to nicer labels
    node_mapping = {
        "mira-kubeflow2-worker3": "Frontend UI",
        "mira-kubeflow2-worker4": "Backend Service",
        "mira-kubeflow2-worker5": "Database"
    }
    df['node'] = df['node'].map(node_mapping).fillna(df['node'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df = (
        df.groupby([pd.Grouper(key='timestamp', freq='30min'), 'node'])
          .agg({'cpu_usage': ['mean', 'min', 'max']})
          .reset_index()
    )
    df.columns = ['timestamp', 'node', 'cpu_usage_mean', 'cpu_usage_min', 'cpu_usage_max']

    df['weekday'] = df['timestamp'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day

#creates lag features — columns with values from 3 intervals back
# These features say: "What was the CPU a moment ago?" - this helps the model detect trends.
    for lag in range(1, 4):
        df[f'cpu_mean_lag{lag}'] = df.groupby('node')['cpu_usage_mean'].shift(lag)
        df[f'cpu_min_lag{lag}'] = df.groupby('node')['cpu_usage_min'].shift(lag)
        df[f'cpu_max_lag{lag}'] = df.groupby('node')['cpu_usage_max'].shift(lag)

#These are rolling features:
    # .shift(1) means: shift back, do not include the current value (so as not to "steal" the future).
    # average CPU for the last 2 intervals (excluding the current one)
    df['cpu_mean_roll2'] = df.groupby('node')['cpu_usage_mean'].transform(lambda x: x.shift(1).rolling(2).mean())
    # cpu_mean_roll3 = average CPU for the last 3 intervals
    df['cpu_mean_roll3'] = df.groupby('node')['cpu_usage_mean'].transform(lambda x: x.shift(1).rolling(3).mean())
    df['cpu_max_roll2'] = df.groupby('node')['cpu_usage_max'].transform(lambda x: x.shift(1).rolling(2).max())
    df['cpu_max_roll3'] = df.groupby('node')['cpu_usage_max'].transform(lambda x: x.shift(1).rolling(3).max())

    # remove all rows that contain NaN/missing values
    df = df.dropna().reset_index(drop=True)
    df.to_csv(output_csv.path, index=False)
    print("Preprocessing complete. Final rows:", len(df))


# Step 3: Train Model
@dsl.component(base_image="python:3.10")
def train_model(input_csv: Input[Dataset], output_model: Output[Dataset], output_pdf: Output[Dataset]):
    import subprocess
    subprocess.run([
        "pip", "install", "pandas", "numpy", "xgboost==1.7.6",
        "scikit-learn", "joblib", "matplotlib", "graphviz"
    ], check=True)
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "graphviz"], check=True)

    import pandas as pd
    import xgboost as xgb
    import joblib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from xgboost import plot_tree, plot_importance


    FEATURE_COLUMNS = [
        'cpu_mean_lag1','cpu_mean_lag2','cpu_mean_lag3',
        'cpu_min_lag1','cpu_min_lag2','cpu_min_lag3',
        'cpu_max_lag1','cpu_max_lag2','cpu_max_lag3',
        'cpu_mean_roll2','cpu_mean_roll3','cpu_max_roll2','cpu_max_roll3',
        'weekday','is_weekend','hour','month','day'
    ]

    df = pd.read_csv(input_csv.path)
    models = {}
    with PdfPages(output_pdf.path) as pdf:
        for node_name, node_df in df.groupby('node'):
            if len(node_df) < 10:
                continue

            test_size = min(4, len(node_df)//4)
            train_df = node_df.iloc[:-test_size]
            test_df = node_df.iloc[-test_size:]

            X_train = train_df[FEATURE_COLUMNS]
            y_train = train_df['cpu_usage_mean']
            X_test = test_df[FEATURE_COLUMNS]
            y_test = test_df['cpu_usage_mean']

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )

            models[node_name] = model

            plt.figure(figsize=(20, 10))
            plot_tree(model, num_trees=0)
            plt.title(f"XGBoost Tree for node {node_name}")
            pdf.savefig()
            plt.close()

    joblib.dump(models, output_model.path)
    print("Saved models for nodes:", list(models.keys()))


# Step 4: Evaluate Model (1-day multi-step prediction + RMSE and plot)
@dsl.component(base_image="python:3.10")
def evaluate_model(input_csv: Input[Dataset], input_model: Input[Dataset], output_report: Output[Dataset], output_pdf: Output[Dataset]):
    import subprocess
    subprocess.run([
        "pip", "install", "pandas", "numpy", "scikit-learn",
        "matplotlib", "seaborn", "joblib", "xgboost==1.7.6"
    ], check=True)

    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    FEATURE_COLUMNS = [
        'cpu_mean_lag1','cpu_mean_lag2','cpu_mean_lag3',
        'cpu_min_lag1','cpu_min_lag2','cpu_min_lag3',
        'cpu_max_lag1','cpu_max_lag2','cpu_max_lag3',
        'cpu_mean_roll2','cpu_mean_roll3','cpu_max_roll2','cpu_max_roll3',
        'weekday','is_weekend','hour','month','day'
    ]

    df = pd.read_csv(input_csv.path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    models = joblib.load(input_model.path)

    report_lines = []
    future_steps = 48  # 48 * 30 min = 24 hours

    with PdfPages(output_pdf.path) as pdf:
        for node_name, node_df in df.groupby('node'):
            if node_name not in models or len(node_df) < 6:
                continue
            model = models[node_name]

            # Calculate RMSE on last 20 known points
            test_df = node_df.tail(20)
            X_test = test_df[FEATURE_COLUMNS]
            y_test = test_df['cpu_usage_mean']
            y_pred_test = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            report_lines.append(f"{node_name} RMSE (last 20 points): {rmse:.4f}")

            # --- Multi-step prediction for 1 day ---
            last_known = node_df.tail(3).copy().reset_index(drop=True)
            predictions = []

            for step in range(future_steps):
                last_ts = last_known['timestamp'].iloc[-1]
                next_ts = last_ts + pd.Timedelta(minutes=30)

                X_input = pd.DataFrame({
                    'cpu_mean_lag1': [last_known['cpu_usage_mean'].iloc[-1]],
                    'cpu_mean_lag2': [last_known['cpu_usage_mean'].iloc[-2]],
                    'cpu_mean_lag3': [last_known['cpu_usage_mean'].iloc[-3]],
                    'cpu_min_lag1': [last_known['cpu_usage_min'].iloc[-1]],
                    'cpu_min_lag2': [last_known['cpu_usage_min'].iloc[-2]],
                    'cpu_min_lag3': [last_known['cpu_usage_min'].iloc[-3]],
                    'cpu_max_lag1': [last_known['cpu_usage_max'].iloc[-1]],
                    'cpu_max_lag2': [last_known['cpu_usage_max'].iloc[-2]],
                    'cpu_max_lag3': [last_known['cpu_usage_max'].iloc[-3]],
                    'cpu_mean_roll2': [last_known['cpu_usage_mean'].iloc[-2:].mean()],
                    'cpu_mean_roll3': [last_known['cpu_usage_mean'].iloc[-3:].mean()],
                    'cpu_max_roll2': [last_known['cpu_usage_max'].iloc[-2:].max()],
                    'cpu_max_roll3': [last_known['cpu_usage_max'].iloc[-3:].max()],
                    'weekday': [next_ts.weekday()],
                    'is_weekend': [int(next_ts.weekday() >= 5)],
                    'hour': [next_ts.hour],
                    'month': [next_ts.month],
                    'day': [next_ts.day]
                })

                y_pred = model.predict(X_input)[0]
                predictions.append(y_pred)

                new_row = last_known.iloc[-1].copy()
                new_row['cpu_usage_mean'] = y_pred
                new_row['timestamp'] = next_ts
                last_known = pd.concat([last_known, new_row.to_frame().T], ignore_index=True)

            # --- Combined Historical + Prediction plot ---
            plt.figure(figsize=(14,6))

            # Historical data
            plt.plot(node_df['timestamp'], node_df['cpu_usage_mean'],
                     label='Historical CPU usage (mean)', color='blue')

            # Forecast
            x_axis = pd.date_range(
                start=last_known['timestamp'].iloc[2],
                periods=future_steps+1, freq='30min'
            )[1:]
            plt.plot(x_axis, predictions,
                     label='Predicted CPU usage (mean)', color='orange', marker='o')

            # Last known values
            plt.plot(last_known['timestamp'].iloc[:3],
                     last_known['cpu_usage_mean'].iloc[:3],
                     label='Last known CPU values', color='green', marker='x')

            plt.xlabel('Timestamp (UTC)')
            plt.ylabel('CPU usage (seconds)')
            plt.title(f"Historical + 1-day Forecast for Node: {node_name}\nRMSE (last 20 points): {rmse:.4f}")
            plt.legend()
            plt.grid(True)
            pdf.savefig()
            plt.close()

    with open(output_report.path, "w") as f:
        f.write("\n".join(report_lines))

    print("Evaluation done with combined historical + forecast plots, 1-day multi-step prediction, and RMSE.")



@dsl.component(base_image="python:3.10")
def classify_cpu_usage(input_csv: Input[Dataset], output_pdf: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "matplotlib", "seaborn"], check=True)

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages

    # Declared CPU per human-readable node label (already mapped in preprocess component)
    declared_cpu_map = {
        'Frontend UI': 2,
        'Backend Service': 2.5,
        'Database': 2.5   # change later if needed
    }

    df = pd.read_csv(input_csv.path)

    # at this point df['node'] already contains labels like "Frontend UI"
    df['declared_cpu'] = df['node'].map(declared_cpu_map)

    # Convert timestamp column
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Compute ratio
    df['cpu_ratio'] = df['cpu_usage_mean'] / df['declared_cpu']

    # Classification
    def classify(r):
        if r < 0.1:
            return "Low usage"
        elif r <= 0.5:
            return "Optimal"
        else:
            return "High usage"

    df['usage_class'] = df['cpu_ratio'].apply(classify)

    # Recommendations
    recommendations = []
    for node, group in df.groupby('node'):
        avg_ratio = group['cpu_ratio'].mean()
        if avg_ratio < 0.1:
            recommendations.append(f"{node}: Low usage → consider reducing allocated resources")
        elif avg_ratio > 0.5:
            recommendations.append(f"{node}: High usage → consider increasing CPU or scaling pods")
        else:
            recommendations.append(f"{node}: Optimal resource utilization")

    # Create PDF
    with PdfPages(output_pdf.path) as pdf:
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        plt.title("CPU Usage Recommendations", fontsize=16)
        plt.text(0, 0.9, "\n".join(recommendations), fontsize=12)
        pdf.savefig()
        plt.close()

        # Plot per node
        for node, group in df.groupby('node'):
            plt.figure(figsize=(12, 5))
            sns.lineplot(
                data=group,
                x='timestamp',
                y='cpu_ratio',
                hue='usage_class',
                palette={'Low usage': 'blue', 'Optimal': 'green', 'High usage': 'red'}
            )
            plt.axhline(0.7, color='yellow', linestyle='--', label='Optimal threshold (0.7)')
            plt.axhline(1.0, color='red', linestyle='--', label='Max threshold (1.0)')
            plt.title(f"CPU Utilization Ratio Over Time for Node: {node}")
            plt.ylabel("CPU Utilization Ratio (used / declared)")
            plt.xlabel("Timestamp (UTC)")
            plt.legend()
            plt.grid(True)
            pdf.savefig()
            plt.close()

    print("CPU usage classification and recommendations PDF generated.")



@dsl.component(base_image="python:3.10")
def what_if_simulator(input_csv: Input[Dataset], input_model: Input[Dataset], output_pdf: Output[Dataset]):
    import subprocess
    subprocess.run([
        "pip", "install", "pandas", "numpy",
        "scikit-learn",  # latest
        "xgboost==2.0.3",  # съвместим с sklearn>=1.5
        "joblib", "matplotlib", "graphviz"
    ], check=True)

    import pandas as pd
    import numpy as np
    import joblib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    FEATURE_COLUMNS = [
        'cpu_mean_lag1','cpu_mean_lag2','cpu_mean_lag3',
        'cpu_min_lag1','cpu_min_lag2','cpu_min_lag3',
        'cpu_max_lag1','cpu_max_lag2','cpu_max_lag3',
        'cpu_mean_roll2','cpu_mean_roll3','cpu_max_roll2','cpu_max_roll3',
        'weekday','is_weekend','hour','month','day'
    ]

    df = pd.read_csv(input_csv.path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    models = joblib.load(input_model.path)

    scenarios = {
        "baseline": 1.0,
        "scale_up": 1.5,   # +50% CPU capacity
        "scale_down": 0.7 # -30% CPU capacity
    }

    with PdfPages(output_pdf.path) as pdf:
        for node_name, node_df in df.groupby('node'):
            if node_name not in models or len(node_df) < 6:
                continue

            model = models[node_name]

            last_known = node_df.tail(3).copy().reset_index(drop=True)
            future_steps = 24  # 12h = 24 * 30min
            predictions_per_scenario = {}

            for scenario, factor in scenarios.items():
                last_sim = last_known.copy()
                predictions = []

                for step in range(future_steps):
                    last_ts = last_sim['timestamp'].iloc[-1]
                    next_ts = last_ts + pd.Timedelta(minutes=30)

                    X_input = pd.DataFrame({
                        'cpu_mean_lag1': [last_sim['cpu_usage_mean'].iloc[-1]],
                        'cpu_mean_lag2': [last_sim['cpu_usage_mean'].iloc[-2]],
                        'cpu_mean_lag3': [last_sim['cpu_usage_mean'].iloc[-3]],
                        'cpu_min_lag1': [last_sim['cpu_usage_min'].iloc[-1]],
                        'cpu_min_lag2': [last_sim['cpu_usage_min'].iloc[-2]],
                        'cpu_min_lag3': [last_sim['cpu_usage_min'].iloc[-3]],
                        'cpu_max_lag1': [last_sim['cpu_usage_max'].iloc[-1]],
                        'cpu_max_lag2': [last_sim['cpu_usage_max'].iloc[-2]],
                        'cpu_max_lag3': [last_sim['cpu_usage_max'].iloc[-3]],
                        'cpu_mean_roll2': [last_sim['cpu_usage_mean'].iloc[-2:].mean()],
                        'cpu_mean_roll3': [last_sim['cpu_usage_mean'].iloc[-3:].mean()],
                        'cpu_max_roll2': [last_sim['cpu_usage_max'].iloc[-2:].max()],
                        'cpu_max_roll3': [last_sim['cpu_usage_max'].iloc[-3:].max()],
                        'weekday': [next_ts.weekday()],
                        'is_weekend': [int(next_ts.weekday() >= 5)],
                        'hour': [next_ts.hour],
                        'month': [next_ts.month],
                        'day': [next_ts.day]
                    })

                    y_pred = model.predict(X_input)[0] * factor
                    predictions.append(y_pred)

                    new_row = last_sim.iloc[-1].copy()
                    new_row['cpu_usage_mean'] = y_pred
                    new_row['timestamp'] = next_ts
                    last_sim = pd.concat([last_sim, new_row.to_frame().T], ignore_index=True)

                predictions_per_scenario[scenario] = (last_sim['timestamp'].iloc[3:], predictions)

            # --- Plot all scenarios ---
            plt.figure(figsize=(14,6))
            plt.plot(node_df['timestamp'], node_df['cpu_usage_mean'], label="Historical", color="blue")

            for scenario, (x_axis, preds) in predictions_per_scenario.items():
                plt.plot(x_axis, preds, marker="o", label=f"Predicted ({scenario})")

            plt.title(f"What-if Simulator for {node_name}")
            plt.xlabel("Timestamp (UTC)")
            plt.ylabel("CPU usage (seconds)")
            plt.legend()
            plt.grid(True)
            pdf.savefig()
            plt.close()

    print("What-if simulation completed and PDF saved.")




@dsl.pipeline(name="kcd_Bulgaria")
def kcd_Bulgaria():
    load = load_data_from_prometheus()
    preprocess = preprocess_data(input_csv=load.outputs["output_csv"])

    # Parallel steps
    train = train_model(input_csv=preprocess.outputs["output_csv"])
    classify = classify_cpu_usage(input_csv=preprocess.outputs["output_csv"])

    # evaluate depends on train
    evaluate = evaluate_model(
        input_csv=preprocess.outputs["output_csv"],
        input_model=train.outputs["output_model"]
    )

    what_if = what_if_simulator(
        input_csv=preprocess.outputs["output_csv"],
        input_model=train.outputs["output_model"]
    )



if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func= kcd_Bulgaria,
        package_path="KCD_Pipeline_Demo.yaml"
    )


