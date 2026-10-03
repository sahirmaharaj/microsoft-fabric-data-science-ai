import os
import sys
import subprocess
import importlib.util
import importlib.metadata

DEPENDENCIES = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<3', 'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2', 'matplotlib': 'matplotlib>=3.8,<4', 'joblib': 'joblib>=1.3,<2', 'threadpoolctl': 'threadpoolctl>=3.2,<4'}
if importlib.util.find_spec("packaging") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging>=23,<27"])
from packaging.requirements import Requirement

missing_packages = []
for module, requirement_text in DEPENDENCIES.items():
    requirement = Requirement(requirement_text)
    try:
        installed_version = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    if importlib.util.find_spec(module) is None or installed_version is None or installed_version not in requirement.specifier:
        missing_packages.append(requirement_text)
if missing_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing_packages])
for thread_variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(thread_variable, "2")


SEED = 42
SAMPLE_SIZE = 1200
OUTPUT_ROOT = os.environ.get("FABRIC_STARTER_OUTPUT", "")
WORKFLOW = "24_probabilistic_demand_forecasting"


import json
import hashlib
import time
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from threadpoolctl import threadpool_limits

thread_limits = threadpool_limits(limits=2)
rng = np.random.default_rng(SEED)
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
lakehouse_files = Path("/lakehouse/default/Files")
output_base = Path(OUTPUT_ROOT).expanduser() if OUTPUT_ROOT else (lakehouse_files / "fabric_python_starters" if lakehouse_files.is_dir() else Path.cwd() / "fabric_python_outputs")
output_dir = output_base / WORKFLOW / run_id
output_dir.mkdir(parents=True, exist_ok=True)
working_context = tempfile.TemporaryDirectory(prefix="fabric_starter_")
working_dir = Path(working_context.name)
run_started = time.perf_counter()
artifact_records = []

def json_value(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    raise TypeError(type(value).__name__)

def record_artifact(path):
    path = Path(path)
    artifact_records.append({"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path

def save_table(frame, name):
    path = output_dir / (name + ".csv")
    frame.to_csv(path, index=False)
    record_artifact(path)
    return path

def save_json(payload, name):
    path = output_dir / (name + ".json")
    path.write_text(json.dumps(payload, indent=2, default=json_value, allow_nan=False), encoding="utf-8")
    record_artifact(path)
    return path

def save_model(model, name):
    path = output_dir / (name + ".joblib")
    joblib.dump(model, path)
    record_artifact(path)
    return path

def save_figure(figure, name):
    figure.tight_layout()
    path = output_dir / (name + ".png")
    figure.savefig(path, dpi=130, bbox_inches="tight")
    record_artifact(path)
    plt.show()
    plt.close(figure)
    return path

def finish(metrics, tables=()):
    versions = {}
    for requirement in DEPENDENCIES.values():
        distribution = requirement.split(">=")[0]
        versions[distribution] = importlib.metadata.version(distribution)
    payload = {"workflow": WORKFLOW, "run_id": run_id, "seed": SEED, "python": sys.version.split()[0], "packages": versions, "storage": "lakehouse" if lakehouse_files in output_dir.parents else "local", "output_dir": str(output_dir), "elapsed_seconds": round(time.perf_counter() - run_started, 3), "metrics": metrics, "artifacts": list(artifact_records)}
    save_json(payload, "run_manifest")
    print(json.dumps(payload, indent=2, default=json_value, allow_nan=False))
    for table in tables:
        print(table.head(12).to_string(index=False))
    working_context.cleanup()
    return payload


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

periods = 560
dates = pd.date_range("2024-01-01", periods=periods, freq="D")
t = np.arange(periods)
y = 100 + 0.08 * t + 12 * np.sin(2 * np.pi * t / 7) + 5 * np.cos(2 * np.pi * t / 30) + rng.normal(0, 4, periods)
series = pd.DataFrame({"date": dates, "demand": y})

def feature_frame(frame):
    output = frame.copy()
    for lag in [1, 2, 7, 14, 28]:
        output[f"lag_{lag}"] = output.demand.shift(lag)
    for window in [7, 14, 28]:
        output[f"rolling_mean_{window}"] = output.demand.shift(1).rolling(window).mean()
        output[f"rolling_std_{window}"] = output.demand.shift(1).rolling(window).std()
    output["weekday_sin"] = np.sin(2 * np.pi * output.date.dt.dayofweek / 7)
    output["weekday_cos"] = np.cos(2 * np.pi * output.date.dt.dayofweek / 7)
    output["trend"] = (output.date - dates[0]).dt.days
    return output

features = feature_frame(series).dropna().reset_index(drop=True)
columns = [column for column in features if column not in ["date", "demand"]]

from sklearn.metrics import mean_pinball_loss

train = features.iloc[:-100]
calibration = features.iloc[-100:-50]
test = features.iloc[-50:]
quantiles = [0.05, 0.5, 0.95]
models = {}
raw = {}
cal = {}
for quantile in quantiles:
    model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=120, max_leaf_nodes=12, l2_regularization=2, random_state=SEED)
    model.fit(train[columns], train.demand)
    models[quantile] = model
    raw[quantile] = model.predict(test[columns])
    cal[quantile] = model.predict(calibration[columns])


cal_lower = np.minimum(cal[0.05], cal[0.95])
cal_upper = np.maximum(cal[0.05], cal[0.95])
nonconformity = np.maximum(cal_lower - calibration.demand.to_numpy(), calibration.demand.to_numpy() - cal_upper)
q = min(1, np.ceil((len(calibration) + 1) * 0.9) / len(calibration))
adjustment = max(0.0, float(np.quantile(nonconformity, q, method="higher")))
ordered = np.sort(np.column_stack([raw[value] for value in quantiles]), axis=1)
output = pd.DataFrame({"date": test.date, "actual": test.demand, "lower": ordered[:, 0] - adjustment, "median": ordered[:, 1], "upper": ordered[:, 2] + adjustment})
output["covered"] = output.actual.between(output.lower, output.upper)
output["interval_width"] = output.upper-output.lower
output["weekday"] = output.date.dt.day_name()
by_weekday = output.groupby("weekday").agg(coverage=("covered", "mean"), width=("interval_width", "mean"), rows=("actual", "size")).reset_index()
losses = pd.DataFrame([{ "quantile": quantile, "pinball_loss": mean_pinball_loss(test.demand, raw[quantile], alpha=quantile)} for quantile in quantiles])
lead_time = 5
simulations = []
residuals = calibration.demand.to_numpy() - cal[0.5]
for simulation in range(500):
    start = rng.integers(0, len(residuals)-lead_time+1)
    simulated = np.maximum(0, ordered[:lead_time, 1] + residuals[start:start+lead_time])
    simulations.append({"simulation": simulation, "lead_time_demand": simulated.sum()})
simulations = pd.DataFrame(simulations)
save_table(output, "one_step_forecast_intervals")
save_table(by_weekday, "weekday_coverage")
save_table(losses, "quantile_losses")
save_table(simulations, "lead_time_simulations")
save_model({"models": models, "adjustment": adjustment, "columns": columns}, "probabilistic_forecaster")
assert output.lower.le(output.upper).all()
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(output.date, output.lower, output.upper, alpha=0.25, label="90% nominal interval")
ax.plot(output.date, output.actual, label="Actual")
ax.plot(output.date, output["median"], label="Median")
ax.legend()
save_figure(fig, "probabilistic_forecast")
result = finish({"one_step_coverage": float(output.covered.mean()), "mean_width": float(output.interval_width.mean()), "median_mae": mean_absolute_error(output.actual, output["median"]), "lead_time_p95": float(simulations.lead_time_demand.quantile(0.95))}, [losses, by_weekday])
