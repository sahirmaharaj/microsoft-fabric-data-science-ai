import os
import sys
import subprocess
import importlib.util
import importlib.metadata

DEPENDENCIES = {'numpy': 'numpy>=1.26,<3', 'pandas': 'pandas>=2.1,<3', 'scipy': 'scipy>=1.11,<2', 'sklearn': 'scikit-learn>=1.4,<2', 'matplotlib': 'matplotlib>=3.8,<4', 'joblib': 'joblib>=1.3,<2', 'threadpoolctl': 'threadpoolctl>=3.2,<4', 'statsmodels': 'statsmodels>=0.14,<1'}
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
WORKFLOW = "26_seasonal_anomalies_and_change_detection"


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


from statsmodels.tsa.seasonal import STL
from scipy.stats import median_abs_deviation
from sklearn.metrics import precision_score, recall_score

n = 400
dates = pd.date_range("2025-01-01", periods=n, freq="D")
t = np.arange(n)
values = 100 + 0.04*t + 8*np.sin(2*np.pi*t/7) + rng.normal(0, 1.8, n)
spike_positions = np.array([220, 255, 301, 335, 370])
values[spike_positions] += [20, -22, 25, -20, 24]
values[320:] += 10
series = pd.Series(values, index=dates)
training = series.iloc[:200]
fit = STL(training, period=7, robust=True).fit()
seasonal_pattern = pd.Series(fit.seasonal.to_numpy()).groupby(np.arange(len(training)) % 7).mean().to_numpy()
training_residual = fit.resid.to_numpy()
scale = max(1.4826 * median_abs_deviation(training_residual), 1.0)
level = float((training.iloc[-14:].to_numpy() - seasonal_pattern[np.arange(186,200)%7]).mean())
positive_cusum, negative_cusum = 0.0, 0.0
rows = []
for index in range(200, n):
    expected = level + seasonal_pattern[index % 7]
    error = values[index]-expected
    score = error/scale
    point_anomaly = abs(score) > 4.5
    positive_cusum = max(0.0, positive_cusum + np.clip(score,-5,5) - 0.75)
    negative_cusum = min(0.0, negative_cusum + np.clip(score,-5,5) + 0.75)
    change = positive_cusum > 12 or negative_cusum < -12
    rows.append({"date": dates[index], "actual": values[index], "expected": expected, "z_score": score, "point_anomaly": point_anomaly, "positive_cusum": positive_cusum, "negative_cusum": negative_cusum, "change_alert": change, "true_spike": index in spike_positions})
    level += 0.08 * np.clip(error, -3*scale, 3*scale)
    if change:
        level += np.clip(error, -6*scale, 6*scale) * 0.5
        positive_cusum, negative_cusum = 0.0, 0.0
output = pd.DataFrame(rows)
output["lower"] = output.expected - 4.5*scale


output["upper"] = output.expected + 4.5*scale
alerts = output.loc[output.point_anomaly | output.change_alert].copy()
output["week"] = output.date.dt.to_period("W").astype(str)
weekly = output.groupby("week").agg(point_alerts=("point_anomaly","sum"), change_alerts=("change_alert","sum"), mean_score=("z_score","mean")).reset_index()
save_table(output, "stream_scores")
save_table(alerts, "alerts")
save_table(weekly, "weekly_monitoring")
save_json({"seasonal_pattern": seasonal_pattern, "robust_scale": scale, "latest_level": level, "period": 7}, "detector_state")
assert np.isfinite(output.z_score).all()
assert len(output) == n-200
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(output.date, output.actual, label="Actual")
axes[0].plot(output.date, output.expected, label="Expected")
axes[0].scatter(alerts.date, alerts.actual, color="red", s=25)
axes[0].legend()
axes[1].plot(output.date, output.positive_cusum)
axes[1].plot(output.date, output.negative_cusum)
save_figure(fig, "streaming_anomalies")
result = finish({"point_precision": precision_score(output.true_spike, output.point_anomaly, zero_division=0), "spike_recall": recall_score(output.true_spike, output.point_anomaly), "change_alerts": int(output.change_alert.sum())}, [alerts])
