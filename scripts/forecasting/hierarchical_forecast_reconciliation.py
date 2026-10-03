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
WORKFLOW = "25_hierarchical_forecast_reconciliation"


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


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

periods, horizon = 180, 14
t = np.arange(periods+horizon)
bottom = np.column_stack([40 + 0.06*t + 5*np.sin(2*np.pi*t/7 + phase) + rng.normal(0, 2, len(t)) for phase in [0, 0.5, 1.0, 1.5]])
S = np.array([[1,1,1,1], [1,1,0,0], [0,0,1,1], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], dtype=float)
names = ["total", "north", "south", "north_a", "north_b", "south_a", "south_b"]
all_series = bottom @ S.T
calendar_features = np.column_stack([t/periods, np.sin(2*np.pi*t/7), np.cos(2*np.pi*t/7)])
base_forecasts = np.zeros((horizon, len(names)))
residuals = np.zeros((40, len(names)))
models = {}
for index, name in enumerate(names):
    model = Ridge(alpha=0.2 + index*0.15).fit(calendar_features[:periods-40], all_series[:periods-40, index])
    residuals[:, index] = all_series[periods-40:periods, index] - model.predict(calendar_features[periods-40:periods])
    model.fit(calendar_features[:periods], all_series[:periods, index])
    base_forecasts[:, index] = model.predict(calendar_features[periods:])
    models[name] = model
covariance = np.cov(residuals, rowvar=False)
shrinkage = 0.4
W = (1-shrinkage)*covariance + shrinkage*np.diag(np.diag(covariance)) + 1e-6*np.eye(len(names))
W_inverse = np.linalg.pinv(W)
projection = S @ np.linalg.pinv(S.T @ W_inverse @ S) @ S.T @ W_inverse
reconciled = base_forecasts @ projection.T
bottom_up = base_forecasts[:, 3:] @ S.T
actual = all_series[periods:]
rows = []
for step in range(horizon):
    for index, name in enumerate(names):
        rows.append({"horizon": step+1, "series": name, "actual": actual[step,index], "base": base_forecasts[step,index], "reconciled": reconciled[step,index], "bottom_up": bottom_up[step,index]})
forecasts = pd.DataFrame(rows)
metrics = []
for name, frame in forecasts.groupby("series"):
    metrics.append({"series": name, **{method+"_mae": mean_absolute_error(frame.actual, frame[method]) for method in ["base", "reconciled", "bottom_up"]}})


metrics = pd.DataFrame(metrics)
coherence_error = np.abs(reconciled[:, 0] - reconciled[:, 3:].sum(axis=1)).max()
assert coherence_error < 1e-8
assert np.allclose(reconciled[:, 1], reconciled[:, 3]+reconciled[:, 4])
assert np.allclose(reconciled[:, 2], reconciled[:, 5]+reconciled[:, 6])
save_table(forecasts, "hierarchical_forecasts")
save_table(metrics, "series_metrics")
save_table(pd.DataFrame(S, index=names).reset_index(names="series"), "summing_matrix")
save_model({"models": models, "projection": projection, "summing_matrix": S, "residual_covariance": W}, "reconciliation_bundle")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
forecasts.loc[forecasts.series.eq("total")].plot(x="horizon", y=["actual", "base", "reconciled"], ax=axes[0])
metrics.plot.bar(x="series", y=["base_mae", "reconciled_mae"], ax=axes[1])
save_figure(fig, "hierarchical_forecasting")
result = finish({"base_mae": mean_absolute_error(actual, base_forecasts), "reconciled_mae": mean_absolute_error(actual, reconciled), "maximum_coherence_error": float(coherence_error)}, [metrics])
