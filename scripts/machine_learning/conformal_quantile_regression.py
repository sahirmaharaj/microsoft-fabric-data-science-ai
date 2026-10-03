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
WORKFLOW = "14_conformal_quantile_regression"


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


from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = make_friedman1(n_samples=SAMPLE_SIZE, n_features=10, noise=1.2, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=SEED)
alpha = 0.1
lower_model = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, n_estimators=120, max_depth=3, random_state=SEED)
upper_model = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, n_estimators=120, max_depth=3, random_state=SEED)
point_model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, random_state=SEED)
for model in [lower_model, upper_model, point_model]:
    model.fit(X_fit, y_fit)
cal_lower = lower_model.predict(X_cal)
cal_upper = upper_model.predict(X_cal)
cal_low = np.minimum(cal_lower, cal_upper)
cal_high = np.maximum(cal_lower, cal_upper)
nonconformity = np.maximum(cal_low - y_cal, y_cal - cal_high)
level = min(1.0, np.ceil((len(y_cal) + 1) * (1 - alpha)) / len(y_cal))
adjustment = max(0.0, float(np.quantile(nonconformity, level, method="higher")))
raw_lower = lower_model.predict(X_test)
raw_upper = upper_model.predict(X_test)
lower = np.minimum(raw_lower, raw_upper) - adjustment
upper = np.maximum(raw_lower, raw_upper) + adjustment
point = point_model.predict(X_test)
covered = (y_test >= lower) & (y_test <= upper)
predictions = pd.DataFrame({"actual": y_test, "point": point, "lower": lower, "upper": upper, "covered": covered, "width": upper - lower})
predictions["slice"] = pd.qcut(X_test[:, 0], q=4, labels=False)
slices = predictions.groupby("slice").agg(rows=("actual", "size"), coverage=("covered", "mean"), mean_width=("width", "mean")).reset_index()
interval_score = upper - lower + 2 / alpha * (lower - y_test) * (y_test < lower) + 2 / alpha * (y_test - upper) * (y_test > upper)


levels = []
for nominal in [0.8, 0.85, 0.9, 0.95]:
    q = min(1, np.ceil((len(y_cal) + 1) * nominal) / len(y_cal))
    delta = max(0.0, float(np.quantile(nonconformity, q, method="higher")))
    lo, hi = np.minimum(raw_lower, raw_upper) - delta, np.maximum(raw_lower, raw_upper) + delta
    levels.append({"nominal": nominal, "empirical": float(((y_test >= lo) & (y_test <= hi)).mean()), "width": float((hi - lo).mean())})
save_table(predictions, "prediction_intervals")
save_table(slices, "conditional_coverage")
save_table(pd.DataFrame(levels), "coverage_sweep")
save_model({"lower": lower_model, "upper": upper_model, "point": point_model, "adjustment": adjustment, "alpha": alpha}, "conformal_bundle")
assert np.all(lower <= upper)
assert np.isfinite(interval_score).all()
order = np.argsort(point)[:100]
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(np.arange(len(order)), lower[order], upper[order], alpha=0.25)
ax.plot(point[order], label="Point")
ax.scatter(np.arange(len(order)), y_test[order], s=12, label="Actual")
ax.legend()
save_figure(fig, "conformal_intervals")
result = finish({"nominal_coverage": 1 - alpha, "empirical_coverage": float(covered.mean()), "mean_width": float((upper - lower).mean()), "interval_score": float(interval_score.mean()), "mae": mean_absolute_error(y_test, point)}, [slices])
