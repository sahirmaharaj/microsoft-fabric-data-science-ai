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
WORKFLOW = "15_multioutput_demand_and_inventory"


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


from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

X = rng.normal(size=(SAMPLE_SIZE, 8))
season = np.sin(X[:, 0])
shared = 20 * season + 10 * X[:, 1] + 5 * X[:, 2] ** 2
Y = np.column_stack([100 + shared + rng.normal(0, 3, SAMPLE_SIZE), 80 + 0.7 * shared + 8 * X[:, 3] + rng.normal(0, 4, SAMPLE_SIZE), 60 + 0.4 * shared - 8 * X[:, 4] + rng.normal(0, 3, SAMPLE_SIZE)])
Y = np.maximum(Y, 0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=SEED)
model = RandomForestRegressor(n_estimators=140, min_samples_leaf=3, max_features=0.8, random_state=SEED, n_jobs=2)
search = GridSearchCV(model, {"min_samples_leaf": [2, 5, 10]}, scoring="neg_mean_absolute_error", cv=3, n_jobs=2)
search.fit(X_train, y_train)
model = search.best_estimator_
prediction = np.maximum(model.predict(X_test), 0)
folds = KFold(4, shuffle=True, random_state=SEED)
out_of_fold = np.zeros_like(y_train)
from sklearn.base import clone
for fit_indices, validation_indices in folds.split(X_train):
    fold_model = clone(model).fit(X_train[fit_indices], y_train[fit_indices])
    out_of_fold[validation_indices] = fold_model.predict(X_train[validation_indices])
residuals = y_train - out_of_fold
service_level = 0.95
safety_stock = np.maximum(0, np.quantile(residuals, service_level, axis=0))
reorder = np.maximum(0, prediction + safety_stock)
unit_holding_cost = np.array([2.0, 1.5, 1.0])
unit_stockout_cost = np.array([15.0, 12.0, 8.0])
rows = []
for index in range(Y.shape[1]):
    over = np.maximum(0, reorder[:, index] - y_test[:, index])
    under = np.maximum(0, y_test[:, index] - reorder[:, index])
    rows.append({"product": f"product_{index}", "mae": mean_absolute_error(y_test[:, index], prediction[:, index]), "r2": r2_score(y_test[:, index], prediction[:, index]), "safety_stock": safety_stock[index], "service_level": float((under == 0).mean()), "mean_cost": float((over * unit_holding_cost[index] + under * unit_stockout_cost[index]).mean())})


metrics = pd.DataFrame(rows)
output = pd.DataFrame({f"actual_{i}": y_test[:, i] for i in range(3)})
for i in range(3):
    output[f"prediction_{i}"] = prediction[:, i]
    output[f"reorder_{i}"] = reorder[:, i]
residual_covariance = pd.DataFrame(np.cov(residuals, rowvar=False), columns=["product_0", "product_1", "product_2"]).reset_index(names="product")
save_table(metrics, "product_metrics")
save_table(output, "inventory_decisions")
save_table(residual_covariance, "residual_covariance")
save_model({"model": model, "safety_stock": safety_stock}, "inventory_bundle")
assert prediction.shape == y_test.shape
assert (reorder >= prediction).all()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
metrics.plot.bar(x="product", y="mae", ax=axes[0], legend=False)
metrics.plot.bar(x="product", y="service_level", ax=axes[1], legend=False)
save_figure(fig, "inventory_metrics")
result = finish({"overall_mae": mean_absolute_error(y_test, prediction), "mean_service_level": float(metrics.service_level.mean()), "mean_inventory_cost": float(metrics.mean_cost.mean())}, [metrics])
