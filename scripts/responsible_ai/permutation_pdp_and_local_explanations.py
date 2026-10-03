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
WORKFLOW = "22_permutation_pdp_and_local_explanations"


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

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

model = RandomForestRegressor(n_estimators=140, min_samples_leaf=3, random_state=SEED, n_jobs=2).fit(X_train, y_train)
prediction = model.predict(X_test)
importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=SEED, scoring="neg_mean_absolute_error", n_jobs=2)
importance_table = pd.DataFrame({"feature": [f"feature_{i}" for i in range(X.shape[1])], "importance": importance.importances_mean, "std": importance.importances_std}).sort_values("importance", ascending=False)
feature_scale = np.maximum(X_train.std(axis=0), 1e-6)
local_rows = []
local_diagnostics = []
for row_id in [0, 1, 2, 3, 4]:
    instance = X_test[row_id]
    perturbations = instance + rng.normal(0, 0.3, (800, X.shape[1])) * feature_scale
    perturbations = np.clip(perturbations, X_train.min(axis=0), X_train.max(axis=0))
    standardized = (perturbations - instance) / feature_scale
    distance = np.linalg.norm(standardized, axis=1)
    weights = np.exp(-(distance ** 2) / 0.75 ** 2)
    local_target = model.predict(perturbations)
    local_fit, local_holdout = train_test_split(np.arange(len(perturbations)), test_size=0.25, random_state=SEED)
    surrogate = Ridge(alpha=0.1).fit(standardized[local_fit], local_target[local_fit], sample_weight=weights[local_fit])
    fidelity = r2_score(local_target[local_holdout], surrogate.predict(standardized[local_holdout]), sample_weight=weights[local_holdout])
    local_diagnostics.append({"row_id": row_id, "model_prediction": float(model.predict(instance.reshape(1, -1))[0]), "surrogate_prediction": float(surrogate.predict(np.zeros((1, X.shape[1])))[0]), "weighted_holdout_r2": fidelity})
    for index, coefficient in enumerate(surrogate.coef_):
        local_rows.append({"row_id": row_id, "feature": f"feature_{index}", "value": instance[index], "local_sensitivity": coefficient})


local_explanations = pd.DataFrame(local_rows)
diagnostics = pd.DataFrame(local_diagnostics)
response_rows = []
for feature in [0, 1, 2]:
    for value in np.linspace(0.05, 0.95, 20):
        changed = X_test.copy()
        changed[:, feature] = value
        response_rows.append({"feature": feature, "value": value, "mean_prediction": float(model.predict(changed).mean())})
response = pd.DataFrame(response_rows)
save_table(importance_table, "global_permutation_importance")
save_table(local_explanations, "local_surrogates")
save_table(diagnostics, "surrogate_fidelity")
save_table(response, "partial_dependence")
save_model(model, "explained_model")
assert len(local_explanations) == 5 * X.shape[1]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
importance_table.plot.barh(x="feature", y="importance", xerr="std", ax=axes[0], legend=False)
for feature, frame in response.groupby("feature"):
    axes[1].plot(frame.value, frame.mean_prediction, label=f"feature_{feature}")
axes[1].legend()
save_figure(fig, "explanations")
result = finish({"mae": mean_absolute_error(y_test, prediction), "mean_surrogate_holdout_r2": float(diagnostics.weighted_holdout_r2.mean()), "top_feature": str(importance_table.iloc[0].feature)}, [importance_table, diagnostics])
