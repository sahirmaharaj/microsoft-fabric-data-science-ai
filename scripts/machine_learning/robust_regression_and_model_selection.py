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
WORKFLOW = "13_robust_regression_and_model_selection"


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

contaminated = y_train.copy()
outlier_indices = rng.choice(len(y_train), len(y_train) // 20, replace=False)
contaminated[outlier_indices] += rng.normal(35, 8, len(outlier_indices))
models = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=5)),
    "huber": make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000, epsilon=1.4)),
    "forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=4, random_state=SEED, n_jobs=2),
    "absolute_boosting": HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=SEED)
}
folds = KFold(n_splits=4, shuffle=True, random_state=SEED)
records = []
for name, model in models.items():
    scores = cross_validate(model, X_train, contaminated, cv=folds, scoring={"mae": "neg_mean_absolute_error", "r2": "r2"}, n_jobs=2)
    records.append({"model": name, "cv_mae": -scores["test_mae"].mean(), "cv_mae_std": scores["test_mae"].std(), "cv_r2": scores["test_r2"].mean()})
leaderboard = pd.DataFrame(records).sort_values("cv_mae")
selected_name = str(leaderboard.iloc[0].model)
selected = models[selected_name].fit(X_train, contaminated)
prediction = selected.predict(X_test)
residual = y_test - prediction
predictions = pd.DataFrame({"actual": y_test, "predicted": prediction, "residual": residual, "absolute_error": abs(residual)})
predictions["prediction_decile"] = pd.qcut(predictions.predicted, 10, duplicates="drop").astype(str)
slices = predictions.groupby("prediction_decile").agg(rows=("actual", "size"), mae=("absolute_error", "mean"), bias=("residual", "mean")).reset_index()
bootstrap_mae = []
for _ in range(400):
    indices = rng.integers(0, len(residual), len(residual))
    bootstrap_mae.append(float(np.abs(residual[indices]).mean()))


interval = np.quantile(bootstrap_mae, [0.025, 0.975])
artifact = save_model(selected, "robust_regressor")
assert np.allclose(joblib.load(artifact).predict(X_test), prediction)
save_table(leaderboard, "cross_validation")
save_table(predictions, "predictions")
save_table(slices, "slice_metrics")
save_json({"selected": selected_name, "contamination_fraction": len(outlier_indices) / len(y_train), "mae_interval": interval.tolist()}, "selection")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(prediction, residual, alpha=0.5)
axes[0].axhline(0, color="black")
axes[0].set(xlabel="Prediction", ylabel="Residual")
leaderboard.plot.bar(x="model", y="cv_mae", ax=axes[1], legend=False)
save_figure(fig, "robust_regression")
result = finish({"mae": mean_absolute_error(y_test, prediction), "rmse": np.sqrt(mean_squared_error(y_test, prediction)), "r2": r2_score(y_test, prediction), "mae_lower": interval[0], "mae_upper": interval[1]}, [leaderboard, slices])
