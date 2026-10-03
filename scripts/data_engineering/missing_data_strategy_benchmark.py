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
WORKFLOW = "07_missing_data_strategy_benchmark"


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


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.special import expit

complete = rng.multivariate_normal(np.zeros(7), 0.5 * np.ones((7, 7)) + 0.5 * np.eye(7), SAMPLE_SIZE)
target = complete @ np.array([6, -4, 3, 0, 2, 1, -1]) + rng.normal(0, 1.5, SAMPLE_SIZE)
train_idx, test_idx = train_test_split(np.arange(SAMPLE_SIZE), test_size=0.25, random_state=SEED)
missing_probability = np.broadcast_to(0.08 + 0.35 * expit(complete[:, [0]]), complete.shape).copy()
missing_probability[:, 0] = 0.05
mask = rng.random(complete.shape) < missing_probability
observed = complete.copy()
observed[mask] = np.nan
strategies = {
    "median": SimpleImputer(strategy="median", add_indicator=True),
    "knn": KNNImputer(n_neighbors=7, weights="distance", add_indicator=True),
    "iterative": IterativeImputer(max_iter=20, random_state=SEED, add_indicator=True, tol=0.01)
}
records = []
all_predictions = pd.DataFrame({"row_id": test_idx, "actual": target[test_idx]})
for name, imputer in strategies.items():
    pipeline = Pipeline([("scale", StandardScaler()), ("impute", imputer), ("regression", Ridge(alpha=3.0))])
    started = time.perf_counter()
    pipeline.fit(observed[train_idx], target[train_idx])
    prediction = pipeline.predict(observed[test_idx])
    scaled = pipeline.named_steps["scale"].transform(observed[test_idx])
    filled = pipeline.named_steps["impute"].transform(scaled)[:, :complete.shape[1]]
    recovered = pipeline.named_steps["scale"].inverse_transform(filled)
    masked_error = np.abs(recovered[mask[test_idx]] - complete[test_idx][mask[test_idx]]).mean()
    records.append({"strategy": name, "mae": mean_absolute_error(target[test_idx], prediction), "imputation_mae": float(masked_error), "seconds": time.perf_counter() - started})
    all_predictions[name] = prediction
    save_model(pipeline, name + "_pipeline")


benchmark = pd.DataFrame(records).sort_values("mae")
missingness = pd.DataFrame({"feature": [f"x{i}" for i in range(complete.shape[1])], "train_missing": np.isnan(observed[train_idx]).mean(axis=0), "test_missing": np.isnan(observed[test_idx]).mean(axis=0)})
pattern_counts = pd.Series(["".join(row.astype(int).astype(str)) for row in mask]).value_counts().rename_axis("pattern").rename("rows").reset_index()
bootstrap = []
for name in strategies:
    errors = np.abs(all_predictions.actual - all_predictions[name]).to_numpy()
    draws = rng.choice(errors, size=(400, len(errors)), replace=True).mean(axis=1)
    bootstrap.append({"strategy": name, "mae_lower": float(np.quantile(draws, 0.025)), "mae_upper": float(np.quantile(draws, 0.975))})
intervals = pd.DataFrame(bootstrap)
benchmark = benchmark.merge(intervals, on="strategy")
assert np.isfinite(all_predictions.select_dtypes(include="number")).all().all()
save_table(benchmark, "benchmark")
save_table(all_predictions, "predictions")
save_table(missingness, "missingness")
save_table(pattern_counts, "patterns")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
benchmark.plot.bar(x="strategy", y="mae", ax=axes[0], legend=False)
missingness.plot.bar(x="feature", ax=axes[1])
save_figure(fig, "missing_data")
result = finish({"missing_rate": float(mask.mean()), "best_strategy": str(benchmark.iloc[0].strategy), "best_mae": float(benchmark.iloc[0].mae)}, [benchmark, missingness])
