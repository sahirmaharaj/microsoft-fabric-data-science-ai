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
WORKFLOW = "16_feature_selection_and_stability"


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


from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=SAMPLE_SIZE, n_features=40, n_informative=8, n_redundant=6, shuffle=False, random_state=SEED)
feature_names = np.array([f"feature_{i:02d}" for i in range(X.shape[1])])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
selectors = {
    "mutual_information": SelectKBest(__import__("functools").partial(mutual_info_classif, random_state=SEED), k=14),
    "l1_selection": SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.15, random_state=SEED)),
    "recursive_elimination": RFE(LogisticRegression(max_iter=1000), n_features_to_select=14, step=5)
}
folds = StratifiedKFold(4, shuffle=True, random_state=SEED)
results = []
pipelines = {}
for name, selector in selectors.items():
    pipeline = Pipeline([("scale", StandardScaler()), ("select", selector), ("model", LogisticRegression(max_iter=1000))])
    scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="roc_auc", n_jobs=2)
    pipeline.fit(X_train, y_train)
    support = pipeline.named_steps["select"].get_support()
    results.append({"selector": name, "cv_auc": scores.mean(), "cv_std": scores.std(), "selected_features": int(support.sum())})
    pipelines[name] = pipeline
leaderboard = pd.DataFrame(results).sort_values("cv_auc", ascending=False)
chosen_name = str(leaderboard.iloc[0].selector)
chosen = pipelines[chosen_name]
probability = chosen.predict_proba(X_test)[:, 1]
selection_counts = np.zeros(X.shape[1])
from sklearn.base import clone
for iteration in range(24):
    indices = rng.integers(0, len(X_train), len(X_train))
    candidate = clone(chosen).fit(X_train[indices], y_train[indices])
    selection_counts += candidate.named_steps["select"].get_support()


stability = pd.DataFrame({"feature": feature_names, "selected": chosen.named_steps["select"].get_support(), "selection_frequency": selection_counts / 24, "role": np.where(np.arange(40) < 8, "informative", np.where(np.arange(40) < 14, "redundant", "noise"))}).sort_values("selection_frequency", ascending=False)
stable_features = stability.loc[stability.selection_frequency.ge(0.75), "feature"].tolist()
save_table(leaderboard, "selection_benchmark")
save_table(stability, "bootstrap_stability")
save_model(chosen, "selected_feature_pipeline")
save_json({"raw_features": feature_names.tolist(), "selected_features": feature_names[chosen.named_steps["select"].get_support()].tolist(), "stable_features": stable_features}, "feature_selection")
predictions = pd.DataFrame({"actual": y_test, "probability": probability})
save_table(predictions, "test_predictions")
assert len(stability) == X.shape[1]
assert stability.selection_frequency.between(0, 1).all()
fig, ax = plt.subplots(figsize=(10, 5))
stability.head(20).plot.barh(x="feature", y="selection_frequency", ax=ax, legend=False)
save_figure(fig, "selection_stability")
result = finish({"chosen_selector": chosen_name, "test_auc": roc_auc_score(y_test, probability), "stable_features": len(stable_features), "selected_features": int(chosen.named_steps["select"].get_support().sum())}, [leaderboard, stability])
