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
WORKFLOW = "20_uplift_modeling_and_targeting"


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


from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

X = rng.normal(size=(SAMPLE_SIZE * 3, 6))
propensity = 0.5
assignment = rng.binomial(1, propensity, len(X))
base_probability = expit(-1 + 0.6 * X[:, 0] - 0.5 * X[:, 1])
treated_probability = expit(-1 + 0.6 * X[:, 0] - 0.5 * X[:, 1] + 1.2 * (X[:, 2] > 0) - 0.7 * (X[:, 3] > 1))
outcome = rng.binomial(1, np.where(assignment, treated_probability, base_probability))
train, test = train_test_split(np.arange(len(X)), test_size=0.3, random_state=SEED, stratify=assignment)
control_model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=10, min_samples_leaf=30, l2_regularization=5, random_state=SEED)
treatment_model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=10, min_samples_leaf=30, l2_regularization=5, random_state=SEED)
control_train = train[assignment[train] == 0]
treatment_train = train[assignment[train] == 1]
control_model.fit(X[control_train], outcome[control_train])
treatment_model.fit(X[treatment_train], outcome[treatment_train])
p0 = control_model.predict_proba(X[test])[:, 1]
p1 = treatment_model.predict_proba(X[test])[:, 1]
uplift = p1 - p0
transformed_outcome = outcome[test] * (assignment[test] / propensity - (1 - assignment[test]) / (1 - propensity))
ranking = pd.DataFrame({"row_id": test, "treatment": assignment[test], "outcome": outcome[test], "p_control": p0, "p_treated": p1, "uplift": uplift, "transformed_outcome": transformed_outcome, "true_effect": treated_probability[test] - base_probability[test]}).sort_values("uplift", ascending=False).reset_index(drop=True)
ranking["targeted_fraction"] = (np.arange(len(ranking)) + 1) / len(ranking)
ranking["cumulative_incremental"] = ranking.transformed_outcome.cumsum() / len(ranking)
ranking["random_baseline"] = ranking.targeted_fraction * ranking.transformed_outcome.mean()
ranking["decile"] = pd.qcut(np.arange(len(ranking)), 10, labels=False)
deciles = ranking.groupby("decile").agg(rows=("row_id", "size"), predicted_uplift=("uplift", "mean"), observed_ipw_uplift=("transformed_outcome", "mean"), true_uplift=("true_effect", "mean")).reset_index()
revenue_per_conversion, contact_cost = 100.0, 4.0
ranking["target"] = ranking.uplift * revenue_per_conversion > contact_cost
ranking["policy_value_ipw"] = np.where(ranking.target, outcome[test][np.argsort(-uplift)] * assignment[test][np.argsort(-uplift)] / propensity * revenue_per_conversion - contact_cost, outcome[test][np.argsort(-uplift)] * (1 - assignment[test][np.argsort(-uplift)]) / (1 - propensity) * revenue_per_conversion)
qini = float(__import__("scipy").integrate.trapezoid(ranking.cumulative_incremental - ranking.random_baseline, ranking.targeted_fraction))
save_table(ranking, "targeting_decisions")
save_table(deciles, "uplift_deciles")
save_model({"control": control_model, "treated": treatment_model, "contact_cost": contact_cost, "conversion_value": revenue_per_conversion}, "uplift_policy")
save_json({"assignment_probability": propensity, "design": "randomized", "test_rows": len(test)}, "experiment")


assert np.all(np.abs(uplift) <= 1)
assert set(train).isdisjoint(test)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ranking.targeted_fraction, ranking.cumulative_incremental, label="Model")
axes[0].plot(ranking.targeted_fraction, ranking.random_baseline, label="Random")
axes[0].legend()
deciles.plot.bar(x="decile", y=["predicted_uplift", "observed_ipw_uplift"], ax=axes[1])
save_figure(fig, "uplift_curves")
result = finish({"qini_area": qini, "target_rate": float(ranking.target.mean()), "policy_value_ipw": float(ranking.policy_value_ipw.mean()), "uplift_rmse": float(np.sqrt(np.mean((uplift - ranking.set_index("row_id").loc[test, "true_effect"].to_numpy()) ** 2)))}, [deciles])
