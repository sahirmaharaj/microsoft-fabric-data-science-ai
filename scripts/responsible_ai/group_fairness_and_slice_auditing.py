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
WORKFLOW = "21_group_fairness_and_slice_auditing"


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
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

n = SAMPLE_SIZE * 2
group = rng.choice(["group_a", "group_b"], n, p=[0.65, 0.35])
region = rng.choice(["north", "south"], n)
X = rng.normal(size=(n, 6))
X[:, 0] += 0.5 * (group == "group_b")
y = rng.binomial(1, expit(0.8 * X[:, 0] - X[:, 1] - 0.7 * (group == "group_b")))
train, test = train_test_split(np.arange(n), test_size=0.3, stratify=y, random_state=SEED)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[train], y[train])
p = model.predict_proba(X[test])[:, 1]
audit = pd.DataFrame({"group": group[test], "region": region[test], "actual": y[test], "probability": p, "decision": p >= 0.5})

def wilson(successes, total):
    if total == 0:
        return 0.0, 1.0
    z = norm.ppf(0.975)
    proportion = successes / total
    center = (proportion + z*z/(2*total)) / (1 + z*z/total)
    radius = z * np.sqrt(proportion*(1-proportion)/total + z*z/(4*total*total)) / (1+z*z/total)
    return float(center-radius), float(center+radius)

def audit_slice(name, frame):
    positives = frame.actual.eq(1)
    negatives = ~positives
    tp = int((positives & frame.decision).sum())
    fp = int((negatives & frame.decision).sum())
    tpr_low, tpr_high = wilson(tp, int(positives.sum()))
    return {"slice": name, "rows": len(frame), "selection_rate": float(frame.decision.mean()), "base_rate": float(frame.actual.mean()), "tpr": tp/max(int(positives.sum()), 1), "fpr": fp/max(int(negatives.sum()), 1), "tpr_lower": tpr_low, "tpr_upper": tpr_high, "brier": brier_score_loss(frame.actual, frame.probability)}



records = [audit_slice("all", audit)]
for key, frame in audit.groupby("group"):
    records.append(audit_slice(key, frame))
for key, frame in audit.groupby(["group", "region"]):
    records.append(audit_slice("/".join(key), frame))
slices = pd.DataFrame(records)
group_rows = slices.loc[slices["slice"].isin(["group_a", "group_b"])]
threshold_rows = []
for threshold in np.linspace(0.1, 0.9, 17):
    modified = audit.copy()
    modified["decision"] = modified.probability >= threshold
    stats = pd.DataFrame([audit_slice(name, frame) for name, frame in modified.groupby("group")])
    threshold_rows.append({"threshold": threshold, "tpr_gap": stats.tpr.max()-stats.tpr.min(), "selection_gap": stats.selection_rate.max()-stats.selection_rate.min(), "accuracy": float((modified.decision == modified.actual).mean())})
tradeoffs = pd.DataFrame(threshold_rows)
save_table(slices, "slice_audit")
save_table(tradeoffs, "threshold_tradeoffs")
save_table(audit, "scored_population")
save_model(model, "audited_model")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
group_rows.plot.bar(x="slice", y=["tpr", "fpr", "selection_rate"], ax=axes[0])
tradeoffs.plot(x="threshold", y=["accuracy", "tpr_gap"], ax=axes[1])
save_figure(fig, "fairness_audit")
result = finish({"auc": roc_auc_score(y[test], p), "equal_opportunity_gap": float(group_rows.tpr.max()-group_rows.tpr.min()), "selection_rate_gap": float(group_rows.selection_rate.max()-group_rows.selection_rate.min()), "minimum_slice_rows": int(slices.rows.min())}, [slices])
