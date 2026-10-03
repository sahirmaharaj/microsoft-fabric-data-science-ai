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
WORKFLOW = "46_champion_challenger_promotion_gate"


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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, confusion_matrix

X_array, y_array = make_classification(n_samples=SAMPLE_SIZE, n_features=16, n_informative=9, n_redundant=3, weights=[0.72, 0.28], class_sep=0.9, random_state=SEED)
X = pd.DataFrame(X_array, columns=[f"feature_{i:02d}" for i in range(16)])
y = pd.Series(y_array, name="target")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

from sklearn.metrics import brier_score_loss

X_fit,X_validation,y_fit,y_validation=train_test_split(X_train,y_train,stratify=y_train,test_size=0.3,random_state=SEED)
champion=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(X_fit,y_fit)
challenger=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=3,random_state=SEED).fit(X_fit,y_fit)
champion_p=champion.predict_proba(X_validation)[:,1]
challenger_p=challenger.predict_proba(X_validation)[:,1]
paired_deltas=[]
for iteration in range(500):
    indices=rng.integers(0,len(y_validation),len(y_validation))
    actual=y_validation.to_numpy()[indices]
    if len(np.unique(actual))<2:
        continue
    paired_deltas.append(roc_auc_score(actual,challenger_p[indices])-roc_auc_score(actual,champion_p[indices]))
interval=np.quantile(paired_deltas,[0.025,0.975])
latency_rows=[]
for name,model in [("champion",champion),("challenger",challenger)]:
    model.predict_proba(X_validation.iloc[:20])
    durations=[]
    for repeat in range(15):
        started=time.perf_counter()
        model.predict_proba(X_validation)
        durations.append((time.perf_counter()-started)*1000)
    latency_rows.append({"model":name,"p50_ms":float(np.median(durations)),"p95_ms":float(np.quantile(durations,0.95))})


latency=pd.DataFrame(latency_rows)
checks=pd.DataFrame([
    {"gate":"auc_improvement_lower_bound","value":float(interval[0]),"threshold":0.0,"passed":bool(interval[0]>0)},
    {"gate":"brier_noninferiority","value":float(brier_score_loss(y_validation,challenger_p)-brier_score_loss(y_validation,champion_p)),"threshold":0.02,"passed":bool(brier_score_loss(y_validation,challenger_p)<=brier_score_loss(y_validation,champion_p)+0.02)},
    {"gate":"latency_budget_ms","value":float(latency.loc[latency.model.eq("challenger"),"p95_ms"].iloc[0]),"threshold":500.0,"passed":bool(latency.loc[latency.model.eq("challenger"),"p95_ms"].iloc[0]<500)}
])
promote=bool(checks.passed.all())
selected=challenger if promote else champion
selected_name="challenger" if promote else "champion"
test_probability=selected.predict_proba(X_test)[:,1]
test_predictions=pd.DataFrame({"actual":y_test.to_numpy(),"probability":test_probability})
save_table(checks,"promotion_gates")
save_table(latency,"latency_benchmark")
save_table(pd.DataFrame({"auc_delta":paired_deltas}),"paired_bootstrap")
save_table(test_predictions,"selected_model_test_predictions")
save_model(champion,"champion")
save_model(challenger,"challenger")
save_json({"selected":selected_name,"promote":promote,"validation_auc_delta_interval":interval,"test_used_for_selection":False},"promotion_decision")
assert set(X_validation.index).isdisjoint(X_test.index)
assert promote==bool(checks.passed.all())
fig,ax=plt.subplots(figsize=(9,4))
ax.hist(paired_deltas,bins=30)
ax.axvline(0,color="red")
ax.set(xlabel="Paired validation AUC improvement",ylabel="Bootstrap draws")
save_figure(fig,"promotion_evidence")
result=finish({"selected_model":selected_name,"promoted":promote,"test_auc":roc_auc_score(y_test,test_probability),"validation_auc_delta_lower":interval[0],"validation_auc_delta_upper":interval[1]},[checks,latency])
