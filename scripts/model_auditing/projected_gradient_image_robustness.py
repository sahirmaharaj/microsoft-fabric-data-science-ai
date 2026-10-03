import os
import sys
import subprocess
import importlib.util
import importlib.metadata
import site

if site.ENABLE_USER_SITE:
    site.addsitedir(site.getusersitepackages())

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
importlib.invalidate_caches()
for thread_variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(thread_variable, "2")


SEED = 42
SAMPLE_SIZE = 1200
OUTPUT_ROOT = os.environ.get("FABRIC_STARTER_OUTPUT", "")
WORKFLOW = "165_projected_gradient_image_robustness"


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


from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import softmax

dataset=load_digits()
X=dataset.data/16.
y=dataset.target
train,test=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=SEED)
model=LogisticRegression(C=1,max_iter=1500).fit(X[train],y[train])
query=X[test[:180]]
truth=y[test[:180]]
clean_prediction=model.predict(query)

def adversarial_examples(epsilon,steps=30):
    candidate=np.clip(query+rng.uniform(-epsilon,epsilon,query.shape),0,1)
    for iteration in range(steps):
        probability=softmax(candidate@model.coef_.T+model.intercept_,axis=1)
        probability[np.arange(len(truth)),truth]-=1
        gradient=probability@model.coef_
        candidate+=epsilon/6*np.sign(gradient)
        candidate=np.minimum(np.maximum(candidate,query-epsilon),query+epsilon)
        candidate=np.clip(candidate,0,1)
    assert np.max(abs(candidate-query))<=epsilon+1e-10
    return candidate

rows=[]
examples={}
for epsilon in [0,.03,.06,.1,.15,.2]:
    attacked=query.copy() if epsilon==0 else adversarial_examples(epsilon)
    prediction=model.predict(attacked)
    probability=model.predict_proba(attacked)
    clean_correct=clean_prediction==truth
    rows.append({'epsilon':epsilon,'robust_accuracy':accuracy_score(truth,prediction),'attack_success_on_clean_correct':float((prediction[clean_correct]!=truth[clean_correct]).mean()),'mean_true_class_probability':float(probability[np.arange(len(truth)),truth].mean()),'max_linf_perturbation':float(np.max(abs(attacked-query)))})
    examples[epsilon]=attacked


selected=examples[.15]
output=pd.DataFrame({'sample':test[:180],'actual':truth,'clean_prediction':clean_prediction,'attacked_prediction':model.predict(selected),'linf_distance':np.max(abs(selected-query),axis=1)})
save_table(pd.DataFrame(rows),'robustness_budget_curve')
save_table(output,'per_image_attack_audit')
array_path=output_dir/'adversarial_digit_arrays.npz'
np.savez_compressed(array_path,clean=query,adversarial=selected,labels=truth)
record_artifact(array_path)
save_model(model,'digit_classifier')
fig,axes=plt.subplots(2,6,figsize=(11,4))
for index in range(6):
    axes[0,index].imshow(query[index].reshape(8,8),cmap='gray',vmin=0,vmax=1)
    axes[1,index].imshow(selected[index].reshape(8,8),cmap='gray',vmin=0,vmax=1)
    axes[0,index].axis('off')
    axes[1,index].axis('off')
save_figure(fig,'bounded_image_perturbations')
finish({'clean_accuracy':accuracy_score(truth,clean_prediction),'epsilon015_accuracy':accuracy_score(truth,model.predict(selected)),'threat_model':'untargeted_white_box_linf','pixel_bounds_and_perturbation_bounds_verified':True},[pd.DataFrame(rows)])
