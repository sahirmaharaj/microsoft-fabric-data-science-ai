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
WORKFLOW = "163_class_conditional_conformal_prediction_sets"


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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X,y=make_classification(n_samples=1800,n_features=12,n_informative=8,n_redundant=0,n_classes=4,n_clusters_per_class=1,weights=[.45,.25,.2,.1],class_sep=1.1,random_state=SEED)
train,remainder=train_test_split(np.arange(len(y)),test_size=.5,stratify=y,random_state=SEED)
calibration,test=train_test_split(remainder,test_size=.5,stratify=y[remainder],random_state=SEED)
model=RandomForestClassifier(n_estimators=180,min_samples_leaf=3,max_features=.8,n_jobs=2,random_state=SEED).fit(X[train],y[train])
calibration_probability=model.predict_proba(X[calibration])
test_probability=model.predict_proba(X[test])
alpha=.1
scores=1-calibration_probability[np.arange(len(calibration)),y[calibration]]

def finite_sample_quantile(values,error_rate):
    rank=int(np.ceil((len(values)+1)*(1-error_rate)))
    if rank>len(values):
        return 1.
    return float(np.sort(values)[rank-1])

marginal_threshold=finite_sample_quantile(scores,alpha)
class_threshold=np.array([finite_sample_quantile(scores[y[calibration]==label],alpha) for label in range(4)])
sets={'marginal':1-test_probability<=marginal_threshold,'class_conditional':1-test_probability<=class_threshold[None,:]}
rows=[]
predictions=[]
for method,included in sets.items():
    covered=included[np.arange(len(test)),y[test]]
    size=included.sum(axis=1)
    for label in range(4):
        selected=y[test]==label
        rows.append({'method':method,'class':label,'test_rows':int(selected.sum()),'coverage':float(covered[selected].mean()),'mean_set_size':float(size[selected].mean()),'empty_set_rate':float((size[selected]==0).mean())})
    for index,sample in enumerate(test):
        predictions.append({'method':method,'sample':int(sample),'actual':int(y[sample]),'included_classes':json.dumps(np.where(included[index])[0].tolist()),'set_size':int(size[index]),'covered':bool(covered[index])})
assert set(train).isdisjoint(calibration) and set(calibration).isdisjoint(test)
assert len(class_threshold)==model.n_classes_
summary=pd.DataFrame(rows)


save_table(summary,'classwise_prediction_set_coverage')
save_table(pd.DataFrame(predictions),'set_valued_predictions')
save_table(pd.DataFrame({'class':range(4),'threshold':class_threshold,'calibration_rows':[int((y[calibration]==label).sum()) for label in range(4)]}),'mondrian_thresholds')
save_model({'classifier':model,'thresholds':class_threshold,'alpha':alpha},'class_conditional_conformal_model')
fig,ax=plt.subplots(figsize=(9,4))
for method,group in summary.groupby('method'):
    ax.plot(group['class'],group.coverage,marker='o',label=method)
ax.axhline(1-alpha,color='gray',linestyle='--')
ax.legend()
save_figure(fig,'class_conditional_coverage')
finish({'target_coverage':1-alpha,'marginal_threshold':marginal_threshold,'class_thresholds':class_threshold,'assumption':'exchangeability_within_class','coverage_is_a_distributional_not_per_sample_guarantee':True},[summary])
