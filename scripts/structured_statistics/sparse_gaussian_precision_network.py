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
WORKFLOW = "134_sparse_gaussian_precision_network"


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


from sklearn.covariance import GraphicalLassoCV, EmpiricalCovariance
from sklearn.model_selection import train_test_split

features=16
precision=np.eye(features)*1.6
edges=[]
for node in range(features-1):
    precision[node,node+1]=precision[node+1,node]=-.35
    edges.append((node,node+1))
for left,right in [(0,5),(3,9),(8,14)]:
    precision[left,right]=precision[right,left]=-.25
    edges.append((left,right))
covariance=np.linalg.inv(precision)
X=rng.multivariate_normal(np.zeros(features),covariance,700)
train,test=train_test_split(np.arange(len(X)),test_size=.3,random_state=SEED)
model=GraphicalLassoCV(alphas=6,cv=4,max_iter=300).fit(X[train])
empirical=EmpiricalCovariance().fit(X[train])
partial=-model.precision_/np.sqrt(np.outer(np.diag(model.precision_),np.diag(model.precision_)))
np.fill_diagonal(partial,1)
records=[]
for left in range(features):
    for right in range(left+1,features):
        records.append({'left':left,'right':right,'partial_correlation':partial[left,right],'true_edge':(left,right) in edges,'selected':abs(partial[left,right])>.04})
edge_table=pd.DataFrame(records)
tp=int((edge_table.true_edge&edge_table.selected).sum())
conditional=[]
for target in range(features):
    other=np.array([index for index in range(features) if index!=target])
    coefficient=-model.precision_[target,other]/model.precision_[target,target]
    prediction=model.location_[target]+(X[test][:,other]-model.location_[other])@coefficient
    conditional.append({'variable':target,'conditional_rmse':float(np.sqrt(np.mean((prediction-X[test,target])**2))),'conditional_sd':float(np.sqrt(1/model.precision_[target,target]))})
assert np.linalg.eigvalsh(model.precision_).min()>0
assert np.allclose(partial,partial.T)
save_table(edge_table,'conditional_dependence_edges')
save_table(pd.DataFrame(conditional),'conditional_prediction_errors')


save_table(pd.DataFrame(partial),'partial_correlation_matrix')
save_model(model,'sparse_precision_estimator')
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].imshow(precision,cmap='coolwarm')
axes[1].imshow(model.precision_,cmap='coolwarm')
save_figure(fig,'sparse_precision_recovery')
finish({'selected_alpha':float(model.alpha_),'edge_precision':tp/max(int(edge_table.selected.sum()),1),'edge_recall':tp/len(edges),'graphical_lasso_test_loglikelihood':model.score(X[test]),'empirical_test_loglikelihood':empirical.score(X[test]),'edges_are_conditional_associations':True},[pd.DataFrame(conditional)])
