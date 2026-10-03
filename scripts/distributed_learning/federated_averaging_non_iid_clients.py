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
WORKFLOW = "162_federated_averaging_non_iid_clients"


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
from sklearn.metrics import roc_auc_score, log_loss
from scipy.special import expit

X,y=make_classification(n_samples=2000,n_features=10,n_informative=7,n_redundant=0,random_state=SEED)
train,test=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=SEED)
clients=[[] for _ in range(6)]
for label in [0,1]:
    indices=train[y[train]==label].copy()
    rng.shuffle(indices)
    proportions=rng.dirichlet(np.full(6,.6))
    partitions=np.split(indices,np.cumsum((proportions[:-1]*len(indices)).astype(int)))
    for client,partition in zip(clients,partitions):
        client.extend(partition.tolist())
clients=[np.asarray(client,dtype=int) for client in clients if len(client)>0]
aggregate_count=sum(len(client) for client in clients)
aggregate_sum=sum(X[client].sum(axis=0) for client in clients)
aggregate_square_sum=sum((X[client]**2).sum(axis=0) for client in clients)
mean=aggregate_sum/aggregate_count
scale=np.sqrt(aggregate_square_sum/aggregate_count-mean**2)
features=np.column_stack([np.ones(len(X)),(X-mean)/scale])
weights=np.zeros(features.shape[1])
learning_rate=.12
regularization=.01
history=[]
client_history=[]
for round_id in range(80):
    local_models=[]
    for client_id,indices in enumerate(clients):
        local=weights.copy()
        for epoch in range(4):
            probability=expit(features[indices]@local)
            penalty=regularization*local
            penalty[0]=0
            gradient=features[indices].T@(probability-y[indices])/len(indices)+penalty
            local-=learning_rate*gradient
        local_models.append(local)
        client_history.append({'round':round_id,'client':client_id,'rows':len(indices),'positive_rate':float(y[indices].mean()),'update_norm':float(np.linalg.norm(local-weights))})
    weights=np.average(local_models,axis=0,weights=[len(client) for client in clients])
    probability=expit(features[test]@weights)
    history.append({'round':round_id,'heldout_auc':roc_auc_score(y[test],probability),'heldout_log_loss':log_loss(y[test],probability),'uploaded_float64_bytes':len(clients)*weights.nbytes})


assert sorted(np.concatenate(clients).tolist())==sorted(train.tolist())
assert set(np.concatenate(clients)).isdisjoint(test)
client_metrics=[]
for client_id,indices in enumerate(clients):
    probability=expit(features[indices]@weights)
    client_metrics.append({'client':client_id,'rows':len(indices),'positive_rate':float(y[indices].mean()),'training_log_loss':log_loss(y[indices],probability,labels=[0,1])})
save_table(pd.DataFrame(history),'federated_round_metrics')
save_table(pd.DataFrame(client_history),'client_update_audit')
save_table(pd.DataFrame(client_metrics),'non_iid_client_metrics')
save_json({'weights':weights,'feature_mean':mean,'feature_scale':scale,'local_epochs':4,'regularization':regularization},'federated_logistic_model')
fig,ax=plt.subplots(figsize=(9,4))
ax.plot([row['round'] for row in history],[row['heldout_log_loss'] for row in history])
ax.set(xlabel='Federated round',ylabel='Heldout log loss')
save_figure(fig,'federated_convergence')
finish({'clients':len(clients),'final_auc':history[-1]['heldout_auc'],'final_log_loss':history[-1]['heldout_log_loss'],'differential_privacy_applied':False,'secure_aggregation_applied':False,'shared_preprocessing_uses_aggregate_moments':True},[pd.DataFrame(client_metrics)])
