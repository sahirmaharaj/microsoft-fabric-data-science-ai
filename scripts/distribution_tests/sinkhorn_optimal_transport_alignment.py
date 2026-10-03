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
WORKFLOW = "125_sinkhorn_optimal_transport_alignment"


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


from scipy.spatial.distance import cdist
from scipy.special import logsumexp

source=rng.normal(size=(130,2))
rotation=np.array([[.8,-.6],[.6,.8]])
target=rng.normal(size=(150,2))@rotation.T*np.array([1.5,.6])+[2,-1]
cost=cdist(source,target,'sqeuclidean')
a=np.full(len(source),1/len(source))
b=np.full(len(target),1/len(target))

def sinkhorn(epsilon,max_iter=2000):
    log_kernel=-cost/epsilon
    log_u=np.zeros(len(a))
    log_v=np.zeros(len(b))
    history=[]
    for iteration in range(max_iter):
        log_u=np.log(a)-logsumexp(log_kernel+log_v[None,:],axis=1)
        log_v=np.log(b)-logsumexp(log_kernel+log_u[:,None],axis=0)
        if iteration%20==0 or iteration==max_iter-1:
            coupling=np.exp(log_kernel+log_u[:,None]+log_v[None,:])
            error=max(abs(coupling.sum(axis=1)-a).max(),abs(coupling.sum(axis=0)-b).max())
            history.append({'iteration':iteration,'marginal_error':float(error)})
            if error<1e-8:
                break
    coupling=np.exp(log_kernel+log_u[:,None]+log_v[None,:])
    return coupling,history

coupling,history=sinkhorn(.25)
mapped=coupling@target/a[:,None]
comparison=[]
for epsilon in [.1,.25,.7,1.5]:
    plan,trace=sinkhorn(epsilon)
    entropy=-np.sum(plan*np.log(np.maximum(plan,1e-300)))
    comparison.append({'epsilon':epsilon,'transport_cost':float(np.sum(plan*cost)),'entropy':float(entropy),'maximum_marginal_error':trace[-1]['marginal_error'],'iterations':trace[-1]['iteration']+1})
assert np.allclose(coupling.sum(axis=1),a,atol=1e-6)


assert np.allclose(coupling.sum(axis=0),b,atol=1e-6)
assert np.all(coupling>=0)
save_table(pd.DataFrame({'source_x':source[:,0],'source_y':source[:,1],'mapped_x':mapped[:,0],'mapped_y':mapped[:,1]}),'barycentric_transport_map')
save_table(pd.DataFrame(coupling),'transport_coupling')
save_table(pd.DataFrame(history),'sinkhorn_convergence')
save_table(pd.DataFrame(comparison),'regularization_tradeoff')
fig,ax=plt.subplots(figsize=(8,6))
ax.scatter(*source.T,label='source',s=15)
ax.scatter(*target.T,label='target',s=15)
ax.scatter(*mapped.T,label='mapped',s=15)
for index in range(0,len(source),8):
    ax.plot([source[index,0],mapped[index,0]],[source[index,1],mapped[index,1]],color='gray',alpha=.4)
ax.legend()
save_figure(fig,'optimal_transport_alignment')
finish({'transport_cost':float(np.sum(coupling*cost)),'mapped_mean_error':float(np.linalg.norm(mapped.mean(axis=0)-target.mean(axis=0))),'transport_is_entropically_regularized':True},[pd.DataFrame(comparison)])
