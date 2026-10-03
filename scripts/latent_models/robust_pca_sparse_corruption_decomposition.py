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
WORKFLOW = "121_robust_pca_sparse_corruption_decomposition"


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


from scipy.linalg import svd

rows,columns,rank=100,80,4
truth=rng.normal(size=(rows,rank))@rng.normal(size=(rank,columns))
corruption=np.zeros((rows,columns))
locations=rng.random((rows,columns))<.06
corruption[locations]=rng.normal(0,12,locations.sum())
observed=truth+corruption
lambda_sparse=1/np.sqrt(max(observed.shape))
spectral_norm=np.linalg.norm(observed,2)
dual=observed/max(spectral_norm,np.max(abs(observed))/lambda_sparse)
mu=1.25/spectral_norm
mu_limit=mu*1e7
sparse=np.zeros_like(observed)
low_rank=np.zeros_like(observed)
history=[]

def soft_threshold(values,threshold):
    return np.sign(values)*np.maximum(abs(values)-threshold,0)

for iteration in range(400):
    U,singular,Vt=svd(observed-sparse+dual/mu,full_matrices=False)
    shrunk=np.maximum(singular-1/mu,0)
    low_rank=(U*shrunk)@Vt
    sparse=soft_threshold(observed-low_rank+dual/mu,lambda_sparse/mu)
    residual=observed-low_rank-sparse
    relative=np.linalg.norm(residual)/np.linalg.norm(observed)
    history.append({'iteration':iteration,'constraint_error':relative,'estimated_rank':int((shrunk>1e-7).sum()),'sparse_entries':int((abs(sparse)>1e-4).sum())})
    dual+=mu*residual
    mu=min(mu*1.5,mu_limit)
    if relative<1e-7:
        break
predicted=abs(sparse)>.1
true_positive=int((predicted&locations).sum())
precision=true_positive/max(int(predicted.sum()),1)


recall=true_positive/max(int(locations.sum()),1)
U,singular,Vt=svd(observed,full_matrices=False)
ordinary=(U[:,:rank]*singular[:rank])@Vt[:rank]
comparison=pd.DataFrame({'method':['truncated_svd','robust_pca'],'relative_recovery_error':[np.linalg.norm(ordinary-truth)/np.linalg.norm(truth),np.linalg.norm(low_rank-truth)/np.linalg.norm(truth)]})
save_table(pd.DataFrame(history),'optimization_history')
save_table(comparison,'low_rank_recovery_comparison')
save_table(pd.DataFrame(low_rank),'recovered_low_rank')
save_table(pd.DataFrame(sparse),'recovered_sparse_corruption')
assert history[-1]['constraint_error']<1e-5
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,matrix in zip(axes,[observed,low_rank,sparse]):
    ax.imshow(matrix,aspect='auto',cmap='coolwarm')
save_figure(fig,'low_rank_sparse_decomposition')
finish({'iterations':len(history),'corruption_precision':precision,'corruption_recall':recall,'relative_recovery_error':float(comparison.iloc[1].relative_recovery_error)},[comparison])
