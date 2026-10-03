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
WORKFLOW = "124_kernel_two_sample_mmd"


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


from scipy.spatial.distance import pdist, squareform, cdist

reference=rng.normal(size=(160,4))
shifted=rng.normal(size=(160,4))
shifted[:,0]+=1.
shifted[:,1]*=1.6
pooled=np.vstack([reference,shifted])
bandwidth=float(np.median(pdist(pooled)))
K=np.exp(-squareform(pdist(pooled,'sqeuclidean'))/(2*bandwidth**2))

def mmd_from_indices(kernel,left,right):
    A=kernel[np.ix_(left,left)]
    B=kernel[np.ix_(right,right)]
    C=kernel[np.ix_(left,right)]
    return float((A.sum()-np.trace(A))/(len(left)*(len(left)-1))+(B.sum()-np.trace(B))/(len(right)*(len(right)-1))-2*C.mean())

left=np.arange(160)
right=np.arange(160,320)
observed=mmd_from_indices(K,left,right)
null=[]
for iteration in range(399):
    order=rng.permutation(320)
    null.append(mmd_from_indices(K,order[:160],order[160:]))
pvalue=(1+np.sum(np.asarray(null)>=observed))/400
witness=K[:,left].mean(axis=1)-K[:,right].mean(axis=1)
output=pd.DataFrame(pooled,columns=['x0','x1','x2','x3'])
output['population']=['reference']*160+['shifted']*160
output['witness']=witness
sensitivity=[]
for multiplier in [.25,.5,1,2,4]:
    kernel=np.exp(-squareform(pdist(pooled,'sqeuclidean'))/(2*(bandwidth*multiplier)**2))
    sensitivity.append({'bandwidth_multiplier':multiplier,'unbiased_mmd_squared':mmd_from_indices(kernel,left,right)})
assert np.allclose(K,K.T)
assert np.allclose(np.diag(K),1)
assert 0<pvalue<=1


save_table(output,'kernel_witness_values')
save_table(pd.DataFrame({'permutation_mmd_squared':null}),'permutation_null')
save_table(pd.DataFrame(sensitivity),'bandwidth_sensitivity')
save_json({'bandwidth':bandwidth,'pooled_bandwidth_is_label_invariant':True,'statistic':'unbiased_mmd_squared','permutations':399},'test_configuration')
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].hist(null,bins=25)
axes[0].axvline(observed,color='red')
axes[1].scatter(pooled[:,0],pooled[:,1],c=witness,cmap='coolwarm',s=15)
save_figure(fig,'mmd_distribution_test')
finish({'unbiased_mmd_squared':observed,'permutation_pvalue':float(pvalue),'bandwidth':bandwidth},[pd.DataFrame(sensitivity)])
