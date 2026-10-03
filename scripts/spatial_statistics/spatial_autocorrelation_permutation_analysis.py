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
WORKFLOW = "141_spatial_autocorrelation_permutation_analysis"


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

points=rng.uniform(0,1,(180,2))
values=np.sin(points[:,0]*6)+np.cos(points[:,1]*5)+rng.normal(0,.35,len(points))
distance=cdist(points,points)
np.fill_diagonal(distance,np.inf)
nearest=np.argsort(distance,axis=1)[:,:7]
weights=np.zeros_like(distance)
weights[np.arange(len(points))[:,None],nearest]=1
weights=np.maximum(weights,weights.T)
weights/=weights.sum(axis=1,keepdims=True)
centered=values-values.mean()
scale=np.mean(centered**2)

def global_moran(vector):
    centered_vector=vector-vector.mean()
    return float(len(vector)/weights.sum()*(centered_vector@weights@centered_vector)/(centered_vector@centered_vector))

observed=global_moran(values)
local=centered*(weights@centered)/scale
null_global=[]
null_local=[]
for iteration in range(499):
    permuted=rng.permutation(centered)
    null_global.append(global_moran(permuted))
    null_local.append(permuted*(weights@permuted)/scale)
null_local=np.asarray(null_local)
local_p=(1+np.sum(abs(null_local)>=abs(local),axis=0))/500
order=np.argsort(local_p)
adjusted=np.empty(len(points))
adjusted[order]=np.minimum.accumulate((local_p[order]*len(points)/np.arange(1,len(points)+1))[::-1])[::-1]
adjusted=np.clip(adjusted,0,1)
lag=weights@centered
quadrant=np.where(centered>=0,np.where(lag>=0,'high_high','high_low'),np.where(lag>=0,'low_high','low_low'))
output=pd.DataFrame({'x':points[:,0],'y':points[:,1],'value':values,'spatial_lag':lag,'local_moran':local,'permutation_pvalue':local_p,'bh_adjusted_pvalue':adjusted,'quadrant':quadrant})


output['significant']=output.bh_adjusted_pvalue<=.05
assert np.allclose(weights.sum(axis=1),1)
assert np.allclose(np.diag(weights),0)
save_table(output,'local_spatial_associations')
save_table(pd.DataFrame({'moran_null':null_global}),'global_permutation_null')
save_table(output.groupby('quadrant').agg(points=('value','size'),significant=('significant','sum')).reset_index(),'quadrant_summary')
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].scatter(points[:,0],points[:,1],c=values,cmap='coolwarm')
axes[1].scatter(centered,lag,c=local,cmap='viridis')
axes[1].set(xlabel='Centered observation',ylabel='Spatial lag')
save_figure(fig,'spatial_dependence_diagnostics')
finish({'global_moran':observed,'null_expectation':-1/(len(points)-1),'positive_autocorrelation_pvalue':(1+sum(value>=observed for value in null_global))/500,'bh_significant_locations':int(output.significant.sum()),'local_null':'unconditional_random_labeling'},[output])
