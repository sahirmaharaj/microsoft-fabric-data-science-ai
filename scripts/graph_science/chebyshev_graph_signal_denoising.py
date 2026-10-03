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
WORKFLOW = "149_chebyshev_graph_signal_denoising"


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
from scipy.linalg import eigh

nodes=180
points=rng.uniform(0,1,(nodes,2))
distance=cdist(points,points)
weights=np.exp(-distance**2/.015)
weights[distance>.23]=0
np.fill_diagonal(weights,0)
degree=weights.sum(axis=1)
normalizer=1/np.sqrt(np.maximum(degree,1e-12))
laplacian=np.eye(nodes)-normalizer[:,None]*weights*normalizer[None,:]
clean=np.sin(points[:,0]*3)+np.cos(points[:,1]*3)
noisy=clean+rng.normal(0,.4,nodes)
signal=np.sqrt(degree)*noisy
scaled_laplacian=laplacian-np.eye(nodes)
quadrature_nodes=120
angles=np.pi*(np.arange(quadrature_nodes)+.5)/quadrature_nodes
spectral_grid=np.cos(angles)


def chebyshev_filter(vector,tau,order):
    response=np.exp(-tau*(spectral_grid+1))
    coefficients=np.array([2/quadrature_nodes*np.sum(response*np.cos(k*angles)) for k in range(order+1)])
    coefficients[0]*=.5
    previous=vector.copy()
    result=coefficients[0]*previous
    if order==0:
        return result
    current=scaled_laplacian@vector
    result+=coefficients[1]*current
    for k in range(2,order+1):
        following=2*scaled_laplacian@current-previous
        result+=coefficients[k]*following
        previous,current=current,following
    return result



values,vectors=eigh(laplacian)
rows=[]
for order in [3,6,12,24]:
    approximate=chebyshev_filter(signal,2.,order)
    exact=vectors@(np.exp(-2*values)*(vectors.T@signal))
    rows.append({'polynomial_order':order,'relative_operator_error':float(np.linalg.norm(approximate-exact)/np.linalg.norm(exact)),'denoising_rmse':float(np.sqrt(np.mean((approximate/np.sqrt(degree)-clean)**2)))})
filtered=chebyshev_filter(signal,2.,24)/np.sqrt(degree)
np.testing.assert_allclose(filtered,exact/np.sqrt(degree),atol=1e-9)
output=pd.DataFrame({'x':points[:,0],'y':points[:,1],'clean':clean,'noisy':noisy,'filtered':filtered})
save_table(output,'graph_signal_values')
save_table(pd.DataFrame(rows),'polynomial_approximation_accuracy')
save_table(pd.DataFrame({'eigenvalue':values}),'graph_laplacian_spectrum')
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,column in zip(axes,['clean','noisy','filtered']):
    ax.scatter(points[:,0],points[:,1],c=output[column],cmap='viridis')
    ax.set_title(column)
save_figure(fig,'local_polynomial_graph_filter')
finish({'noisy_rmse':float(np.sqrt(np.mean((noisy-clean)**2))),'filtered_rmse':float(np.sqrt(np.mean((filtered-clean)**2))),'spectral_reference_verified':True,'filter_order':24},[pd.DataFrame(rows)])
