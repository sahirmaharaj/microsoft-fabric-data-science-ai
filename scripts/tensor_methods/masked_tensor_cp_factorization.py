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
WORKFLOW = "167_masked_tensor_cp_factorization"


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


shape=(24,18,14)
rank=3
true_factors=[rng.normal(size=(size,rank)) for size in shape]
truth=np.einsum('ir,jr,kr->ijk',*true_factors)
observed=truth+rng.normal(0,.08,shape)
mask=rng.random(shape)>.18
A=rng.normal(size=(shape[0],rank))
B=rng.normal(size=(shape[1],rank))
C=rng.normal(size=(shape[2],rank))
reconstruction=np.zeros(shape)
history=[]
regularization=1e-5
for iteration in range(180):
    completed=np.where(mask,observed,reconstruction)
    gram=(B.T@B)*(C.T@C)+regularization*np.eye(rank)
    A=np.linalg.solve(gram,np.einsum('ijk,jr,kr->ir',completed,B,C).T).T
    gram=(A.T@A)*(C.T@C)+regularization*np.eye(rank)
    B=np.linalg.solve(gram,np.einsum('ijk,ir,kr->jr',completed,A,C).T).T
    gram=(A.T@A)*(B.T@B)+regularization*np.eye(rank)
    C=np.linalg.solve(gram,np.einsum('ijk,ir,jr->kr',completed,A,B).T).T
    for factor in [B,C]:
        norms=np.maximum(np.linalg.norm(factor,axis=0),1e-12)
        factor/=norms
        A*=norms
    previous=reconstruction
    reconstruction=np.einsum('ir,jr,kr->ijk',A,B,C)
    change=np.linalg.norm(reconstruction-previous)/max(np.linalg.norm(previous),1e-12)
    history.append({'iteration':iteration,'observed_rmse':float(np.sqrt(np.mean((reconstruction[mask]-observed[mask])**2))),'missing_truth_rmse':float(np.sqrt(np.mean((reconstruction[~mask]-truth[~mask])**2))),'relative_change':float(change)})
    if iteration>20 and change<1e-7:
        break
coordinates=np.argwhere(~mask)
missing=pd.DataFrame(coordinates,columns=['mode0','mode1','mode2'])
missing['truth']=truth[~mask]
missing['imputed']=reconstruction[~mask]
assert np.isfinite(reconstruction).all()


assert np.allclose(np.linalg.norm(B,axis=0),1)
save_table(missing,'missing_tensor_entry_predictions')
save_table(pd.DataFrame(history),'tensor_completion_convergence')
for mode,factor in enumerate([A,B,C]):
    save_table(pd.DataFrame(factor,columns=[f'component_{index}' for index in range(rank)]),f'mode_{mode}_factor')
array_path=output_dir/'tensor_completion_arrays.npz'
np.savez_compressed(array_path,observed=np.where(mask,observed,np.nan),mask=mask,reconstruction=reconstruction)
record_artifact(array_path)
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,array in zip(axes,[truth[:,:,0],np.where(mask[:,:,0],observed[:,:,0],np.nan),reconstruction[:,:,0]]):
    ax.imshow(array,aspect='auto',cmap='coolwarm')
save_figure(fig,'tensor_slice_completion')
finish({'tensor_shape':shape,'rank':rank,'missing_fraction':float((~mask).mean()),'missing_entry_rmse':history[-1]['missing_truth_rmse'],'heldout_values_used_for_fit':False},[pd.DataFrame(history).tail(10)])
