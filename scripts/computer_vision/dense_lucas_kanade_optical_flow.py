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
WORKFLOW = "173_dense_lucas_kanade_optical_flow"


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


from scipy.ndimage import gaussian_filter, shift, sobel, uniform_filter, map_coordinates

height=width=128
y,x=np.mgrid[:height,:width]
reference=np.zeros((height,width))
for index in range(55):
    center=rng.uniform(10,118,2)
    size=rng.uniform(2,7)
    reference+=rng.uniform(.3,1)*np.exp(-((x-center[0])**2+(y-center[1])**2)/(2*size**2))
reference/=reference.max()
true_u,true_v=1.8,-1.2
moving=shift(reference,(true_v,true_u),order=3,mode='reflect')
moving=np.clip(moving+rng.normal(0,.002,moving.shape),0,1)
gradient_x=sobel(moving,axis=1)/8
gradient_y=sobel(moving,axis=0)/8
u=np.zeros_like(reference)
v=np.zeros_like(reference)
window=11
history=[]
for iteration in range(25):
    coordinates=np.array([y+v,x+u])
    warped=map_coordinates(moving,coordinates,order=1,mode='reflect')
    Ix=map_coordinates(gradient_x,coordinates,order=1,mode='reflect')
    Iy=map_coordinates(gradient_y,coordinates,order=1,mode='reflect')
    residual=reference-warped
    A=uniform_filter(Ix*Ix,size=window)+1e-6
    B=uniform_filter(Ix*Iy,size=window)
    C=uniform_filter(Iy*Iy,size=window)+1e-6
    D=uniform_filter(Ix*residual,size=window)
    E=uniform_filter(Iy*residual,size=window)
    determinant=A*C-B**2
    du=(C*D-B*E)/np.maximum(determinant,1e-12)
    dv=(A*E-B*D)/np.maximum(determinant,1e-12)
    u=np.clip(u+np.clip(du,-.4,.4),-5,5)
    v=np.clip(v+np.clip(dv,-.4,.4),-5,5)
    history.append({'iteration':iteration,'photometric_rmse':float(np.sqrt(np.mean(residual[10:-10,10:-10]**2))),'mean_update':float(np.mean(np.hypot(du,dv)))})


minimum_eigenvalue=.5*(A+C-np.sqrt((A-C)**2+4*B**2))
reliable=(minimum_eigenvalue>2e-5)&(x>12)&(x<width-13)&(y>12)&(y<height-13)
assert reliable.any()
endpoint_error=np.hypot(u-true_u,v-true_v)
warped=map_coordinates(moving,np.array([y+v,x+u]),order=1,mode='reflect')
save_table(pd.DataFrame(history),'optical_flow_iterations')
save_table(pd.DataFrame({'x':x[reliable],'y':y[reliable],'horizontal_flow':u[reliable],'vertical_flow':v[reliable],'endpoint_error':endpoint_error[reliable]}),'reliable_flow_vectors')
array_path=output_dir/'optical_flow_arrays.npz'
np.savez_compressed(array_path,reference=reference,moving=moving,warped=warped,u=u,v=v,reliable=reliable)
record_artifact(array_path)
fig,axes=plt.subplots(1,3,figsize=(12,4))
axes[0].imshow(reference,cmap='gray')
axes[1].imshow(u,cmap='coolwarm',vmin=-3,vmax=3)
axes[2].imshow(v,cmap='coolwarm',vmin=-3,vmax=3)
save_figure(fig,'dense_motion_field')
finish({'true_horizontal_displacement':true_u,'true_vertical_displacement':true_v,'median_endpoint_error_reliable':float(np.median(endpoint_error[reliable])),'reliable_pixels':int(reliable.sum()),'photometric_rmse_after':float(np.sqrt(np.mean((reference[reliable]-warped[reliable])**2)))},[pd.DataFrame(history).tail(8)])
