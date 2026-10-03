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
WORKFLOW = "174_trimmed_icp_point_cloud_registration"


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


from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

source=rng.normal(size=(650,3))*[1.5,.8,.4]
source[:,2]+=.15*source[:,0]**2
true_rotation=Rotation.from_euler('xyz',[8,-6,12],degrees=True).as_matrix()
true_translation=np.array([.3,-.2,.12])
target_clean=source@true_rotation.T+true_translation
target=target_clean+rng.normal(0,.008,target_clean.shape)
target=np.vstack([target,rng.uniform(-4,4,(60,3))])
rotation=np.eye(3)
translation=np.median(target,axis=0)-np.median(source,axis=0)
tree=cKDTree(target)
history=[]
for iteration in range(100):
    aligned=source@rotation.T+translation
    distances,indices=tree.query(aligned)
    retained=distances<=np.quantile(distances,.85)
    left=aligned[retained]
    right=target[indices[retained]]
    left_mean=left.mean(axis=0)
    right_mean=right.mean(axis=0)
    covariance=(left-left_mean).T@(right-right_mean)
    U,singular,Vt=np.linalg.svd(covariance)
    correction=np.eye(3)
    correction[-1,-1]=np.linalg.det(Vt.T@U.T)
    incremental_rotation=Vt.T@correction@U.T
    incremental_translation=right_mean-incremental_rotation@left_mean
    rotation=incremental_rotation@rotation
    translation=incremental_rotation@translation+incremental_translation
    error=float(np.sqrt(np.mean(distances[retained]**2)))
    history.append({'iteration':iteration,'trimmed_nearest_neighbor_rmse':error,'retained_correspondences':int(retained.sum()),'incremental_rotation_radians':float(Rotation.from_matrix(incremental_rotation).magnitude()),'incremental_translation_norm':float(np.linalg.norm(incremental_translation))})
    if np.linalg.norm(incremental_translation)<1e-7 and Rotation.from_matrix(incremental_rotation).magnitude()<1e-7:
        break
aligned=source@rotation.T+translation


rotation_error=float(np.rad2deg(Rotation.from_matrix(rotation@true_rotation.T).magnitude()))
translation_error=float(np.linalg.norm(translation-true_translation))
np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-9)
assert np.isclose(np.linalg.det(rotation),1)
cloud=pd.DataFrame(source,columns=['source_x','source_y','source_z'])
cloud[['aligned_x','aligned_y','aligned_z']]=aligned
cloud[['true_target_x','true_target_y','true_target_z']]=target_clean
save_table(cloud,'registered_point_cloud')
save_table(pd.DataFrame(history),'icp_convergence')
save_json({'rotation':rotation,'translation':translation,'true_rotation':true_rotation,'true_translation':true_translation,'transform_convention':'row_points_at_rotation_transpose_plus_translation'},'rigid_registration_transform')
fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(*target_clean[::3].T,s=8,label='target')
ax.scatter(*aligned[::3].T,s=8,alpha=.5,label='registered')
ax.legend()
save_figure(fig,'registered_3d_clouds')
finish({'rotation_error_degrees':rotation_error,'translation_error':translation_error,'point_correspondence_rmse':float(np.sqrt(np.mean((aligned-target_clean)**2))),'rigid_transform_verified':True,'global_registration_guaranteed':False},[pd.DataFrame(history).tail(10)])
