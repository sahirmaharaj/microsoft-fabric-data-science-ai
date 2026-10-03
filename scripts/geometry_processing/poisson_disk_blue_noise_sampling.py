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
WORKFLOW = "175_poisson_disk_blue_noise_sampling"


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


from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import pdist

radius=.055
cell_size=radius/np.sqrt(2)
grid_size=int(np.ceil(1/cell_size))
grid=np.full((grid_size,grid_size),-1,dtype=int)
points=[rng.uniform(0,1,2)]
active=[0]
initial_cell=np.floor(points[0]/cell_size).astype(int)
grid[tuple(initial_cell)]=0
attempts=30
rejections=0
while active:
    active_position=int(rng.integers(len(active)))
    center=points[active[active_position]]
    accepted=False
    for attempt in range(attempts):
        angle=rng.uniform(0,2*np.pi)
        distance=np.sqrt(rng.uniform(radius**2,4*radius**2))
        candidate=center+distance*np.array([np.cos(angle),np.sin(angle)])
        if np.any(candidate<0) or np.any(candidate>=1):
            rejections+=1
            continue
        cell=np.floor(candidate/cell_size).astype(int)
        neighboring=grid[max(0,cell[0]-2):min(grid_size,cell[0]+3),max(0,cell[1]-2):min(grid_size,cell[1]+3)]
        neighbor_indices=neighboring[neighboring>=0]
        if any(np.linalg.norm(candidate-points[index])<radius for index in neighbor_indices):
            rejections+=1
            continue
        index=len(points)
        points.append(candidate)
        active.append(index)
        grid[tuple(cell)]=index
        accepted=True
        break
    if not accepted:
        active.pop(active_position)


points=np.asarray(points)
uniform=rng.uniform(0,1,points.shape)
poisson_nearest=cKDTree(points).query(points,k=2)[0][:,1]
uniform_nearest=cKDTree(uniform).query(uniform,k=2)[0][:,1]
assert pdist(points).min()>=radius-1e-12
triangulation=Delaunay(points)
triangle_points=points[triangulation.simplices]
v1=triangle_points[:,1]-triangle_points[:,0]
v2=triangle_points[:,2]-triangle_points[:,0]
areas=.5*abs(v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0])
triangles=pd.DataFrame(triangulation.simplices,columns=['vertex_0','vertex_1','vertex_2'])
triangles['area']=areas
comparison=pd.DataFrame({'design':['poisson_disk','uniform'],'minimum_neighbor_distance':[poisson_nearest.min(),uniform_nearest.min()],'mean_neighbor_distance':[poisson_nearest.mean(),uniform_nearest.mean()],'neighbor_distance_cv':[poisson_nearest.std()/poisson_nearest.mean(),uniform_nearest.std()/uniform_nearest.mean()]})
save_table(pd.DataFrame(points,columns=['x','y']),'blue_noise_sample_points')
save_table(triangles,'delaunay_mesh_triangles')
save_table(comparison,'sampling_uniformity_comparison')
fig,axes=plt.subplots(1,2,figsize=(10,5))
axes[0].triplot(points[:,0],points[:,1],triangulation.simplices,linewidth=.5)
axes[0].scatter(*points.T,s=8)
axes[1].scatter(*uniform.T,s=8)
for ax in axes:
    ax.set_aspect('equal')
save_figure(fig,'blue_noise_vs_uniform_sampling')
finish({'sample_count':len(points),'minimum_required_distance':radius,'observed_minimum_distance':float(poisson_nearest.min()),'rejected_candidates':rejections,'mesh_triangles':len(triangles),'minimum_spacing_verified':True},[comparison])
