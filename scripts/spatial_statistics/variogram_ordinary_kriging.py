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
WORKFLOW = "140_variogram_ordinary_kriging"


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


from scipy.spatial.distance import cdist, pdist
from scipy.optimize import least_squares
from scipy.linalg import solve

locations=rng.uniform(0,10,(180,2))
full_distance=cdist(locations,locations)
true_covariance=1.8*np.exp(-full_distance/2.8)+np.eye(len(locations))*.08
values=3+rng.multivariate_normal(np.zeros(len(locations)),true_covariance)
training=np.arange(130)
test=np.arange(130,180)
training_distance=pdist(locations[training])
semivariance=.5*pdist(values[training,None],'sqeuclidean')
bins=np.linspace(0,12,16)
rows=[]
for left,right in zip(bins[:-1],bins[1:]):
    selected=(training_distance>=left)&(training_distance<right)
    if selected.sum()>15:
        rows.append({'distance':float(training_distance[selected].mean()),'semivariance':float(semivariance[selected].mean()),'pairs':int(selected.sum())})
empirical=pd.DataFrame(rows)

def variogram(parameters,distance):
    nugget,sill,range_parameter=parameters
    return nugget+sill*(1-np.exp(-distance/range_parameter))

fit=least_squares(lambda parameters:(variogram(parameters,empirical.distance)-empirical.semivariance)*np.sqrt(empirical.pairs),[.1,1.5,3],bounds=([.001,.01,.2],[2,10,20]))
nugget,sill,range_parameter=fit.x
K=sill*np.exp(-cdist(locations[training],locations[training])/range_parameter)+np.eye(len(training))*nugget
system=np.block([[K,np.ones((len(training),1))],[np.ones((1,len(training))),np.zeros((1,1))]])

def krige(query):
    cross=sill*np.exp(-cdist(locations[training],query)/range_parameter)
    solution=solve(system,np.vstack([cross,np.ones((1,len(query)))]),assume_a='sym')
    weights=solution[:-1]
    prediction=values[training]@weights
    variance=sill+nugget-np.sum(weights*cross,axis=0)-solution[-1]
    assert np.allclose(weights.sum(axis=0),1)
    return prediction,np.maximum(variance,0)



prediction,variance=krige(locations[test])
output=pd.DataFrame({'x':locations[test,0],'y':locations[test,1],'actual':values[test],'prediction':prediction,'standard_error':np.sqrt(variance)})
output['covered_95']=abs(output.actual-output.prediction)<=1.96*output.standard_error
grid_x,grid_y=np.meshgrid(np.linspace(0,10,35),np.linspace(0,10,35))
query=np.column_stack([grid_x.ravel(),grid_y.ravel()])
surface,uncertainty=krige(query)
save_table(output,'spatial_holdout_predictions')
save_table(empirical,'empirical_variogram')
save_table(pd.DataFrame({'x':query[:,0],'y':query[:,1],'prediction':surface,'variance':uncertainty}),'kriging_surface')
save_json({'nugget':nugget,'partial_sill':sill,'range':range_parameter,'optimizer_success':bool(fit.success),'parameter_uncertainty_included':False},'fitted_variogram')
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].imshow(surface.reshape(grid_x.shape),origin='lower',extent=[0,10,0,10])
axes[1].imshow(np.sqrt(uncertainty).reshape(grid_x.shape),origin='lower',extent=[0,10,0,10])
save_figure(fig,'kriging_mean_uncertainty')
finish({'heldout_rmse':float(np.sqrt(np.mean((prediction-values[test])**2))),'conditional_interval_coverage':float(output.covered_95.mean()),'kriging_weight_sum_verified':True},[output])
