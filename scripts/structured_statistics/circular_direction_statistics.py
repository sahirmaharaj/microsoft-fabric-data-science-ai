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
WORKFLOW = "139_circular_direction_statistics"


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


from scipy.stats import circmean, circstd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

angle=np.mod(rng.vonmises(np.deg2rad(350),2.2,1000),2*np.pi)
speed=6+2*np.cos(angle-np.deg2rad(30))+.7*np.sin(2*angle)+rng.normal(0,.6,len(angle))
mean_direction=float(circmean(angle,high=2*np.pi,low=0))
resultant=abs(np.mean(np.exp(1j*angle)))
bootstrap=[]
for iteration in range(600):
    sample=rng.choice(angle,len(angle),replace=True)
    delta=np.angle(np.mean(np.exp(1j*sample))*np.exp(-1j*mean_direction))
    bootstrap.append(delta)
interval_delta=np.quantile(bootstrap,[.025,.975])
features=np.column_stack([np.cos(angle),np.sin(angle),np.cos(2*angle),np.sin(2*angle)])
train,test=train_test_split(np.arange(len(angle)),test_size=.3,random_state=SEED)
model=LinearRegression().fit(features[train],speed[train])
grid=np.linspace(0,2*np.pi,361)
grid_features=np.column_stack([np.cos(grid),np.sin(grid),np.cos(2*grid),np.sin(2*grid)])
curve=model.predict(grid_features)
assert np.isclose(curve[0],curve[-1])
sectors=np.floor(angle/(np.pi/4)).astype(int)
sector_summary=pd.DataFrame({'sector':sectors,'speed':speed}).groupby('sector').agg(count=('speed','size'),mean_speed=('speed','mean'),sd_speed=('speed','std')).reset_index()
null_resultant=[]
for iteration in range(500):
    uniform=rng.uniform(0,2*np.pi,len(angle))
    null_resultant.append(abs(np.mean(np.exp(1j*uniform))))
pvalue=(1+sum(value>=resultant for value in null_resultant))/501
circular_interval=np.mod(mean_direction+interval_delta,2*np.pi)
save_table(pd.DataFrame({'direction_degrees':np.rad2deg(angle),'speed':speed,'sector':sectors}),'directional_observations')
save_table(pd.DataFrame({'degrees':np.rad2deg(grid),'predicted_speed':curve}),'periodic_response_curve')
save_table(sector_summary,'direction_sector_statistics')
save_model(model,'circular_harmonic_regression')
save_json({'circular_mean_degrees':np.rad2deg(mean_direction),'mean_interval_endpoints_degrees':np.rad2deg(circular_interval),'interval_wraps_zero':bool(circular_interval[0]>circular_interval[1]),'resultant_length':resultant},'circular_location')
fig=plt.figure(figsize=(7,6))


ax=fig.add_subplot(111,projection='polar')
ax.scatter(angle[:250],speed[:250],s=8,alpha=.3)
ax.plot(grid,curve)
save_figure(fig,'periodic_direction_response')
finish({'circular_mean_degrees':float(np.rad2deg(mean_direction)),'resultant_length':float(resultant),'uniformity_monte_carlo_pvalue':pvalue,'heldout_rmse':float(np.sqrt(np.mean((model.predict(features[test])-speed[test])**2)))},[sector_summary])
