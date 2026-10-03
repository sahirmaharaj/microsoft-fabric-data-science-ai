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
WORKFLOW = "126_approximate_bayesian_computation"


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


from scipy.stats import gaussian_kde

TRUE_PERSISTENCE=.65
TRUE_NOISE=.8
OBSERVATIONS=100

def simulate(persistence,noise,random):
    series=np.empty(OBSERVATIONS)
    series[0]=random.normal(0,noise/np.sqrt(1-persistence**2))
    for timepoint in range(1,OBSERVATIONS):
        series[timepoint]=persistence*series[timepoint-1]+random.normal(0,noise)
    return series

def summaries(series):
    return np.array([np.std(series,ddof=1),np.corrcoef(series[:-1],series[1:])[0,1],np.mean(abs(np.diff(series)))])

observed=simulate(TRUE_PERSISTENCE,TRUE_NOISE,rng)
observed_summary=summaries(observed)
proposals=7000
parameters=np.column_stack([rng.uniform(-.85,.9,proposals),rng.uniform(.2,1.8,proposals)])
simulated=np.stack([summaries(simulate(persistence,noise,rng)) for persistence,noise in parameters])
scale=np.std(simulated,axis=0,ddof=1)
distance=np.linalg.norm((simulated-observed_summary)/scale,axis=1)
order=np.argsort(distance)
selected=order[:250]
posterior=parameters[selected]
posterior_table=pd.DataFrame(posterior,columns=['persistence','noise'])
posterior_table['distance']=distance[selected]
predictive=[]
for index in rng.choice(len(posterior),150,replace=True):
    series=simulate(*posterior[index],rng)
    predictive.append(summaries(series))
predictive=np.asarray(predictive)
checks=pd.DataFrame({'summary':['standard_deviation','lag1_correlation','mean_absolute_increment'],'observed':observed_summary,'predictive_lower':np.quantile(predictive,.025,axis=0),'predictive_median':np.median(predictive,axis=0),'predictive_upper':np.quantile(predictive,.975,axis=0)})
sensitivity=[]


for retained in [100,250,500,1000]:
    sample=parameters[order[:retained]]
    sensitivity.append({'retained':retained,'epsilon':float(distance[order[retained-1]]),'persistence_mean':float(sample[:,0].mean()),'noise_mean':float(sample[:,1].mean())})
save_table(posterior_table,'abc_posterior_draws')
save_table(checks,'posterior_predictive_checks')
save_table(pd.DataFrame(sensitivity),'acceptance_sensitivity')
save_table(pd.DataFrame({'time':range(OBSERVATIONS),'observed':observed}),'observed_series')
assert len(posterior_table)==250
assert np.isfinite(distance).all()
fig,ax=plt.subplots(figsize=(7,5))
ax.scatter(posterior[:,0],posterior[:,1],s=12,alpha=.5)
ax.scatter([TRUE_PERSISTENCE],[TRUE_NOISE],marker='x',s=100,color='red')
ax.set(xlabel='Persistence',ylabel='Innovation standard deviation')
save_figure(fig,'likelihood_free_posterior')
finish({'simulations':proposals,'accepted':250,'acceptance_rate':250/proposals,'posterior_persistence_mean':float(posterior[:,0].mean()),'posterior_noise_mean':float(posterior[:,1].mean()),'approximation_uses_summary_statistics':True},[checks])
