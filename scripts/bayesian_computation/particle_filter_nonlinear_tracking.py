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
WORKFLOW = "130_particle_filter_nonlinear_tracking"


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


from scipy.special import logsumexp

steps=120
particle_count=1500
process_sd=.7
observation_sd=.8
truth=np.zeros(steps)
observations=np.zeros(steps)

def transition(state,timepoint):
    return .5*state+8*state/(1+state**2)+2*np.cos(1.2*timepoint)

for timepoint in range(steps):
    previous=truth[timepoint-1] if timepoint else 0.
    truth[timepoint]=transition(previous,timepoint)+rng.normal(0,process_sd)
    observations[timepoint]=truth[timepoint]**2/12+rng.normal(0,observation_sd)
particles=rng.normal(0,2,particle_count)
weights=np.full(particle_count,1/particle_count)
rows=[]
resamples=0

def systematic_resample(probabilities):
    positions=(rng.random()+np.arange(len(probabilities)))/len(probabilities)
    cumulative=np.cumsum(probabilities)
    cumulative[-1]=1.
    return np.searchsorted(cumulative,positions)

for timepoint,observation in enumerate(observations):
    particles=transition(particles,timepoint)+rng.normal(0,process_sd,particle_count)
    log_weights=np.log(np.maximum(weights,1e-300))-.5*((observation-particles**2/12)/observation_sd)**2
    log_weights-=logsumexp(log_weights)
    weights=np.exp(log_weights)
    estimate=float(weights@particles)
    order=np.argsort(particles)
    cumulative=np.cumsum(weights[order])
    lower,upper=np.interp([.025,.975],cumulative,particles[order])
    ess=1/np.sum(weights**2)
    rows.append({'time':timepoint,'truth':truth[timepoint],'observation':observation,'posterior_mean':estimate,'lower':lower,'upper':upper,'effective_sample_size':ess,'positive_state_probability':float(weights[particles>0].sum())})
    if ess<particle_count*.5:
        particles=particles[systematic_resample(weights)]
        weights.fill(1/particle_count)
        resamples+=1


output=pd.DataFrame(rows)
assert np.allclose(weights.sum(),1)
assert output.effective_sample_size.between(1,particle_count+1e-6).all()
save_table(output,'particle_filter_trajectory')
save_table(pd.DataFrame({'particle':particles,'weight':weights}),'final_weighted_particles')
fig,axes=plt.subplots(2,1,figsize=(11,7))
axes[0].plot(output.time,output.truth,label='truth')
axes[0].plot(output.time,output.posterior_mean,label='filtered mean')
axes[0].fill_between(output.time,output.lower,output.upper,alpha=.2)
axes[0].legend()
axes[1].plot(output.time,output.effective_sample_size)
save_figure(fig,'nonlinear_particle_tracking')
finish({'filter_rmse':float(np.sqrt(np.mean((output.posterior_mean-output.truth)**2))),'credible_interval_coverage':float(output.truth.between(output.lower,output.upper).mean()),'resampling_count':resamples,'particles':particle_count},[output])
