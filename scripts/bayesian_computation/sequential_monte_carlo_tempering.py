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
WORKFLOW = "132_sequential_monte_carlo_tempering"


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
from scipy.stats import norm
from scipy.integrate import trapezoid

particles=rng.normal(0,3,2500)
log_weights=np.full(len(particles),-np.log(len(particles)))
observation=2.4
noise=.35

def log_likelihood(theta):
    return norm.logpdf(observation,loc=abs(theta),scale=noise)

def log_prior(theta):
    return norm.logpdf(theta,0,3)

history=[]
previous_temperature=0.
for temperature in np.linspace(.025,1.,40):
    log_weights+=(temperature-previous_temperature)*log_likelihood(particles)
    log_weights-=logsumexp(log_weights)
    weights=np.exp(log_weights)
    ess=1/np.sum(weights**2)
    resampled=False
    if ess<len(particles)*.65:
        indices=rng.choice(len(particles),len(particles),p=weights)
        particles=particles[indices]
        log_weights.fill(-np.log(len(particles)))
        weights.fill(1/len(particles))
        resampled=True
    accepted=0
    for move in range(5):
        proposal=particles+rng.normal(0,.5,len(particles))
        log_ratio=log_prior(proposal)+temperature*log_likelihood(proposal)-log_prior(particles)-temperature*log_likelihood(particles)
        accept=np.log(rng.random(len(particles)))<log_ratio
        particles[accept]=proposal[accept]
        accepted+=int(accept.sum())
    history.append({'temperature':temperature,'effective_sample_size':ess,'resampled':resampled,'move_acceptance':accepted/(5*len(particles)),'positive_mode_mass':float(weights[particles>0].sum())})
    previous_temperature=temperature


weights=np.exp(log_weights)
grid=np.linspace(-6,6,1600)
log_density=log_prior(grid)+log_likelihood(grid)
density=np.exp(log_density-log_density.max())
density/=trapezoid(density,grid)
exact_absolute_mean=trapezoid(abs(grid)*density,grid)
estimated_absolute_mean=float(weights@abs(particles))
assert np.isclose(weights.sum(),1)
assert np.isfinite(particles).all()
save_table(pd.DataFrame({'particle':particles,'weight':weights}),'tempered_posterior_particles')
save_table(pd.DataFrame(history),'tempering_history')
save_table(pd.DataFrame({'theta':grid,'normalized_density':density}),'numerical_reference_posterior')
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].hist(particles,bins=60,weights=weights,density=True,alpha=.5)
axes[0].plot(grid,density)
axes[1].plot([row['temperature'] for row in history],[row['effective_sample_size'] for row in history])
save_figure(fig,'multimodal_tempered_posterior')
finish({'positive_mode_probability':float(weights[particles>0].sum()),'posterior_absolute_mean':estimated_absolute_mean,'reference_absolute_mean':float(exact_absolute_mean),'absolute_mean_error':abs(estimated_absolute_mean-exact_absolute_mean)},[pd.DataFrame(history)])
