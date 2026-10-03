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
WORKFLOW = "166_random_feature_score_matching_diffusion"


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


from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import cdist

angles=np.arange(8)*2*np.pi/8
centers=2*np.column_stack([np.cos(angles),np.sin(angles)])
training=centers[rng.integers(0,8,2500)]+rng.normal(0,.18,(2500,2))
heldout=centers[rng.integers(0,8,600)]+rng.normal(0,.18,(600,2))
noise_levels=[1.,.55,.3,.16]
models={}
training_metrics=[]
for level,sigma in enumerate(noise_levels):
    clean=training[rng.integers(0,len(training),5000)]
    noise=rng.normal(0,sigma,clean.shape)
    noisy=clean+noise
    target=-noise/(sigma**2)
    model=make_pipeline(RBFSampler(gamma=1/(2*(sigma**2+.15)),n_components=320,random_state=SEED+level),Ridge(alpha=2.))
    model.fit(noisy,target)
    validation_clean=heldout
    validation_noise=rng.normal(0,sigma,validation_clean.shape)
    predicted=model.predict(validation_clean+validation_noise)
    weighted_mse=float(sigma**2*np.mean((predicted+validation_noise/sigma**2)**2))
    training_metrics.append({'noise_sigma':sigma,'heldout_weighted_denoising_score_mse':weighted_mse})
    models[sigma]=model
samples=rng.normal(0,2.5,(600,2))
trajectory=[]
for sigma in noise_levels:
    step_size=.025*sigma**2
    for step in range(70):
        score=models[sigma].predict(samples)
        norm=np.linalg.norm(score,axis=1,keepdims=True)
        score=score*np.minimum(1,20/np.maximum(norm,1e-12))
        samples+=step_size*score+np.sqrt(2*step_size)*rng.normal(size=samples.shape)
        samples=np.clip(samples,-5,5)
    nearest=cdist(samples,centers).min(axis=1)
    trajectory.append({'sigma':sigma,'mean_distance_to_nearest_mode':float(nearest.mean()),'fraction_within_point5':float((nearest<.5).mean())})


nearest_mode=cdist(samples,centers).argmin(axis=1)
mode_counts=np.bincount(nearest_mode,minlength=8)
energy_distance=2*cdist(samples,heldout).mean()-cdist(samples,samples).mean()-cdist(heldout,heldout).mean()
assert np.isfinite(samples).all()
save_table(pd.DataFrame(samples,columns=['x','y']),'generated_diffusion_samples')
save_table(pd.DataFrame(training_metrics),'score_matching_validation')
save_table(pd.DataFrame(trajectory),'annealed_langevin_progress')
save_table(pd.DataFrame({'mode':range(8),'generated_count':mode_counts}),'mode_coverage')
save_model({'score_models':models,'noise_levels':noise_levels,'sampling_bounds':[-5,5]},'score_diffusion_bundle')
fig,axes=plt.subplots(1,2,figsize=(10,5))
axes[0].scatter(*heldout.T,s=8,alpha=.5)
axes[0].set_title('Reference samples')
axes[1].scatter(*samples.T,s=8,alpha=.5)
axes[1].set_title('Learned score samples')
save_figure(fig,'score_based_generation')
finish({'generated_samples':len(samples),'energy_distance_statistic':float(energy_distance),'represented_modes':int((mode_counts>0).sum()),'sampler':'bounded_annealed_langevin','score_model':'random_fourier_features_ridge','exact_sampling_guaranteed':False},[pd.DataFrame(trajectory)])
