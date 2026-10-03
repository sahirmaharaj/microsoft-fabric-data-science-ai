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
WORKFLOW = "170_haar_wavelet_threshold_denoising"


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


length=2048
time_axis=np.linspace(0,1,length,endpoint=False)
clean=np.sin(2*np.pi*4*time_axis)+.7*(time_axis>.3)-.8*(time_axis>.65)+1.2*np.exp(-((time_axis-.48)/.035)**2)
noisy=clean+rng.normal(0,.25,length)

def decompose(signal):
    approximation=signal.copy()
    details=[]
    if len(signal)&(len(signal)-1):
        raise ValueError('power_of_two_length_required')
    while len(approximation)>1:
        left=approximation[0::2]
        right=approximation[1::2]
        details.append((left-right)/np.sqrt(2))
        approximation=(left+right)/np.sqrt(2)
    return approximation,details

def reconstruct(approximation,details):
    output=approximation.copy()
    for detail in reversed(details):
        expanded=np.empty(len(detail)*2)
        expanded[0::2]=(output+detail)/np.sqrt(2)
        expanded[1::2]=(output-detail)/np.sqrt(2)
        output=expanded
    return output

approximation,details=decompose(noisy)
np.testing.assert_allclose(reconstruct(approximation,details),noisy,atol=1e-12)
assert np.isclose(np.sum(noisy**2),np.sum(approximation**2)+sum(np.sum(detail**2) for detail in details))
noise_estimate=float(np.median(abs(details[0]-np.median(details[0])))/.67448975)
threshold=noise_estimate*np.sqrt(2*np.log(length))
shrunken=[np.sign(detail)*np.maximum(abs(detail)-threshold,0) for detail in details]
denoised=reconstruct(approximation,shrunken)
levels=[]
for index,(before,after) in enumerate(zip(details,shrunken)):
    levels.append({'level':index+1,'coefficients':len(before),'original_energy':float(before@before),'retained_energy':float(after@after),'retained_nonzero':int(np.count_nonzero(after))})


sensitivity=[]
for multiplier in [.4,.7,1,1.3,1.6]:
    filtered=[np.sign(detail)*np.maximum(abs(detail)-multiplier*threshold,0) for detail in details]
    recovered=reconstruct(approximation,filtered)
    sensitivity.append({'threshold_multiplier':multiplier,'synthetic_truth_rmse':float(np.sqrt(np.mean((recovered-clean)**2)))})
save_table(pd.DataFrame({'time':time_axis,'clean':clean,'noisy':noisy,'denoised':denoised}),'wavelet_denoised_signal')
save_table(pd.DataFrame(levels),'multiresolution_energy')
save_table(pd.DataFrame(sensitivity),'synthetic_threshold_sensitivity')
save_json({'wavelet':'orthonormal_haar','noise_sigma_estimate':noise_estimate,'threshold':threshold,'rule':'universal_soft_threshold'},'wavelet_parameters')
fig,ax=plt.subplots(figsize=(11,4))
ax.plot(time_axis,noisy,alpha=.25,label='noisy')
ax.plot(time_axis,clean,label='clean')
ax.plot(time_axis,denoised,label='denoised')
ax.legend()
save_figure(fig,'wavelet_signal_recovery')
finish({'noise_sigma_estimate':noise_estimate,'noisy_rmse':float(np.sqrt(np.mean((noisy-clean)**2))),'denoised_rmse':float(np.sqrt(np.mean((denoised-clean)**2))),'perfect_reconstruction_and_energy_verified':True},[pd.DataFrame(levels)])
