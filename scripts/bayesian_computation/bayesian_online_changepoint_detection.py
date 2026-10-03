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
WORKFLOW = "133_bayesian_online_changepoint_detection"


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

means=np.r_[np.zeros(70),np.full(70,2.5),np.full(70,-1.5),np.ones(70)]
observed=means+rng.normal(0,.55,len(means))
noise_variance=.55**2
prior_mean=0.
prior_variance=9.
hazard=1/60
probabilities=np.array([1.])
posterior_mean=np.array([prior_mean])
posterior_variance=np.array([prior_variance])
runlength_matrix=np.zeros((len(observed),len(observed)+1))
rows=[]
for timepoint,value in enumerate(observed):
    log_predictive=norm.logpdf(value,posterior_mean,np.sqrt(posterior_variance+noise_variance))
    prior_predictive=norm.logpdf(value,prior_mean,np.sqrt(prior_variance+noise_variance))
    growth=np.log(np.maximum(probabilities,1e-300))+np.log1p(-hazard)+log_predictive
    change=np.log(hazard)+prior_predictive
    updated=np.r_[change,growth]
    updated-=logsumexp(updated)
    probabilities=np.exp(updated)
    new_variance=1/(1/posterior_variance+1/noise_variance)
    new_mean=new_variance*(posterior_mean/posterior_variance+value/noise_variance)
    reset_variance=1/(1/prior_variance+1/noise_variance)
    reset_mean=reset_variance*(prior_mean/prior_variance+value/noise_variance)
    posterior_mean=np.r_[reset_mean,new_mean]
    posterior_variance=np.r_[reset_variance,new_variance]
    runlength_matrix[timepoint,:len(probabilities)]=probabilities
    rows.append({'time':timepoint,'observation':value,'synthetic_mean':means[timepoint],'changepoint_probability':probabilities[0],'recent_change_probability':float(probabilities[:4].sum()),'most_likely_run_length':int(probabilities.argmax()),'filtered_mean':float(probabilities@posterior_mean)})
output=pd.DataFrame(rows)
output['alarm']=output.recent_change_probability>.65
known_changes=[70,140,210]
detection=[]
for change in known_changes:
    alarms=output.loc[output.time.between(change,change+15)&output.alarm,'time']
    detection.append({'true_change':change,'first_alarm':int(alarms.iloc[0]) if len(alarms) else -1,'delay':int(alarms.iloc[0]-change) if len(alarms) else -1})


assert np.allclose(runlength_matrix.sum(axis=1),1)
save_table(output,'online_change_probabilities')
save_table(pd.DataFrame(detection),'synthetic_change_detection_delays')
save_table(pd.DataFrame(runlength_matrix),'run_length_posterior')
fig,axes=plt.subplots(2,1,figsize=(11,7))
axes[0].plot(observed,alpha=.4)
axes[0].plot(output.filtered_mean)
axes[1].imshow(np.log10(runlength_matrix[:,:-1].T+1e-8),aspect='auto',origin='lower',cmap='magma')
axes[1].set(xlabel='Time',ylabel='Run length')
save_figure(fig,'online_run_length_inference')
finish({'hazard':hazard,'filtered_mean_rmse':float(np.sqrt(np.mean((output.filtered_mean-means)**2))),'alarms':int(output.alarm.sum()),'known_variance':noise_variance},[pd.DataFrame(detection)])
