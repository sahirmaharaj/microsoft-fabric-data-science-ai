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
WORKFLOW = "127_metropolis_hastings_multi_chain_diagnostics"


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

observations=rng.normal(2.5,1.2,180)

def log_posterior(theta):
    mean,log_scale=theta
    if abs(mean)>50 or abs(log_scale)>10:
        return -np.inf
    variance=np.exp(2*log_scale)
    return float(-len(observations)*log_scale-.5*np.sum((observations-mean)**2)/variance-.5*(mean/5)**2-.5*log_scale**2)

chains=[]
acceptance=[]
for chain_id in range(4):
    random=np.random.default_rng(SEED+chain_id)
    theta=np.array([random.normal(0,2),random.normal(0,.3)])
    current=log_posterior(theta)
    draws=[]
    accepted=0
    for iteration in range(6000):
        proposal=theta+random.normal(size=2)*[.15,.09]
        value=log_posterior(proposal)
        if np.log(random.random())<value-current:
            theta,current=proposal,value
            accepted+=1
        if iteration>=1500:
            draws.append(theta.copy())
    chains.append(draws)
    acceptance.append({'chain':chain_id,'acceptance_rate':accepted/6000})
chains=np.asarray(chains)
within=np.var(chains,axis=1,ddof=1).mean(axis=0)
between=chains.shape[1]*np.var(chains.mean(axis=1),axis=0,ddof=1)
variance=(chains.shape[1]-1)/chains.shape[1]*within+between/chains.shape[1]
rhat=np.sqrt(variance/within)
pooled=chains.reshape(-1,2)


summary=pd.DataFrame({'parameter':['mean','log_scale'],'mean':pooled.mean(axis=0),'lower':np.quantile(pooled,.025,axis=0),'upper':np.quantile(pooled,.975,axis=0),'classical_rhat':rhat})
trace=pd.DataFrame(chains.reshape(-1,2),columns=['mean','log_scale'])
trace['chain']=np.repeat(np.arange(4),chains.shape[1])
trace['draw']=np.tile(np.arange(chains.shape[1]),4)
assert np.isfinite(chains).all()
save_table(trace,'posterior_chains')
save_table(summary,'posterior_summary')
save_table(pd.DataFrame(acceptance),'acceptance_rates')
fig,axes=plt.subplots(2,1,figsize=(10,6))
for chain_id in range(4):
    axes[0].plot(chains[chain_id,:,0],alpha=.4)
    axes[1].plot(np.exp(chains[chain_id,:,1]),alpha=.4)
axes[0].set_ylabel('Mean')
axes[1].set_ylabel('Standard deviation')
save_figure(fig,'metropolis_traces')
finish({'posterior_mean':float(pooled[:,0].mean()),'posterior_scale':float(np.exp(pooled[:,1]).mean()),'maximum_classical_rhat':float(rhat.max()),'rhat_is_not_rank_normalized':True},[summary])
