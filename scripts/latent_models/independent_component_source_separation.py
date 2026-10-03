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
WORKFLOW = "119_independent_component_source_separation"


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


from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
from scipy.signal import sawtooth, square

samples=2400
time_axis=np.linspace(0,8,samples)
sources=np.column_stack([np.sin(2*np.pi*1.7*time_axis),sawtooth(2*np.pi*.9*time_axis),square(2*np.pi*.6*time_axis)])
sources=(sources-sources.mean(axis=0))/sources.std(axis=0)
mixing=np.array([[1,.5,.3],[.4,1,.7],[.8,.2,1],[.2,.6,.4]])
observed=sources@mixing.T+rng.normal(0,.03,(samples,4))
train=np.arange(1700)
test=np.arange(1700,samples)
model=FastICA(n_components=3,whiten='unit-variance',max_iter=1500,tol=1e-6,random_state=SEED)
training_components=model.fit_transform(observed[train])
components=model.transform(observed[test])
training_correlation=np.corrcoef(training_components.T,sources[train].T)[:3,3:]
rows,columns=linear_sum_assignment(-abs(training_correlation))
ordered=np.zeros_like(components)
assignment=[]
for component,source in zip(rows,columns):
    sign=np.sign(training_correlation[component,source])
    ordered[:,source]=sign*components[:,component]
    assignment.append({'learned_component':int(component),'source':int(source),'orientation':float(sign),'training_absolute_correlation':float(abs(training_correlation[component,source]))})
metrics=[]
for source in range(3):
    truth=sources[test,source]
    prediction=ordered[:,source]
    metrics.append({'source':source,'heldout_correlation':float(np.corrcoef(truth,prediction)[0,1]),'heldout_rmse':float(np.sqrt(np.mean((truth-prediction)**2)))})
reconstructed=model.inverse_transform(components)
residual=observed[test]-reconstructed
independence=np.corrcoef(components.T)
save_table(pd.DataFrame(metrics),'source_recovery_metrics')
save_table(pd.DataFrame(assignment),'training_only_component_assignment')
save_table(pd.DataFrame(ordered,columns=['source_0','source_1','source_2']),'separated_test_sources')
save_table(pd.DataFrame(independence),'component_correlation')


save_model(model,'ica_unmixing_model')
assert np.isfinite(reconstructed).all()
assert len(set(columns))==3
fig,axes=plt.subplots(3,1,figsize=(11,7))
for index,ax in enumerate(axes):
    ax.plot(time_axis[test],sources[test,index],label='source')
    ax.plot(time_axis[test],ordered[:,index],alpha=.7,label='recovered')
axes[0].legend()
save_figure(fig,'separated_waveforms')
finish({'reconstruction_rmse':float(np.sqrt(np.mean(residual**2))),'minimum_source_correlation':min(row['heldout_correlation'] for row in metrics),'component_matching_uses_test_labels':False},[pd.DataFrame(metrics)])
