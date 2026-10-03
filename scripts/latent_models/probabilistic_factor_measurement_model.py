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
WORKFLOW = "122_probabilistic_factor_measurement_model"


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


from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from scipy.linalg import orthogonal_procrustes

samples,features,factors=1000,12,3
loadings=np.zeros((features,factors))
for feature in range(features):
    loadings[feature,feature//4]=rng.uniform(.7,1.3)
    loadings[feature]+=rng.normal(0,.08,factors)
latent=rng.normal(size=(samples,factors))
noise_variance=np.linspace(.15,.6,features)
observed=latent@loadings.T+rng.normal(size=(samples,features))*np.sqrt(noise_variance)
train,test=train_test_split(np.arange(samples),test_size=.3,random_state=SEED)
selection=[]
models={}
for dimension in range(1,7):
    model=FactorAnalysis(n_components=dimension,rotation='varimax',random_state=SEED,max_iter=1000).fit(observed[train])
    parameters=features*dimension+features-dimension*(dimension-1)/2
    bic=-2*model.score(observed[train])*len(train)+parameters*np.log(len(train))
    selection.append({'factors':dimension,'bic':float(bic),'test_loglikelihood':model.score(observed[test])})
    models[dimension]=model
selected=min(selection,key=lambda row:row['bic'])['factors']
model=models[selected]
scores=model.transform(observed[test])
reconstruction=scores@model.components_+model.mean_
communalities=np.sum(model.components_**2,axis=0)
total_variance=communalities+model.noise_variance_
measurement=pd.DataFrame({'variable':np.arange(features),'communality':communalities,'uniqueness':model.noise_variance_,'shared_variance_fraction':communalities/total_variance,'true_noise_variance':noise_variance})
covariance=model.get_covariance()
residual_covariance=np.cov(observed[test],rowvar=False)-covariance
assert np.linalg.eigvalsh(covariance).min()>0
assert np.all(model.noise_variance_>0)
save_table(pd.DataFrame(selection),'factor_dimension_selection')
save_table(measurement,'measurement_reliability')
save_table(pd.DataFrame(model.components_.T),'rotated_loadings')


save_table(pd.DataFrame(scores),'heldout_factor_scores')
save_table(pd.DataFrame(residual_covariance),'covariance_residuals')
save_model(model,'factor_measurement_model')
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].imshow(model.components_.T,aspect='auto',cmap='coolwarm')
axes[0].set(xlabel='Factor',ylabel='Measurement')
axes[1].bar(measurement.variable,measurement.shared_variance_fraction)
axes[1].set(ylabel='Shared variance fraction')
save_figure(fig,'factor_measurement_diagnostics')
finish({'selected_factors':selected,'heldout_loglikelihood':model.score(observed[test]),'covariance_residual_frobenius':float(np.linalg.norm(residual_covariance)),'posterior_mean_reconstruction_mse':float(np.mean((reconstruction-observed[test])**2))},[measurement])
