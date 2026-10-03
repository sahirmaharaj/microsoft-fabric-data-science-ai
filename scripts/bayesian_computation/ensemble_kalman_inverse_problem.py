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
WORKFLOW = "131_ensemble_kalman_inverse_problem"


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


from scipy.linalg import solve

parameter_count=5
measurement_count=14
ensemble_size=2500
operator=rng.normal(size=(measurement_count,parameter_count))
truth=np.array([1.2,-.8,.4,1.6,-.2])
prior_mean=np.zeros(parameter_count)
prior_cov=np.diag([1.,1.5,.8,2.,1.])
noise_cov=np.diag(np.linspace(.2,.5,measurement_count)**2)
observation=operator@truth+rng.multivariate_normal(np.zeros(measurement_count),noise_cov)
prior_ensemble=rng.multivariate_normal(prior_mean,prior_cov,ensemble_size)
predicted=prior_ensemble@operator.T
centered_parameters=prior_ensemble-prior_ensemble.mean(axis=0)
centered_prediction=predicted-predicted.mean(axis=0)
cross_cov=centered_parameters.T@centered_prediction/(ensemble_size-1)
prediction_cov=centered_prediction.T@centered_prediction/(ensemble_size-1)
gain=solve(prediction_cov+noise_cov,cross_cov.T,assume_a='pos').T
perturbed_observations=observation+rng.multivariate_normal(np.zeros(measurement_count),noise_cov,ensemble_size)
posterior=prior_ensemble+(perturbed_observations-predicted)@gain.T
exact_gain=solve(operator@prior_cov@operator.T+noise_cov,operator@prior_cov,assume_a='pos').T
exact_mean=prior_mean+exact_gain@(observation-operator@prior_mean)
exact_cov=prior_cov-exact_gain@operator@prior_cov
estimated_cov=np.cov(posterior,rowvar=False)
comparison=pd.DataFrame({'parameter':np.arange(parameter_count),'truth':truth,'ensemble_mean':posterior.mean(axis=0),'analytic_mean':exact_mean,'ensemble_sd':posterior.std(axis=0,ddof=1),'analytic_sd':np.sqrt(np.diag(exact_cov)),'lower':np.quantile(posterior,.025,axis=0),'upper':np.quantile(posterior,.975,axis=0)})
posterior_measurements=posterior@operator.T
measurement=pd.DataFrame({'sensor':range(measurement_count),'observed':observation,'posterior_mean':posterior_measurements.mean(axis=0),'lower':np.quantile(posterior_measurements,.025,axis=0),'upper':np.quantile(posterior_measurements,.975,axis=0)})
assert np.linalg.eigvalsh(estimated_cov).min()>0
assert np.isfinite(posterior).all()
save_table(comparison,'ensemble_vs_analytic_posterior')
save_table(measurement,'measurement_reconstruction')
save_table(pd.DataFrame(posterior),'posterior_parameter_ensemble')
save_table(pd.DataFrame(estimated_cov),'ensemble_posterior_covariance')
save_json({'operator':operator,'noise_covariance':noise_cov,'prior_covariance':prior_cov,'analytic_posterior_mean':exact_mean,'analytic_posterior_covariance':exact_cov},'inverse_problem_definition')
fig,ax=plt.subplots(figsize=(8,5))


ax.errorbar(comparison.parameter,comparison.ensemble_mean,yerr=1.96*comparison.ensemble_sd,fmt='o',label='ensemble')
ax.scatter(comparison.parameter,truth,marker='x',label='truth')
ax.legend()
save_figure(fig,'ensemble_parameter_inference')
finish({'posterior_mean_l2_error_to_analytic':float(np.linalg.norm(posterior.mean(axis=0)-exact_mean)),'posterior_covariance_relative_error':float(np.linalg.norm(estimated_cov-exact_cov)/np.linalg.norm(exact_cov)),'ensemble_members':ensemble_size},[comparison])
