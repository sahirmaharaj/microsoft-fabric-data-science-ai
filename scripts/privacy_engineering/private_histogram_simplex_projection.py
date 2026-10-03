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
WORKFLOW = "161_private_histogram_simplex_projection"


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


import secrets

EPSILON=1.
PUBLIC_TOTAL=1800
DOMAIN=['segment_a','segment_b','segment_c','segment_d','segment_e','segment_f','segment_g','segment_h']
if EPSILON<=0:
    raise ValueError('positive_epsilon_required')
synthetic_private_labels=rng.choice(len(DOMAIN),PUBLIC_TOTAL,p=[.3,.2,.15,.12,.09,.06,.05,.03])
private_counts=np.bincount(synthetic_private_labels,minlength=len(DOMAIN))
l1_sensitivity=2.
scale=l1_sensitivity/EPSILON
secure_random=secrets.SystemRandom()

def secure_laplace(noise_scale):
    uniform=secure_random.random()-.5
    while abs(uniform)>=.5:
        uniform=secure_random.random()-.5
    return float(-noise_scale*np.sign(uniform)*np.log1p(-2*abs(uniform)))

def project_simplex(vector,total):
    ordered=np.sort(vector)[::-1]
    cumulative=np.cumsum(ordered)-total
    eligible=np.where(ordered-cumulative/np.arange(1,len(vector)+1)>0)[0]
    threshold=cumulative[eligible[-1]]/(eligible[-1]+1)
    return np.maximum(vector-threshold,0)

noise=np.array([secure_laplace(scale) for _ in DOMAIN])
released=private_counts+noise
consistent=project_simplex(released,PUBLIC_TOTAL)
public_interval_radius=-scale*np.log(.05)
output=pd.DataFrame({'category':DOMAIN,'noisy_count':released,'projected_count':consistent,'released_share':consistent/PUBLIC_TOTAL,'raw_noisy_lower_95':released-public_interval_radius,'raw_noisy_upper_95':released+public_interval_radius})
assert np.all(consistent>=0)
assert np.isclose(consistent.sum(),PUBLIC_TOTAL)
np.testing.assert_allclose(project_simplex(consistent,PUBLIC_TOTAL),consistent,atol=1e-9)
public_noise_diagnostics=[]


for epsilon in [.25,.5,1,2,4]:
    public_noise_diagnostics.append({'epsilon':epsilon,'expected_absolute_noise':l1_sensitivity/epsilon,'noise_standard_deviation':np.sqrt(2)*l1_sensitivity/epsilon,'marginal_95_radius':-l1_sensitivity/epsilon*np.log(.05)})
save_table(output,'private_histogram_release')
save_table(pd.DataFrame(public_noise_diagnostics),'public_noise_accuracy_tradeoff')
save_json({'epsilon':EPSILON,'delta':0,'adjacency':'replace_one_record','l1_sensitivity':l1_sensitivity,'public_total':PUBLIC_TOTAL,'public_domain':DOMAIN,'releases_in_this_run':1,'noise_source':'system_random','postprocessing':'euclidean_simplex_projection','intervals_are_marginal_not_simultaneous':True},'privacy_release_parameters')
fig,ax=plt.subplots(figsize=(10,4))
ax.bar(output.category,output.projected_count)
ax.tick_params(axis='x',rotation=35)
ax.set_ylabel('Privately released projected count')
save_figure(fig,'private_category_histogram')
del private_counts,synthetic_private_labels,noise
finish({'epsilon':EPSILON,'delta':0,'public_total':PUBLIC_TOTAL,'released_total':float(consistent.sum()),'categories':len(DOMAIN),'repeated_runs_compose_privacy_loss':True,'true_counts_not_exported':True},[output])
