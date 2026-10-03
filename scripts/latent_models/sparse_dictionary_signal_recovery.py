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
WORKFLOW = "118_sparse_dictionary_signal_recovery"


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


from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode
from scipy.ndimage import gaussian_filter1d

length=48
position=np.linspace(0,1,length)
true_dictionary=np.stack([np.sin(2*np.pi*f*position+phase)*np.exp(-((position-center)/.3)**2) for f,phase,center in [(2,0,.3),(3,.5,.5),(4,1,.7),(1,0,.5),(5,.2,.4),(2,1,.7)]])
true_dictionary/=np.linalg.norm(true_dictionary,axis=1,keepdims=True)
coefficients=np.zeros((900,6))
for row in coefficients:
    selected=rng.choice(6,2,replace=False)
    row[selected]=rng.normal(0,2,2)
clean=coefficients@true_dictionary
noisy=clean+rng.normal(0,.12,clean.shape)
model=MiniBatchDictionaryLearning(n_components=12,alpha=.12,max_iter=220,batch_size=80,transform_algorithm='omp',transform_n_nonzero_coefs=3,random_state=SEED)
model.fit(noisy[:700])
encoded=model.transform(noisy[700:])
recovered=encoded@model.components_
smoothed=gaussian_filter1d(noisy[700:],sigma=1.2,axis=1)
comparison=pd.DataFrame({'sample':np.arange(200),'noisy_mse':np.mean((noisy[700:]-clean[700:])**2,axis=1),'dictionary_mse':np.mean((recovered-clean[700:])**2,axis=1),'smoothing_mse':np.mean((smoothed-clean[700:])**2,axis=1),'active_atoms':np.count_nonzero(encoded,axis=1)})
sparsity=[]
for atoms in [1,2,3,4,6]:
    codes=sparse_encode(noisy[700:],model.components_,algorithm='omp',n_nonzero_coefs=atoms)
    estimate=codes@model.components_
    sparsity.append({'atoms':atoms,'heldout_mse':float(np.mean((estimate-clean[700:])**2)),'reconstruction_mse':float(np.mean((estimate-noisy[700:])**2))})
correspondence=abs(model.components_@true_dictionary.T)
save_table(comparison,'signal_recovery_errors')
save_table(pd.DataFrame(sparsity),'sparsity_sensitivity')
save_table(pd.DataFrame(model.components_),'learned_dictionary')
save_table(pd.DataFrame(correspondence),'atom_correspondence')
save_model(model,'sparse_dictionary')
assert np.isfinite(encoded).all()
assert np.max(np.count_nonzero(encoded,axis=1))<=3
fig,axes=plt.subplots(2,2,figsize=(10,6))
for index,ax in enumerate(axes.ravel()):
    ax.plot(clean[700+index],label='clean')
    ax.plot(noisy[700+index],alpha=.35,label='noisy')
    ax.plot(recovered[index],label='recovered')


axes[0,0].legend()
save_figure(fig,'sparse_signal_recovery')
finish({'heldout_dictionary_mse':float(comparison.dictionary_mse.mean()),'heldout_noisy_mse':float(comparison.noisy_mse.mean()),'active_atoms_mean':float(comparison.active_atoms.mean())},[comparison])
