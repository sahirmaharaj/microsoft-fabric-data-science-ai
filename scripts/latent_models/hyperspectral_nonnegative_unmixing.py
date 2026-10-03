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
WORKFLOW = "120_hyperspectral_nonnegative_unmixing"


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


from sklearn.decomposition import NMF
from scipy.optimize import nnls, linear_sum_assignment

bands=60
wavelength=np.linspace(400,1000,bands)
endmembers=np.stack([.1+np.exp(-((wavelength-center)/width)**2) for center,width in [(480,55),(670,80),(870,65)]])
abundance=rng.dirichlet([.4,.4,.4],900)
spectra=np.maximum(abundance@endmembers+rng.normal(0,.015,(900,bands)),0)
train=spectra[:700]
test=spectra[700:]
model=NMF(n_components=3,init='nndsvda',max_iter=1200,tol=1e-5,random_state=SEED)
train_weights=model.fit_transform(train)
learned=model.components_.copy()
learned/=learned.max(axis=1,keepdims=True)
truth=endmembers/endmembers.max(axis=1,keepdims=True)
cosine=(learned@truth.T)/(np.linalg.norm(learned,axis=1)[:,None]*np.linalg.norm(truth,axis=1)[None,:])
rows,columns=linear_sum_assignment(-cosine)
ordered=np.zeros_like(learned)
for row,column in zip(rows,columns):
    ordered[column]=learned[row]
weights=np.stack([nnls(ordered.T,spectrum)[0] for spectrum in test])
reconstruction=weights@ordered
normalized=weights/np.maximum(weights.sum(axis=1,keepdims=True),1e-12)
angles=np.arccos(np.clip(np.sum(reconstruction*test,axis=1)/(np.linalg.norm(reconstruction,axis=1)*np.linalg.norm(test,axis=1)),-1,1))
metrics=pd.DataFrame({'pixel':np.arange(len(test)),'spectral_angle_radians':angles,'reconstruction_rmse':np.sqrt(np.mean((reconstruction-test)**2,axis=1)),'abundance_mae':np.mean(abs(normalized-abundance[700:]),axis=1)})
endmember_table=pd.DataFrame({'wavelength':wavelength})
for material in range(3):
    endmember_table[f'learned_{material}']=ordered[material]
    endmember_table[f'true_{material}']=truth[material]
assert np.all(weights>=0)
assert np.allclose(normalized.sum(axis=1),1)
save_table(metrics,'heldout_pixel_errors')
save_table(endmember_table,'spectral_endmembers')
save_table(pd.DataFrame(normalized,columns=['material_0','material_1','material_2']),'relative_abundances')
save_model({'nmf':model,'normalized_endmembers':ordered},'spectral_unmixing_bundle')


fig,axes=plt.subplots(1,2,figsize=(11,4))
for material in range(3):
    axes[0].plot(wavelength,ordered[material],label=f'material_{material}')
axes[0].legend()
axes[1].scatter(abundance[700:,0],normalized[:,0],s=10)
axes[1].set(xlabel='True abundance',ylabel='Estimated relative abundance')
save_figure(fig,'hyperspectral_unmixing')
finish({'mean_spectral_angle':float(angles.mean()),'mean_abundance_mae':float(metrics.abundance_mae.mean()),'scale_ambiguity_resolved_by_peak_normalization':True},[metrics])
