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
WORKFLOW = "123_isomap_geodesic_manifold_recovery"


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


from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, trustworthiness
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

X,position=make_swiss_roll(n_samples=650,noise=.12,random_state=SEED)
scaler=StandardScaler().fit(X[:500])
scaled=scaler.transform(X)
training=scaled[:500]
test=scaled[500:]
models={}
evaluation=[]
for neighbors in [8,12,20]:
    model=Isomap(n_neighbors=neighbors,n_components=2,eigen_solver='arpack')
    embedding=model.fit_transform(training)
    heldout=model.transform(test)
    score=trustworthiness(test,heldout,n_neighbors=8)
    evaluation.append({'neighbors':neighbors,'heldout_trustworthiness':score,'training_reconstruction_error':model.reconstruction_error()})
    models[neighbors]=(model,embedding,heldout)
model,embedding,heldout=models[12]
pca=PCA(n_components=2).fit(training)
pca_test=pca.transform(test)
latent_coordinates=np.column_stack([position[500:],X[500:,1]])
latent_coordinates=StandardScaler().fit_transform(latent_coordinates)
reference_distance=pairwise_distances(latent_coordinates)
upper=np.triu_indices(len(test),1)
geodesic_rank=float(spearmanr(reference_distance[upper],pairwise_distances(heldout)[upper]).statistic)
pca_rank=float(spearmanr(reference_distance[upper],pairwise_distances(pca_test)[upper]).statistic)
output=pd.DataFrame({'sample':np.arange(500,650),'manifold_0':heldout[:,0],'manifold_1':heldout[:,1],'roll_position':position[500:],'height':X[500:,1]})
save_table(output,'out_of_sample_manifold_coordinates')
save_table(pd.DataFrame(evaluation),'neighborhood_sensitivity')
save_table(pd.DataFrame({'method':['isomap','pca'],'latent_distance_spearman':[geodesic_rank,pca_rank],'trustworthiness':[trustworthiness(test,heldout,n_neighbors=8),trustworthiness(test,pca_test,n_neighbors=8)]}),'geometry_comparison')
save_model({'scaler':scaler,'isomap':model,'pca':pca},'manifold_transform')


assert heldout.shape==(150,2)
assert np.isfinite(heldout).all()
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].scatter(heldout[:,0],heldout[:,1],c=position[500:],s=14)
axes[0].set_title('Isomap')
axes[1].scatter(pca_test[:,0],pca_test[:,1],c=position[500:],s=14)
axes[1].set_title('PCA')
save_figure(fig,'geodesic_vs_linear_embedding')
finish({'isomap_latent_distance_correlation':geodesic_rank,'pca_latent_distance_correlation':pca_rank,'neighbors_fixed_before_test_evaluation':12},[pd.DataFrame(evaluation)])
