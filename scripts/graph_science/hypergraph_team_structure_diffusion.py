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
WORKFLOW = "146_hypergraph_team_structure_diffusion"


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


from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

people=75
teams=100
groups=np.repeat(np.arange(3),25)
incidence=np.zeros((people,teams))
for team in range(teams):
    group=int(rng.integers(0,3))
    members=rng.choice(np.where(groups==group)[0],int(rng.integers(3,9)),replace=False)
    incidence[members,team]=1
    if rng.random()<.3:
        incidence[int(rng.integers(people)),team]=1
for person in range(people):
    if incidence[person].sum()==0:
        incidence[person,int(rng.integers(teams))]=1
team_degree=incidence.sum(axis=0)
team_weight=1/np.sqrt(team_degree)
vertex_degree=incidence@team_weight
normalizer=1/np.sqrt(vertex_degree)
affinity=(incidence*(team_weight/team_degree))@incidence.T
normalized=normalizer[:,None]*affinity*normalizer[None,:]
laplacian=np.eye(people)-normalized
eigenvalues,eigenvectors=eigh(laplacian)
embedding=eigenvectors[:,1:4]
clusters=KMeans(n_clusters=3,n_init=15,random_state=SEED).fit_predict(embedding)
clean=np.choose(groups,[-1.,.4,1.5])
noisy=clean+rng.normal(0,.6,people)
normalized_signal=np.sqrt(vertex_degree)*noisy
filtered=eigenvectors@(np.exp(-2*eigenvalues)*(eigenvectors.T@normalized_signal))
recovered=filtered/np.sqrt(vertex_degree)
energy_before=float(normalized_signal@laplacian@normalized_signal)
energy_after=float(filtered@laplacian@filtered)
assert energy_after<=energy_before+1e-9


assert np.max(abs(laplacian-laplacian.T))<1e-10
assert eigenvalues.min()>-1e-8
vertices=pd.DataFrame({'person':range(people),'group':groups,'cluster':clusters,'weighted_degree':vertex_degree,'noisy_signal':noisy,'diffused_signal':recovered,'true_signal':clean})
events=pd.DataFrame([{'person':int(person),'team':int(team),'team_size':int(team_degree[team])} for person,team in zip(*np.nonzero(incidence))])
save_table(vertices,'hypergraph_vertex_features')
save_table(events,'team_incidence_events')
save_table(pd.DataFrame({'eigenvalue':eigenvalues}),'hypergraph_spectrum')
save_json({'weighted_team_size':team_degree,'team_weights':team_weight,'diffusion_time':2.},'hypergraph_configuration')
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].scatter(embedding[:,0],embedding[:,1],c=clusters)
axes[1].plot(clean,label='clean')
axes[1].plot(recovered,label='diffused')
axes[1].legend()
save_figure(fig,'hypergraph_structure_and_diffusion')
finish({'cluster_ari':adjusted_rand_score(groups,clusters),'energy_before':energy_before,'energy_after':energy_after,'noisy_rmse':float(np.sqrt(np.mean((noisy-clean)**2))),'diffused_rmse':float(np.sqrt(np.mean((recovered-clean)**2)))},[vertices])
