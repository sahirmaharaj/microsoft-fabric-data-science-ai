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
WORKFLOW = "148_personalized_pagerank_topic_retrieval"


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


from scipy.sparse import csr_matrix
from scipy.linalg import solve

nodes=90
topics=np.repeat(np.arange(3),30)
adjacency=np.zeros((nodes,nodes))
for left in range(nodes):
    for right in range(nodes):
        if left!=right and rng.random()<( .11 if topics[left]==topics[right] else .012):
            adjacency[left,right]=1
adjacency[-3:]=0
out_degree=adjacency.sum(axis=1)
dangling=out_degree==0
transition=np.divide(adjacency,out_degree[:,None],out=np.zeros_like(adjacency),where=out_degree[:,None]>0)
sparse_transition=csr_matrix(transition)
alpha=.85

def pagerank(personalization):
    rank=personalization.copy()
    history=[]
    for iteration in range(1000):
        updated=alpha*(sparse_transition.T@rank+rank[dangling].sum()*personalization)+(1-alpha)*personalization
        residual=float(abs(updated-rank).sum())
        rank=updated
        history.append({'iteration':iteration,'l1_residual':residual})
        if residual<1e-12:
            break
    repaired=transition+dangling[:,None]*personalization[None,:]
    exact=solve(np.eye(nodes)-alpha*repaired.T,(1-alpha)*personalization)
    np.testing.assert_allclose(rank,exact,atol=1e-10)
    return rank,history

uniform=np.full(nodes,1/nodes)
global_rank,_=pagerank(uniform)
rankings=[]


traces=[]
for topic in range(3):
    seeds=np.where(topics==topic)[0][:3]
    personalization=np.zeros(nodes)
    personalization[seeds]=1/len(seeds)
    rank,history=pagerank(personalization)
    for row in history:
        traces.append({'query_topic':topic,**row})
    order=np.argsort(-rank)
    for position,node in enumerate(order):
        rankings.append({'query_topic':topic,'rank':position+1,'node':int(node),'node_topic':int(topics[node]),'personalized_score':rank[node],'global_score':global_rank[node],'seed':node in seeds})
output=pd.DataFrame(rankings)
output['topic_match']=output.node_topic.eq(output.query_topic)
summary=output.loc[output['rank'].le(10)].groupby('query_topic').agg(precision_at_10=('topic_match','mean'),top10_probability_mass=('personalized_score','sum')).reset_index()
assert np.isclose(global_rank.sum(),1)
save_table(output,'topic_personalized_rankings')
save_table(pd.DataFrame(traces),'pagerank_convergence')
save_table(summary,'topic_retrieval_metrics')
fig,ax=plt.subplots(figsize=(9,4))
for topic in range(3):
    frame=output.loc[output.query_topic.eq(topic)].sort_values('node')
    ax.plot(frame.node,frame.personalized_score,label=f'topic_{topic}')
ax.legend()
save_figure(fig,'topic_pagerank_profiles')
finish({'nodes':nodes,'dangling_nodes':int(dangling.sum()),'linear_system_agreement_verified':True,'mean_precision_at_10':float(summary.precision_at_10.mean())},[summary])
