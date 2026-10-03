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
WORKFLOW = "145_directed_triad_motif_randomization"


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


from itertools import combinations, permutations
from collections import Counter

node_count=24
adjacency=(rng.random((node_count,node_count))<.085).astype(int)
np.fill_diagonal(adjacency,0)
for start in range(0,18,3):
    adjacency[start,start+1]=1
    adjacency[start+1,start+2]=1
    adjacency[start,start+2]=1
pairs=[(left,right) for left in range(3) for right in range(3) if left!=right]
canonical_cache={}
for bits in range(64):
    matrix=np.zeros((3,3),dtype=int)
    for bit,(left,right) in enumerate(pairs):
        matrix[left,right]=(bits>>bit)&1
    codes=[]
    for order in permutations(range(3)):
        codes.append(sum(int(matrix[order[left],order[right]])<<bit for bit,(left,right) in enumerate(pairs)))
    canonical_cache[bits]=min(codes)

def census(matrix):
    counts=Counter()
    for nodes in combinations(range(len(matrix)),3):
        bits=sum(int(matrix[nodes[left],nodes[right]])<<bit for bit,(left,right) in enumerate(pairs))
        counts[canonical_cache[bits]]+=1
    return counts

def rewire(matrix,attempts):
    output=matrix.copy()
    edges=list(zip(*np.nonzero(output)))
    for iteration in range(attempts):
        first,second=rng.choice(len(edges),2,replace=False)
        a,b=edges[first]
        c,d=edges[second]
        if len({a,b,c,d})<4 or output[a,d] or output[c,b]:
            continue
        output[a,b]=output[c,d]=0
        output[a,d]=output[c,b]=1
        edges[first]=(a,d)
        edges[second]=(c,b)
    assert np.array_equal(output.sum(axis=0),matrix.sum(axis=0))
    assert np.array_equal(output.sum(axis=1),matrix.sum(axis=1))
    return output



observed=census(adjacency)
null=[census(rewire(adjacency,2500)) for _ in range(30)]
rows=[]
for motif in sorted(set(canonical_cache.values())):
    distribution=np.array([sample[motif] for sample in null])
    rows.append({'canonical_motif_code':motif,'observed_count':observed[motif],'null_mean':float(distribution.mean()),'null_sd':float(distribution.std(ddof=1)),'enrichment_pvalue':(1+sum(distribution>=observed[motif]))/31})
output=pd.DataFrame(rows)
assert len(output)==16
assert output.observed_count.sum()==node_count*(node_count-1)*(node_count-2)//6
save_table(output,'directed_triad_census')
save_table(pd.DataFrame(adjacency),'directed_adjacency')
save_table(pd.DataFrame(null).fillna(0),'degree_preserving_null_counts')
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(output.canonical_motif_code.astype(str),output.observed_count,marker='o',label='observed')
ax.plot(output.canonical_motif_code.astype(str),output.null_mean,marker='x',label='rewired mean')
ax.legend()
save_figure(fig,'triad_motif_enrichment')
finish({'nodes':node_count,'edges':int(adjacency.sum()),'isomorphism_classes':len(output),'randomizations':len(null),'degree_sequences_preserved':True,'pvalues_are_unadjusted':True},[output])
