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
WORKFLOW = "160_greenwald_khanna_quantile_summary"


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


from bisect import bisect_right
import math

class QuantileSummary:
    def __init__(self,epsilon=.01):
        if not 0<epsilon<.5:
            raise ValueError('epsilon_out_of_range')
        self.epsilon=epsilon
        self.count=0
        self.summary=[]
    def insert(self,value):
        if not np.isfinite(value):
            raise ValueError('finite_value_required')
        self.count+=1
        index=bisect_right([entry[0] for entry in self.summary],value)
        delta=0 if index in [0,len(self.summary)] else max(math.floor(2*self.epsilon*self.count)-1,0)
        self.summary.insert(index,[float(value),1,delta])
        if self.count%max(1,math.floor(1/(2*self.epsilon)))==0:
            self.compress()
    def compress(self):
        bound=math.floor(2*self.epsilon*self.count)
        for index in range(len(self.summary)-2,0,-1):
            current=self.summary[index]
            following=self.summary[index+1]
            if current[1]+following[1]+following[2]<=bound:
                following[1]+=current[1]
                self.summary.pop(index)
    def quantile(self,probability):
        if not self.summary or not 0<=probability<=1:
            raise ValueError('invalid_quantile_query')
        if probability==0:
            return self.summary[0][0]
        if probability==1:
            return self.summary[-1][0]
        rank=probability*self.count
        allowed=self.epsilon*self.count
        cumulative=0
        previous=self.summary[0][0]
        for value,gap,delta in self.summary:
            cumulative+=gap
            if cumulative+delta>rank+allowed:
                return previous
            previous=value
        return self.summary[-1][0]



stream=rng.lognormal(2,1.2,25000)
summary=QuantileSummary(.01)
history=[]
for index,value in enumerate(stream,1):
    summary.insert(value)
    if index%1000==0:
        history.append({'observations':index,'retained_tuples':len(summary.summary),'median':summary.quantile(.5)})
summary.compress()
ordered=np.sort(stream)
rows=[]
for probability in [.01,.05,.25,.5,.75,.9,.95,.99]:
    estimate=summary.quantile(probability)
    rank=int(np.searchsorted(ordered,estimate,side='right'))
    error=abs(rank-probability*len(stream))
    assert error<=summary.epsilon*len(stream)+1
    rows.append({'quantile':probability,'estimate':estimate,'exact_value':float(np.quantile(stream,probability)),'absolute_rank_error':error,'allowed_rank_error':summary.epsilon*len(stream)})
assert sum(entry[1] for entry in summary.summary)==len(stream)
save_table(pd.DataFrame(rows),'quantile_rank_accuracy')
save_table(pd.DataFrame(history),'streaming_memory_profile')
save_table(pd.DataFrame(summary.summary,columns=['value','gap','delta']),'quantile_summary_state')
fig,ax=plt.subplots(figsize=(8,4))
ax.plot([row['observations'] for row in history],[row['retained_tuples'] for row in history])
ax.set(xlabel='Stream observations',ylabel='Retained tuples')
save_figure(fig,'bounded_quantile_summary_growth')
finish({'observations':len(stream),'retained_tuples':len(summary.summary),'epsilon':summary.epsilon,'all_queried_rank_bounds_verified':True},[pd.DataFrame(rows)])
