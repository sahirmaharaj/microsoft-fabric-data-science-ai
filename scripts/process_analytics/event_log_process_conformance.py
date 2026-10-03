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
WORKFLOW = "142_event_log_process_conformance"


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


from collections import Counter, defaultdict

records=[]
allowed={('start','submit'),('submit','review'),('review','approve'),('review','revise'),('revise','review'),('approve','pay'),('pay','end'),('review','reject'),('reject','end')}
for case in range(260):
    path=['start','submit','review']
    if rng.random()<.25:
        path+=['revise','review']
    if rng.random()<.15:
        path+=['reject','end']
    else:
        path+=['approve','pay','end']
    if case>=210 and rng.random()<.4:
        path=[activity for activity in path if activity!='approve']
    timestamp=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(hours=case*3)
    for sequence,activity in enumerate(path):
        timestamp+=pd.Timedelta(minutes=float(rng.gamma(2,35)))
        records.append({'case':case,'sequence':sequence,'activity':activity,'timestamp':timestamp})
events=pd.DataFrame(records)
variants=Counter()
transitions=defaultdict(list)
case_rows=[]
violations=[]
for case,group in events.groupby('case'):
    group=group.sort_values('sequence')
    path=group.activity.tolist()
    variants[tuple(path)]+=1
    invalid=0
    for index,(left,right) in enumerate(zip(path,path[1:])):
        elapsed=(group.timestamp.iloc[index+1]-group.timestamp.iloc[index]).total_seconds()/60
        transitions[(left,right)].append(elapsed)
        if (left,right) not in allowed:
            invalid+=1
            violations.append({'case':case,'from_activity':left,'to_activity':right,'transition_index':index})
    case_rows.append({'case':case,'duration_minutes':(group.timestamp.iloc[-1]-group.timestamp.iloc[0]).total_seconds()/60,'events':len(path),'invalid_transitions':invalid,'transition_fitness':1-invalid/(len(path)-1),'cohort':'later' if case>=210 else 'earlier'})


transition_table=pd.DataFrame([{'from_activity':left,'to_activity':right,'count':len(durations),'mean_minutes':np.mean(durations),'p90_minutes':np.quantile(durations,.9),'allowed':(left,right) in allowed} for (left,right),durations in transitions.items()])
variant_table=pd.DataFrame([{'variant':json.dumps(path),'cases':count,'share':count/260} for path,count in variants.most_common()])
cases=pd.DataFrame(case_rows)
cohorts=cases.groupby('cohort').agg(cases=('case','size'),mean_fitness=('transition_fitness','mean'),nonconforming_cases=('invalid_transitions',lambda x:int((x>0).sum())),median_duration=('duration_minutes','median')).reset_index()
assert variant_table.cases.sum()==260
assert transition_table['count'].sum()==len(events)-260
save_table(events,'process_event_log')
save_table(transition_table,'directly_follows_performance')
save_table(variant_table,'process_variants')
save_table(pd.DataFrame(violations),'conformance_violations')
save_table(cohorts,'cohort_conformance')
fig,ax=plt.subplots(figsize=(9,4))
ax.hist(cases.duration_minutes,bins=25)
ax.set(xlabel='Case duration in minutes',ylabel='Cases')
save_figure(fig,'process_duration_distribution')
finish({'cases':260,'unique_variants':len(variants),'nonconforming_cases':int((cases.invalid_transitions>0).sum()),'mean_transition_fitness':float(cases.transition_fitness.mean())},[cohorts])
