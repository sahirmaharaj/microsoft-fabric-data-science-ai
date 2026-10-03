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
WORKFLOW = "151_capacity_constrained_stable_matching"


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


from collections import deque

applicants=32
institutions=5
capacities=np.array([5,6,4,7,4])
applicant_preferences=np.stack([rng.permutation(institutions) for _ in range(applicants)])
institution_preferences=np.stack([rng.permutation(applicants) for _ in range(institutions)])
institution_rank=np.argsort(institution_preferences,axis=1)
applicant_rank=np.argsort(applicant_preferences,axis=1)
accepted=[[] for _ in range(institutions)]
next_choice=np.zeros(applicants,dtype=int)
free=deque(range(applicants))
proposal_log=[]
while free:
    applicant=free.popleft()
    if next_choice[applicant]>=institutions:
        continue
    institution=int(applicant_preferences[applicant,next_choice[applicant]])
    next_choice[applicant]+=1
    pool=accepted[institution]+[applicant]
    pool.sort(key=lambda candidate:institution_rank[institution,candidate])
    kept=pool[:capacities[institution]]
    rejected=pool[capacities[institution]:]
    accepted[institution]=kept
    proposal_log.append({'step':len(proposal_log),'applicant':applicant,'institution':institution,'accepted_at_step':applicant in kept,'displaced_applicant':int(rejected[0]) if rejected else -1})
    free.extend(rejected)
matching=np.full(applicants,-1,dtype=int)
for institution,group in enumerate(accepted):
    matching[group]=institution
blocking=[]
for applicant in range(applicants):
    assigned=matching[applicant]
    for institution in range(institutions):
        prefers=assigned<0 or applicant_rank[applicant,institution]<applicant_rank[applicant,assigned]
        room=len(accepted[institution])<capacities[institution]
        outranks=bool(accepted[institution]) and institution_rank[institution,applicant]<max(institution_rank[institution,candidate] for candidate in accepted[institution])
        if prefers and (room or outranks):
            blocking.append((applicant,institution))


assert not blocking
assert all(len(group)<=capacities[index] for index,group in enumerate(accepted))
rank=np.array([applicant_rank[index,institution]+1 if institution>=0 else institutions+1 for index,institution in enumerate(matching)])
output=pd.DataFrame({'applicant':range(applicants),'institution':matching,'preference_rank':rank,'proposals':next_choice})
summary=pd.DataFrame({'institution':range(institutions),'capacity':capacities,'filled':[len(group) for group in accepted]})
save_table(output,'stable_assignments')
save_table(pd.DataFrame(proposal_log),'deferred_acceptance_trace')
save_table(summary,'institution_capacity_usage')
save_table(pd.DataFrame(applicant_preferences),'applicant_preferences')
save_table(pd.DataFrame(institution_preferences),'institution_preferences')
fig,ax=plt.subplots(figsize=(8,4))
output.preference_rank.value_counts().sort_index().plot.bar(ax=ax)
ax.set(xlabel='Assigned preference rank, 6 means unmatched',ylabel='Applicants')
save_figure(fig,'stable_matching_preference_outcomes')
finish({'matched_applicants':int((matching>=0).sum()),'unmatched_applicants':int((matching<0).sum()),'blocking_pairs':len(blocking),'proposals':len(proposal_log),'strict_preferences':True},[summary])
