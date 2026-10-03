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
WORKFLOW = "158_hidden_semi_markov_duration_decoding"


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


from scipy.stats import norm, poisson

state_means=np.array([-2.,.2,2.3])
state_sd=.75
maximum_duration=18
duration_support=np.arange(1,maximum_duration+1)
duration_probability=np.stack([poisson.pmf(duration_support-1,mean) for mean in [6,9,4]])
duration_probability/=duration_probability.sum(axis=1,keepdims=True)
transition=np.array([[0,.8,.2],[.3,0,.7],[.7,.3,0]])
true_states=[]
state=0
while len(true_states)<220:
    duration=int(rng.choice(duration_support,p=duration_probability[state]))
    true_states.extend([state]*duration)
    state=int(rng.choice(3,p=transition[state]))
true_states=np.asarray(true_states[:220])
observations=state_means[true_states]+rng.normal(0,state_sd,len(true_states))
emission=norm.logpdf(observations[:,None],state_means,state_sd)
cumulative=np.vstack([np.zeros(3),np.cumsum(emission,axis=0)])
T=len(observations)
score=np.full((T+1,3),-np.inf)
back_state=np.full((T+1,3),-1,dtype=int)
back_duration=np.zeros((T+1,3),dtype=int)
for end in range(1,T+1):
    for state in range(3):
        for duration in range(1,min(maximum_duration,end)+1):
            start=end-duration
            segment_emission=cumulative[end,state]-cumulative[start,state]
            duration_score=np.log(duration_probability[state,duration-1])
            if start==0:
                candidate=np.log(1/3)+duration_score+segment_emission
                previous=-1
            else:
                previous_scores=score[start]+np.where(transition[:,state]>0,np.log(np.maximum(transition[:,state],1e-300)),-np.inf)
                previous=int(previous_scores.argmax())
                candidate=previous_scores[previous]+duration_score+segment_emission
            if candidate>score[end,state]:
                score[end,state]=candidate
                back_state[end,state]=previous
                back_duration[end,state]=duration


prediction=np.empty(T,dtype=int)
segments=[]
end=T
state=int(score[T].argmax())
while end>0:
    duration=int(back_duration[end,state])
    start=end-duration
    prediction[start:end]=state
    segments.append({'start':start,'end_exclusive':end,'state':state,'duration':duration})
    state,end=int(back_state[end,state]),start
assert sum(segment['duration'] for segment in segments)==T
assert all(1<=segment['duration']<=maximum_duration for segment in segments)
output=pd.DataFrame({'time':range(T),'observation':observations,'true_state':true_states,'decoded_state':prediction})
save_table(output,'duration_aware_state_decoding')
save_table(pd.DataFrame(segments).sort_values('start'),'decoded_segments')
save_table(pd.DataFrame(duration_probability.T,columns=['state0','state1','state2']).assign(duration=duration_support),'duration_distributions')
fig,axes=plt.subplots(2,1,figsize=(11,6))
axes[0].plot(observations)
axes[1].step(range(T),true_states,label='truth')
axes[1].step(range(T),prediction,alpha=.6,label='decoded')
axes[1].legend()
save_figure(fig,'semi_markov_segments')
finish({'state_accuracy':float((prediction==true_states).mean()),'decoded_segments':len(segments),'model_parameters_known_in_synthetic_example':True,'final_segment_scored_as_complete':True},[pd.DataFrame(segments).sort_values('start')])
