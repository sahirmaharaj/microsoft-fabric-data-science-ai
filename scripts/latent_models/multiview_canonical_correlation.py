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
WORKFLOW = "117_multiview_canonical_correlation"


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


from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

latent=rng.normal(size=(900,3))
left=latent@rng.normal(size=(3,9))+rng.normal(0,.5,(900,9))
right=latent@rng.normal(size=(3,7))+rng.normal(0,.5,(900,7))
train,test=train_test_split(np.arange(900),test_size=.3,random_state=SEED)
left_scaler=StandardScaler().fit(left[train])
right_scaler=StandardScaler().fit(right[train])
L=left_scaler.transform(left)
R=right_scaler.transform(right)
model=CCA(n_components=3,max_iter=1500,tol=1e-7).fit(L[train],R[train])
train_left,train_right=model.transform(L[train],R[train])
test_left,test_right=model.transform(L[test],R[test])
correlations=[]
for component in range(3):
    observed=float(np.corrcoef(test_left[:,component],test_right[:,component])[0,1])
    permutations=np.array([np.corrcoef(test_left[:,component],rng.permutation(test_right[:,component]))[0,1] for _ in range(250)])
    correlations.append({'component':component,'training_correlation':np.corrcoef(train_left[:,component],train_right[:,component])[0,1],'test_correlation':observed,'permutation_pvalue':(1+sum(abs(permutations)>=abs(observed)))/251})
translator=Ridge(alpha=1).fit(train_left,R[train])
predicted=translator.predict(test_left)
errors=pd.DataFrame({'feature':range(R.shape[1]),'cross_view_rmse':np.sqrt(np.mean((predicted-R[test])**2,axis=0)),'mean_baseline_rmse':np.sqrt(np.mean(R[test]**2,axis=0))})
retrieval_distance=np.linalg.norm(test_left[:,None,:]-test_right[None,:,:],axis=2)
nearest=np.argsort(retrieval_distance,axis=1)[:,:5]
retrieval_hit=np.array([index in candidates for index,candidates in enumerate(nearest)])
scores=pd.DataFrame(test_left,columns=['canonical_0','canonical_1','canonical_2'])
scores['sample_id']=test
scores['paired_view_in_top5']=retrieval_hit
assert len(scores)==len(test)
assert np.isfinite(predicted).all()
save_table(pd.DataFrame(correlations),'canonical_correlations')
save_table(errors,'cross_view_prediction')
save_table(scores,'shared_embedding')


save_model({'left_scaler':left_scaler,'right_scaler':right_scaler,'cca':model,'translator':translator},'multiview_model')
fig,ax=plt.subplots(figsize=(7,5))
ax.scatter(test_left[:,0],test_right[:,0],s=10,alpha=.5)
ax.set(xlabel='Left canonical score',ylabel='Right canonical score')
save_figure(fig,'heldout_alignment')
finish({'top5_paired_retrieval':float(retrieval_hit.mean()),'mean_cross_view_rmse':float(errors.cross_view_rmse.mean()),'heldout_rows':len(test)},[pd.DataFrame(correlations),errors])
