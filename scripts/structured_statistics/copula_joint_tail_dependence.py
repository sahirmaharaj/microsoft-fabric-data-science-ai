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
WORKFLOW = "135_copula_joint_tail_dependence"


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


from scipy.stats import norm, rankdata, t, gamma, lognorm

count=1500
correlation=np.array([[1,.65,.35],[.65,1,.5],[.35,.5,1]])
normal=rng.multivariate_normal(np.zeros(3),correlation,count)
student=normal/np.sqrt(rng.chisquare(4,count)[:,None]/4)
uniform=t.cdf(student,df=4)
X=np.column_stack([lognorm.ppf(uniform[:,0],s=.7),gamma.ppf(uniform[:,1],a=3),t.ppf(uniform[:,2],df=7)])
training=X[:1000]
heldout=X[1000:]
pseudo=np.column_stack([rankdata(training[:,column])/(len(training)+1) for column in range(3)])
latent=norm.ppf(pseudo)
estimated_correlation=np.corrcoef(latent,rowvar=False)
synthetic_latent=rng.multivariate_normal(np.zeros(3),estimated_correlation,10000)
synthetic_uniform=norm.cdf(synthetic_latent)
synthetic=np.column_stack([np.quantile(training[:,column],synthetic_uniform[:,column]) for column in range(3)])
rows=[]
for quantile in [.8,.9,.95]:
    threshold=np.quantile(training,quantile,axis=0)
    for left,right in [(0,1),(0,2),(1,2)]:
        actual=(heldout[:,left]>threshold[left])&(heldout[:,right]>threshold[right])
        generated=(synthetic[:,left]>threshold[left])&(synthetic[:,right]>threshold[right])
        rows.append({'quantile':quantile,'left':left,'right':right,'heldout_joint_exceedance':float(actual.mean()),'gaussian_copula_joint_exceedance':float(generated.mean()),'independence_reference':(1-quantile)**2})
comparison=pd.DataFrame(rows)
comparison['absolute_tail_gap']=abs(comparison.heldout_joint_exceedance-comparison.gaussian_copula_joint_exceedance)
marginal=pd.DataFrame({'variable':range(3),'training_mean':training.mean(axis=0),'synthetic_mean':synthetic.mean(axis=0),'training_sd':training.std(axis=0),'synthetic_sd':synthetic.std(axis=0)})
assert np.linalg.eigvalsh(estimated_correlation).min()>0
assert np.isfinite(synthetic).all()
save_table(comparison,'joint_tail_backtest')
save_table(marginal,'marginal_preservation')
save_table(pd.DataFrame(synthetic,columns=['metric_0','metric_1','metric_2']),'copula_synthetic_observations')
save_json({'latent_correlation':estimated_correlation,'marginal_method':'empirical_quantile','fitted_copula':'gaussian','synthetic_truth_copula':'student_t_4'},'copula_configuration')
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].scatter(training[:,0],training[:,1],s=8,alpha=.4)
axes[1].scatter(synthetic[:1000,0],synthetic[:1000,1],s=8,alpha=.4)


save_figure(fig,'copula_dependence_comparison')
finish({'mean_joint_tail_absolute_gap':float(comparison.absolute_tail_gap.mean()),'heldout_rows':len(heldout),'gaussian_copula_can_miss_tail_dependence':True},[comparison])
