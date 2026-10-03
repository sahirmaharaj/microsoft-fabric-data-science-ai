import os
import sys
import subprocess
import importlib.util
import importlib.metadata

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
for thread_variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(thread_variable, "2")


SEED = 42
SAMPLE_SIZE = 1200
OUTPUT_ROOT = os.environ.get("FABRIC_STARTER_OUTPUT", "")
WORKFLOW = "40_denoising_autoencoder_anomaly_scoring"


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


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

digits=load_digits()
X=digits.data/16.0
normal=X[digits.target<5]
anomalous=X[digits.target>=5]
train,remaining=train_test_split(normal,test_size=0.35,random_state=SEED)
calibration,test_normal=train_test_split(remaining,test_size=0.5,random_state=SEED)
test=np.vstack([test_normal,anomalous[:len(test_normal)]])
y_test=np.r_[np.zeros(len(test_normal)),np.ones(len(test_normal))]
input_size,hidden_size=64,20
W1=rng.normal(0,0.12,(input_size,hidden_size))
b1=np.zeros(hidden_size)
W2=rng.normal(0,0.12,(hidden_size,input_size))
b2=np.zeros(input_size)
learning_rate=0.08
history=[]

def reconstruct(data):
    hidden=np.tanh(data@W1+b1)
    output=1/(1+np.exp(-np.clip(hidden@W2+b2,-30,30)))
    return hidden,output

for epoch in range(180):
    indices=rng.permutation(len(train))
    batch_losses=[]
    for start in range(0,len(indices),64):
        batch=train[indices[start:start+64]]
        noisy=np.clip(batch+rng.normal(0,0.12,batch.shape),0,1)
        hidden,output=reconstruct(noisy)
        output_gradient=(output-batch)*output*(1-output)*2/len(batch)
        hidden_gradient=(output_gradient@W2.T)*(1-hidden**2)
        dW2=hidden.T@output_gradient+0.0001*W2
        db2=output_gradient.sum(axis=0)
        dW1=noisy.T@hidden_gradient+0.0001*W1
        db1=hidden_gradient.sum(axis=0)
        W1-=learning_rate*dW1
        b1-=learning_rate*db1
        W2-=learning_rate*dW2
        b2-=learning_rate*db2
        batch_losses.append(float(np.mean((output-batch)**2)))
    cal_output=reconstruct(calibration)[1]
    history.append({"epoch":epoch,"train_mse":float(np.mean(batch_losses)),"calibration_mse":float(np.mean((cal_output-calibration)**2))})


cal_scores=np.mean((reconstruct(calibration)[1]-calibration)**2,axis=1)
threshold=float(np.quantile(cal_scores,0.95,method="higher"))
latent,reconstructed=reconstruct(test)
scores=np.mean((reconstructed-test)**2,axis=1)
output=pd.DataFrame({"actual_anomaly":y_test,"reconstruction_error":scores,"alert":scores>threshold})
latent_frame=pd.DataFrame(latent,columns=[f"latent_{i}" for i in range(hidden_size)])
weights_path=output_dir/"autoencoder_weights.npz"
np.savez_compressed(weights_path,W1=W1,b1=b1,W2=W2,b2=b2)
record_artifact(weights_path)
save_table(output,"anomaly_scores")
save_table(pd.DataFrame(history),"training_history")
save_table(latent_frame,"latent_features")
save_json({"normal_digits":[0,1,2,3,4],"threshold":threshold,"calibration_quantile":0.95,"input_scale":16.0},"anomaly_configuration")
assert reconstructed.shape==test.shape
assert np.isfinite(scores).all()
fig,axes=plt.subplots(2,6,figsize=(12,4))
for column,index in enumerate(np.argsort(scores)[-6:]):
    axes[0,column].imshow(test[index].reshape(8,8),cmap="gray")
    axes[1,column].imshow(reconstructed[index].reshape(8,8),cmap="gray")
    axes[0,column].axis("off")
    axes[1,column].axis("off")
save_figure(fig,"reconstructions")
result=finish({"anomaly_auc":roc_auc_score(y_test,scores),"average_precision":average_precision_score(y_test,scores),"threshold":threshold,"alerts":int(output.alert.sum())},[output.sort_values("reconstruction_error",ascending=False)])
