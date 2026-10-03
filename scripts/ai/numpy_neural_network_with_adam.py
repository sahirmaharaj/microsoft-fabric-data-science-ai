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
WORKFLOW = "39_numpy_neural_network_with_adam"


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


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

X,y=make_moons(n_samples=SAMPLE_SIZE,noise=0.23,random_state=SEED)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,stratify=y,random_state=SEED)
X_fit,X_valid,y_fit,y_valid=train_test_split(X_train,y_train,test_size=0.2,stratify=y_train,random_state=SEED)
scaler=StandardScaler().fit(X_fit)
X_fit=scaler.transform(X_fit)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)

class DenseNetwork:
    def __init__(self,input_size,hidden_size,seed):
        generator=np.random.default_rng(seed)
        self.parameters={"W1":generator.normal(0,np.sqrt(2/input_size),(input_size,hidden_size)),"b1":np.zeros(hidden_size),"W2":generator.normal(0,np.sqrt(2/hidden_size),(hidden_size,1)),"b2":np.zeros(1)}
        self.first={key:np.zeros_like(value) for key,value in self.parameters.items()}
        self.second={key:np.zeros_like(value) for key,value in self.parameters.items()}
        self.step=0
    def forward(self,x):
        hidden=np.maximum(0,x@self.parameters["W1"]+self.parameters["b1"])
        logits=np.clip(hidden@self.parameters["W2"]+self.parameters["b2"],-35,35)
        probability=1/(1+np.exp(-logits))
        return hidden,probability.ravel()
    def update(self,x,y,learning_rate=0.01,l2=0.0005):
        hidden,p=self.forward(x)
        delta=(p-y).reshape(-1,1)/len(x)
        hidden_gradient=(delta@self.parameters["W2"].T)*(hidden>0)
        gradients={"W2":hidden.T@delta+l2*self.parameters["W2"],"b2":delta.sum(axis=0),"W1":x.T@hidden_gradient+l2*self.parameters["W1"],"b1":hidden_gradient.sum(axis=0)}
        self.step+=1
        for name,gradient in gradients.items():
            gradient=np.clip(gradient,-5,5)
            self.first[name]=0.9*self.first[name]+0.1*gradient
            self.second[name]=0.999*self.second[name]+0.001*gradient**2
            first_hat=self.first[name]/(1-0.9**self.step)
            second_hat=self.second[name]/(1-0.999**self.step)
            self.parameters[name]-=learning_rate*first_hat/(np.sqrt(second_hat)+1e-8)
        return float(log_loss(y,p,labels=[0,1]))



network=DenseNetwork(X_fit.shape[1],32,SEED)
best_loss=float("inf")
best_parameters=None
patience=0
history=[]
for epoch in range(250):
    permutation=rng.permutation(len(X_fit))
    losses=[]
    for start in range(0,len(permutation),64):
        batch=permutation[start:start+64]
        losses.append(network.update(X_fit[batch],y_fit[batch]))
    validation=network.forward(X_valid)[1]
    validation_loss=log_loss(y_valid,validation)
    history.append({"epoch":epoch,"training_loss":float(np.mean(losses)),"validation_loss":validation_loss})
    if validation_loss<best_loss-0.0001:
        best_loss=validation_loss
        best_parameters={key:value.copy() for key,value in network.parameters.items()}
        patience=0
    else:
        patience+=1
    if patience>=25:
        break
network.parameters=best_parameters
probability=network.forward(X_test)[1]
predictions=pd.DataFrame({"actual":y_test,"probability":probability,"prediction":probability>=0.5})
history=pd.DataFrame(history)
parameter_path=output_dir/"network_weights.npz"
np.savez_compressed(parameter_path,**network.parameters)
record_artifact(parameter_path)
save_model(scaler,"input_scaler")
save_table(history,"training_history")
save_table(predictions,"predictions")
save_json({"input_size":2,"hidden_size":32,"activation":"relu","output":"sigmoid","best_validation_loss":best_loss},"architecture")
assert all(np.isfinite(value).all() for value in network.parameters.values())


fig,axes=plt.subplots(1,2,figsize=(11,4))
history.plot(x="epoch",y=["training_loss","validation_loss"],ax=axes[0])
axes[1].scatter(X_test[:,0],X_test[:,1],c=probability,cmap="coolwarm",s=15)
save_figure(fig,"neural_network")
result=finish({"test_accuracy":accuracy_score(y_test,probability>=0.5),"test_auc":roc_auc_score(y_test,probability),"test_log_loss":log_loss(y_test,probability),"epochs":len(history)},[history.tail(),predictions])
