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
WORKFLOW = "44_character_language_model_from_scratch"


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


training_text=(
    "data science transforms observations into useful predictions. "
    "machine learning models learn patterns from training examples. "
    "validate models with held out data and measure uncertainty. "
    "python notebooks combine experiments models and visualizations. "
    "reliable data pipelines validate schemas and monitor quality. "
    "artificial intelligence systems need evaluation and human judgment. "
    "good experiments compare strong baselines and report clear metrics. "
)*12
characters=sorted(set(training_text))
char_to_index={character:index for index,character in enumerate(characters)}
index_to_char=dict(enumerate(characters))
encoded=np.array([char_to_index[character] for character in training_text])
vocabulary_size=len(characters)
hidden_size=40
sequence_length=40
parameters={"Wx":rng.normal(0,0.08,(hidden_size,vocabulary_size)),"Wh":rng.normal(0,0.08,(hidden_size,hidden_size)),"Wy":rng.normal(0,0.08,(vocabulary_size,hidden_size)),"bh":np.zeros((hidden_size,1)),"by":np.zeros((vocabulary_size,1))}
accumulators={key:np.zeros_like(value) for key,value in parameters.items()}

class CharacterRNN:
    def __init__(self,weights):
        self.weights=weights
    def forward_backward(self,inputs,targets,initial_hidden):
        hidden={-1:initial_hidden.copy()}
        probabilities={}
        loss=0.0
        for step,token in enumerate(inputs):
            hidden[step]=np.tanh(self.weights["Wx"][:,[token]]+self.weights["Wh"]@hidden[step-1]+self.weights["bh"])
            logits=self.weights["Wy"]@hidden[step]+self.weights["by"]
            exponent=np.exp(logits-logits.max())
            probabilities[step]=exponent/exponent.sum()
            loss-=float(np.log(probabilities[step][targets[step],0]+1e-12))
        gradients={key:np.zeros_like(value) for key,value in self.weights.items()}
        next_hidden=np.zeros_like(initial_hidden)
        for step in reversed(range(len(inputs))):
            delta=probabilities[step].copy()
            delta[targets[step]]-=1
            gradients["Wy"]+=delta@hidden[step].T
            gradients["by"]+=delta
            raw=(1-hidden[step]**2)*(self.weights["Wy"].T@delta+next_hidden)
            gradients["bh"]+=raw
            gradients["Wx"][:,[inputs[step]]]+=raw
            gradients["Wh"]+=raw@hidden[step-1].T
            next_hidden=self.weights["Wh"].T@raw
        for gradient in gradients.values():
            np.clip(gradient,-5,5,out=gradient)
        return loss/len(inputs),gradients,hidden[len(inputs)-1]
    def generate(self,prompt,length=180,temperature=0.8):
        hidden=np.zeros((hidden_size,1))
        tokens=[char_to_index.get(character,char_to_index[" "]) for character in prompt]
        for token in tokens[:-1]:
            hidden=np.tanh(self.weights["Wx"][:,[token]]+self.weights["Wh"]@hidden+self.weights["bh"])
        current=tokens[-1]
        output=list(prompt)
        for _ in range(length):
            hidden=np.tanh(self.weights["Wx"][:,[current]]+self.weights["Wh"]@hidden+self.weights["bh"])
            logits=(self.weights["Wy"]@hidden+self.weights["by"])/temperature
            probability=np.exp(logits-logits.max()).ravel()
            probability/=probability.sum()
            current=int(rng.choice(vocabulary_size,p=probability))
            output.append(index_to_char[current])
        return "".join(output)



model=CharacterRNN(parameters)
hidden=np.zeros((hidden_size,1))
pointer=0
history=[]
for step in range(700):
    if pointer+sequence_length+1>=len(encoded):
        pointer=0
        hidden=np.zeros((hidden_size,1))
    inputs=encoded[pointer:pointer+sequence_length]
    targets=encoded[pointer+1:pointer+sequence_length+1]
    loss,gradients,hidden=model.forward_backward(inputs,targets,hidden)
    for key in parameters:
        accumulators[key]+=gradients[key]**2
        parameters[key]-=0.075*gradients[key]/(np.sqrt(accumulators[key])+1e-8)
    pointer+=sequence_length
    history.append({"step":step,"training_cross_entropy":loss,"training_perplexity":float(np.exp(min(loss,20)))})
samples=pd.DataFrame([{"temperature":temperature,"prompt":prompt,"generated_text":model.generate(prompt,temperature=temperature)} for temperature in [0.5,0.8,1.1] for prompt in ["data ","machine "]])
weights_path=output_dir/"character_rnn_weights.npz"
np.savez_compressed(weights_path,**parameters)
record_artifact(weights_path)
save_json({"characters":characters,"hidden_size":hidden_size,"sequence_length":sequence_length,"training_characters":len(encoded),"evaluation_scope":"training_only"},"language_model_config")
save_table(pd.DataFrame(history),"training_history")
save_table(samples,"generated_samples")
assert all(np.isfinite(value).all() for value in parameters.values())
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(pd.DataFrame(history).training_cross_entropy.rolling(20).mean())
ax.set(xlabel="Training step",ylabel="Rolling cross entropy")
save_figure(fig,"language_model_training")
result=finish({"vocabulary_size":vocabulary_size,"parameters":sum(value.size for value in parameters.values()),"initial_mean_loss":float(pd.DataFrame(history).training_cross_entropy.head(30).mean()),"final_mean_loss":float(pd.DataFrame(history).training_cross_entropy.tail(30).mean())},[samples])
