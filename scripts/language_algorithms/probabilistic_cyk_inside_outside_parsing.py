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
WORKFLOW = "169_probabilistic_cyk_inside_outside_parsing"


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


from collections import defaultdict

binary=[('S','NP','VP',1.),('NP','Det','N',.7),('NP','NP','PP',.3),('VP','V','NP',.7),('VP','VP','PP',.3),('PP','P','NP',1.)]
lexical={'Det':{'the':.7,'a':.3},'N':{'analyst':.3,'scientist':.3,'telescope':.2,'report':.2},'V':{'saw':.6,'reviewed':.4},'P':{'with':.7,'near':.3}}
tokens='the analyst saw the scientist with the telescope'.split()
length=len(tokens)
inside=defaultdict(lambda:-np.inf)
vit=defaultdict(lambda:-np.inf)
back={}
for position,word in enumerate(tokens):
    for symbol,words in lexical.items():
        if word in words:
            key=(position,position+1,symbol)
            inside[key]=vit[key]=np.log(words[word])
            back[key]=word
for width in range(2,length+1):
    for start in range(length-width+1):
        end=start+width
        for parent,left,right,probability in binary:
            key=(start,end,parent)
            for split in range(start+1,end):
                a=(start,split,left)
                b=(split,end,right)
                candidate=np.log(probability)+inside[a]+inside[b]
                inside[key]=np.logaddexp(inside[key],candidate)
                best=np.log(probability)+vit[a]+vit[b]
                if best>vit[key]:
                    vit[key]=best
                    back[key]=(a,b)
root=(0,length,'S')
assert np.isfinite(inside[root])
outside=defaultdict(lambda:-np.inf)
outside[root]=0.
for width in range(length,1,-1):
    for start in range(length-width+1):
        end=start+width
        for parent,left,right,probability in binary:
            parent_score=outside[(start,end,parent)]
            for split in range(start+1,end):
                a=(start,split,left)
                b=(split,end,right)
                outside[a]=np.logaddexp(outside[a],parent_score+np.log(probability)+inside[b])
                outside[b]=np.logaddexp(outside[b],parent_score+np.log(probability)+inside[a])



def render_tree(key):
    value=back[key]
    if isinstance(value,str):
        return '('+key[2]+' '+value+')'
    return '('+key[2]+' '+render_tree(value[0])+' '+render_tree(value[1])+')'

rows=[]
for key,value in list(inside.items()):
    if np.isfinite(value) and np.isfinite(outside[key]):
        posterior=float(np.exp(value+outside[key]-inside[root]))
        rows.append({'start':key[0],'end_exclusive':key[1],'symbol':key[2],'text':' '.join(tokens[key[0]:key[1]]),'constituent_posterior':posterior})
output=pd.DataFrame(rows).sort_values(['start','end_exclusive','symbol'])
assert output.constituent_posterior.le(1+1e-9).all()
assert vit[root]<=inside[root]+1e-10
save_table(output,'constituent_posteriors')
save_json({'tokens':tokens,'viterbi_tree':render_tree(root),'sentence_log_probability':inside[root],'viterbi_log_probability':vit[root],'best_tree_posterior':np.exp(vit[root]-inside[root])},'probabilistic_parse')
save_json({'binary_rules':binary,'lexical_rules':lexical},'pcfg_grammar')
finish({'tokens':length,'sentence_probability':float(np.exp(inside[root])),'best_tree_posterior':float(np.exp(vit[root]-inside[root])),'posterior_constituents':len(output),'grammar_form':'binary_CNF_with_lexical_rules'},[output])
