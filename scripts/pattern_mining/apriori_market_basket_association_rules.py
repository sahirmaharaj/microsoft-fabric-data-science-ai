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
WORKFLOW = "143_apriori_market_basket_association_rules"


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


from itertools import combinations
from collections import Counter

items=['bread','milk','coffee','tea','butter','jam','rice','beans','pasta','sauce']
baskets=[]
for customer in range(1600):
    basket={item for item in items if rng.random()<.12}
    if 'bread' in basket and rng.random()<.7:
        basket.add('butter')
    if 'pasta' in basket and rng.random()<.8:
        basket.add('sauce')
    if 'coffee' in basket and rng.random()<.55:
        basket.add('milk')
    baskets.append(frozenset(basket))
minimum_count=65
support={}
level={frozenset([item]) for item in items}
level_history=[]
for size in range(1,5):
    counts={candidate:sum(candidate<=basket for basket in baskets) for candidate in level}
    frequent={candidate for candidate,count in counts.items() if count>=minimum_count}
    support.update({candidate:counts[candidate] for candidate in frequent})
    level_history.append({'size':size,'candidates':len(level),'frequent':len(frequent)})
    candidates={left|right for left in frequent for right in frequent if len(left|right)==size+1}
    level={candidate for candidate in candidates if all(frozenset(subset) in frequent for subset in combinations(candidate,size))}
    if not level:
        break
rules=[]
for itemset,count in support.items():
    if len(itemset)<2:
        continue
    for size in range(1,len(itemset)):
        for antecedent_tuple in combinations(sorted(itemset),size):
            antecedent=frozenset(antecedent_tuple)
            consequent=itemset-antecedent
            confidence=count/support[antecedent]
            consequent_probability=support[consequent]/len(baskets)
            lift=confidence/consequent_probability
            rules.append({'antecedent':','.join(sorted(antecedent)),'consequent':','.join(sorted(consequent)),'support':count/len(baskets),'confidence':confidence,'lift':lift,'leverage':count/len(baskets)-(support[antecedent]/len(baskets))*consequent_probability,'conviction':(1-consequent_probability)/max(1-confidence,1e-12)})


rule_table=pd.DataFrame(rules).sort_values(['lift','support'],ascending=False)
itemsets=pd.DataFrame([{'items':','.join(sorted(itemset)),'size':len(itemset),'count':count,'support':count/len(baskets)} for itemset,count in support.items()])
assert rule_table.confidence.between(0,1).all()
assert itemsets['count'].ge(minimum_count).all()
save_table(rule_table,'association_rules')
save_table(itemsets,'frequent_itemsets')
save_table(pd.DataFrame(level_history),'candidate_pruning_history')
save_table(pd.DataFrame({'basket_id':range(len(baskets)),'items':[json.dumps(sorted(basket)) for basket in baskets]}),'market_baskets')
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(rule_table.support,rule_table.confidence,c=rule_table.lift,cmap='viridis')
ax.set(xlabel='Support',ylabel='Confidence')
save_figure(fig,'association_rule_strength')
finish({'transactions':len(baskets),'frequent_itemsets':len(itemsets),'rules':len(rule_table),'maximum_lift':float(rule_table.lift.max()),'rules_are_associations_not_causal_effects':True},[rule_table.head(12)])
