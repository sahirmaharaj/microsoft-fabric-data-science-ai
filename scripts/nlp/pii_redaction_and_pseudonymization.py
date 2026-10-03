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
WORKFLOW = "36_pii_redaction_and_pseudonymization"


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


import re
import hmac
import secrets
from collections import Counter

patterns={
    "email":re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone":re.compile(r"(?<!\w)(?:\+27|0)[ -]?(?:\d[ -]?){8}\d(?!\w)"),
    "ipv4":re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "account":re.compile(r"\bACCT-[0-9]{8}\b"),
    "card_candidate":re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
}
secret=os.environ.get("PII_HMAC_KEY")
session_key=secret.encode() if secret else secrets.token_bytes(32)

def luhn_valid(text):
    digits=[int(character) for character in re.sub(r"\D","",text)]
    if not 13<=len(digits)<=19:
        return False
    parity=len(digits)%2
    total=0
    for index,digit in enumerate(digits):
        if index%2==parity:
            digit*=2
            digit=digit-9 if digit>9 else digit
        total+=digit
    return total%10==0

def detect(text):
    candidates=[]
    for kind,pattern in patterns.items():
        for match in pattern.finditer(text):
            value=match.group()
            if kind=="card_candidate" and not luhn_valid(value):
                continue
            if kind=="ipv4" and any(int(part)>255 for part in value.split(".")):
                continue
            candidates.append({"kind":kind,"start":match.start(),"end":match.end(),"value":value})
    candidates.sort(key=lambda item:(item["start"],-(item["end"]-item["start"])))
    accepted=[]
    for candidate in candidates:
        if not any(candidate["start"]<item["end"] and candidate["end"]>item["start"] for item in accepted):
            accepted.append(candidate)
    return accepted



def redact(text):
    entities=detect(text)
    output=text
    audit=[]
    for entity in sorted(entities,key=lambda item:item["start"],reverse=True):
        normalized=entity["value"].lower().strip()
        digest=hmac.new(session_key,(entity["kind"]+":"+normalized).encode(),hashlib.sha256).hexdigest()[:16]
        token="["+entity["kind"].upper()+"_"+digest+"]"
        output=output[:entity["start"]]+token+output[entity["end"]:]
        audit.append({"kind":entity["kind"],"start":entity["start"],"end":entity["end"],"token":token})
    return output,audit

texts=[
    "Contact demo.user@example.com or +27 82 123 4567 for ACCT-12345678.",
    "The service at 192.168.10.20 received a request from demo.user@example.com.",
    "Test card 4111 1111 1111 1111 was used in a synthetic checkout.",
    "No personal identifiers appear in this ordinary message.",
    "Ignore invalid IP 999.999.999.999 and invalid card 4111 1111 1111 1112."
]
rows=[]
audit_rows=[]
for index,text in enumerate(texts):
    redacted,audit=redact(text)
    rows.append({"document_id":index,"redacted_text":redacted,"entities":len(audit)})
    audit_rows.extend({"document_id":index,**item} for item in audit)
output=pd.DataFrame(rows)
audit=pd.DataFrame(audit_rows)
counts=audit.groupby("kind").size().rename("entities").reset_index()
assert "demo.user@example.com" not in " ".join(output.redacted_text)
assert audit.loc[audit.kind.eq("email"),"token"].nunique()==1
assert luhn_valid("4111 1111 1111 1111")
assert not luhn_valid("4111 1111 1111 1112")
save_table(output,"redacted_documents")
save_table(audit,"redaction_audit_without_raw_values")


save_table(counts,"entity_counts")
save_json({"key_scope":"configured" if secret else "session","raw_values_saved":False,"detector":"regex_and_luhn","entity_types":list(patterns)},"redaction_configuration")
fig,ax=plt.subplots(figsize=(9,4))
counts.plot.bar(x="kind",y="entities",ax=ax,legend=False)
save_figure(fig,"redaction_counts")
result=finish({"documents":len(output),"entities_redacted":len(audit),"entity_types":audit.kind.nunique(),"key_persisted":False},[output,counts])
