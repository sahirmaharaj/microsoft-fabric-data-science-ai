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
WORKFLOW = "159_hamming_secded_error_correction"


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

positions=np.arange(1,8)
data_positions=np.array([2,4,5,6])
parity_matrix=np.array([((positions>>bit)&1) for bit in range(3)])

def encode(message):
    code=np.zeros(8,dtype=np.uint8)
    code[data_positions]=message
    for bit,position in enumerate([0,1,3]):
        code[position]=np.sum(code[:7]*parity_matrix[bit])%2
    code[7]=np.sum(code[:7])%2
    return code

def decode(received):
    code=received.copy()
    syndrome_bits=(parity_matrix@code[:7])%2
    syndrome=int(syndrome_bits@np.array([1,2,4]))
    odd=int(code.sum()%2)
    status='clean'
    if syndrome and odd:
        code[syndrome-1]^=1
        status='corrected_data_or_check_bit'
    elif not syndrome and odd:
        code[7]^=1
        status='corrected_overall_parity'
    elif syndrome and not odd:
        status='uncorrectable_double_error'
    return code[data_positions],status,code

checks=[]
for number in range(16):
    message=((number>>np.arange(4))&1).astype(np.uint8)
    code=encode(message)
    assert np.all((parity_matrix@code[:7])%2==0)
    for error_count in range(3):
        for flips in combinations(range(8),error_count):
            received=code.copy()
            received[list(flips)]^=1
            recovered,status,corrected=decode(received)
            if error_count<=1:
                assert np.array_equal(recovered,message)
                assert np.array_equal(corrected,code)
            else:
                assert status=='uncorrectable_double_error'
            checks.append({'message':number,'error_count':error_count,'flipped_positions':json.dumps(flips),'status':status,'payload_matches':bool(np.array_equal(recovered,message))})


channel=[]
for probability in [.001,.01,.03,.08]:
    count=5000
    corrected_count=detected_count=undetected_corruption=0
    for index in range(count):
        message=rng.integers(0,2,4,dtype=np.uint8)
        encoded=encode(message)
        received=encoded^(rng.random(8)<probability).astype(np.uint8)
        recovered,status,_=decode(received)
        corrected_count+=status.startswith('corrected')
        detected_count+=status=='uncorrectable_double_error'
        undetected_corruption+=status!='uncorrectable_double_error' and not np.array_equal(recovered,message)
    channel.append({'bit_error_probability':probability,'blocks':count,'corrected_status':corrected_count,'detected_uncorrectable':detected_count,'undetected_payload_corruptions':undetected_corruption})
save_table(pd.DataFrame(checks),'exhaustive_zero_one_two_bit_checks')
save_table(pd.DataFrame(channel),'noisy_channel_simulation')
save_json({'data_positions_zero_based':data_positions,'parity_check_matrix':parity_matrix,'payload_bits':4,'codeword_bits':8,'minimum_distance':4},'secded_code_definition')
fig,ax=plt.subplots(figsize=(8,4))
frame=pd.DataFrame(channel)
ax.plot(frame.bit_error_probability,frame.undetected_payload_corruptions,marker='o')
ax.set(xlabel='Channel bit error probability',ylabel='Undetected payload corruption count')
save_figure(fig,'secded_higher_order_error_risk')
finish({'exhaustive_checks':len(checks),'all_single_errors_corrected':True,'all_double_errors_detected':True,'three_or_more_errors_not_guaranteed':True},[pd.DataFrame(channel)])
