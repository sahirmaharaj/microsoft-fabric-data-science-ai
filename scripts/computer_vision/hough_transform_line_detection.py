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
WORKFLOW = "172_hough_transform_line_detection"


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


from scipy.ndimage import gaussian_filter, sobel

height,width=150,180
image=np.zeros((height,width))
true_segments=[((15,25),(165,100)),((25,130),(150,20)),((30,45),(160,45))]
for (x1,y1),(x2,y2) in true_segments:
    steps=int(max(abs(x2-x1),abs(y2-y1))*3)
    xx=np.rint(np.linspace(x1,x2,steps)).astype(int)
    yy=np.rint(np.linspace(y1,y2,steps)).astype(int)
    image[yy,xx]=1
image=gaussian_filter(image,.9)
image/=image.max()
noisy=np.clip(image+rng.normal(0,.06,image.shape),0,1)
smooth=gaussian_filter(noisy,.8)
gradient_x=sobel(smooth,axis=1)/8
gradient_y=sobel(smooth,axis=0)/8
magnitude=np.hypot(gradient_x,gradient_y)
edges=magnitude>.11
edge_y,edge_x=np.nonzero(edges)
theta=np.deg2rad(np.arange(-90,90))
rho_limit=int(np.ceil(np.hypot(height,width)))
rho_values=np.arange(-rho_limit,rho_limit+1)
accumulator=np.zeros((len(rho_values),len(theta)),dtype=int)
for angle_index,angle in enumerate(theta):
    rho=np.rint(edge_x*np.cos(angle)+edge_y*np.sin(angle)).astype(int)+rho_limit
    np.add.at(accumulator[:,angle_index],rho,1)
working=accumulator.copy()
peaks=[]
for index in range(8):
    rho_index,angle_index=np.unravel_index(np.argmax(working),working.shape)
    votes=int(working[rho_index,angle_index])
    if votes<35:
        break
    peaks.append({'line':index,'rho':int(rho_values[rho_index]),'theta_degrees':float(np.rad2deg(theta[angle_index])),'votes':votes})
    working[max(0,rho_index-7):rho_index+8,max(0,angle_index-6):angle_index+7]=0


rows=[]
for peak in peaks:
    angle=np.deg2rad(peak['theta_degrees'])
    distance=abs(edge_x*np.cos(angle)+edge_y*np.sin(angle)-peak['rho'])
    rows.append({**peak,'supporting_edge_pixels':int((distance<1.5).sum()),'mean_support_distance':float(distance[distance<1.5].mean())})
assert accumulator.sum()==len(edge_x)*len(theta)
assert len(peaks)>0
save_table(pd.DataFrame(rows),'detected_line_parameters')
save_table(pd.DataFrame([{'line':index,'x1':a[0],'y1':a[1],'x2':b[0],'y2':b[1]} for index,(a,b) in enumerate(true_segments)]),'synthetic_line_segments')
array_path=output_dir/'hough_transform_arrays.npz'
np.savez_compressed(array_path,image=noisy,edges=edges,accumulator=accumulator,theta=theta,rho=rho_values)
record_artifact(array_path)
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].imshow(noisy,cmap='gray')
for peak in peaks:
    angle=np.deg2rad(peak['theta_degrees'])
    if abs(np.sin(angle))>.1:
        xx=np.array([0,width-1])
        yy=(peak['rho']-xx*np.cos(angle))/np.sin(angle)
        axes[0].plot(xx,yy,alpha=.7)
axes[0].set(xlim=(0,width-1),ylim=(height-1,0))
axes[1].imshow(accumulator,aspect='auto',extent=[-90,90,rho_limit,-rho_limit])
axes[1].set(xlabel='Normal angle in degrees',ylabel='Rho')
save_figure(fig,'hough_lines_and_accumulator')
finish({'edge_pixels':len(edge_x),'detected_line_peaks':len(peaks),'accumulator_votes_conserved':True,'thick_line_edges_can_produce_parallel_detections':True},[pd.DataFrame(rows)])
