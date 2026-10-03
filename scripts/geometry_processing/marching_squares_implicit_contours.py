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
WORKFLOW = "176_marching_squares_implicit_contours"


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

axis=np.linspace(-1.6,1.6,130)
xx,yy=np.meshgrid(axis,axis)

def scalar_field(x,y):
    return (x*x+y*y-1)**2-.2+.07*x

field=scalar_field(xx,yy)
segments=[]
ambiguous_cells=0
edge_pairs=[(0,1),(1,2),(2,3),(3,0)]
for row in range(len(axis)-1):
    for column in range(len(axis)-1):
        vertices=np.array([[axis[column],axis[row]],[axis[column+1],axis[row]],[axis[column+1],axis[row+1]],[axis[column],axis[row+1]]])
        values=np.array([field[row,column],field[row,column+1],field[row+1,column+1],field[row+1,column]])
        intersections={}
        for edge,(left,right) in enumerate(edge_pairs):
            if (values[left]>=0)!=(values[right]>=0):
                fraction=values[left]/(values[left]-values[right])
                intersections[edge]=vertices[left]+fraction*(vertices[right]-vertices[left])
        if len(intersections)==2:
            left,right=intersections.values()
            segments.append((left,right))
        elif len(intersections)==4:
            ambiguous_cells+=1
            determinant=values[0]*values[2]-values[1]*values[3]
            connections=[(0,1),(2,3)] if determinant>=0 else [(3,0),(1,2)]
            for left,right in connections:
                segments.append((intersections[left],intersections[right]))

node_coordinates={}
adjacency=defaultdict(list)
for segment_id,(left,right) in enumerate(segments):
    keys=[tuple(np.round(point,10)) for point in [left,right]]
    for key,point in zip(keys,[left,right]):
        node_coordinates[key]=point
    adjacency[keys[0]].append((keys[1],segment_id))
    adjacency[keys[1]].append((keys[0],segment_id))


assert all(len(neighbors)==2 for neighbors in adjacency.values())
unused=set(range(len(segments)))
polylines=[]
while unused:
    segment_id=next(iter(unused))
    start=tuple(np.round(segments[segment_id][0],10))
    current=start
    polyline=[node_coordinates[start]]
    while True:
        candidates=[entry for entry in adjacency[current] if entry[1] in unused]
        if not candidates:
            break
        following,edge=candidates[0]
        unused.remove(edge)
        polyline.append(node_coordinates[following])
        current=following
        if current==start:
            break
    polylines.append(np.asarray(polyline))
rows=[]
contours=[]
for contour_id,polyline in enumerate(polylines):
    closed=np.allclose(polyline[0],polyline[-1])
    area=.5*abs(np.sum(polyline[:-1,0]*polyline[1:,1]-polyline[1:,0]*polyline[:-1,1]))
    length=float(np.linalg.norm(np.diff(polyline,axis=0),axis=1).sum())
    contours.append({'contour':contour_id,'vertices':len(polyline),'closed':bool(closed),'enclosed_area':float(area),'length':length})
    rows.extend({'contour':contour_id,'vertex':index,'x':point[0],'y':point[1],'absolute_field_residual':abs(float(scalar_field(*point)))} for index,point in enumerate(polyline))
output=pd.DataFrame(rows)
assert len(polylines)==2
assert all(record['closed'] for record in contours)
save_table(output,'ordered_contour_vertices')
save_table(pd.DataFrame(contours),'contour_geometry')
save_json({'grid_size':len(axis),'level':0.,'ambiguous_cells':ambiguous_cells,'saddle_rule':'bilinear_asymptotic_determinant'},'contour_parameters')
fig,ax=plt.subplots(figsize=(7,7))
ax.imshow(field,origin='lower',extent=[axis[0],axis[-1],axis[0],axis[-1]],cmap='coolwarm')


for polyline in polylines:
    ax.plot(*polyline.T,color='black')
ax.set_aspect('equal')
save_figure(fig,'implicit_level_set_contours')
finish({'closed_contours':len(polylines),'segments':len(segments),'maximum_vertex_field_residual':float(output.absolute_field_residual.max()),'annular_region_area':max(record['enclosed_area'] for record in contours)-min(record['enclosed_area'] for record in contours)},[pd.DataFrame(contours)])
