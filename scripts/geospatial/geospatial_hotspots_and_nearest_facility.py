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
WORKFLOW = "52_geospatial_hotspots_and_nearest_facility"


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


from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

earth_radius_km=6371.0088
centers=np.array([[-33.9249,18.4241],[-33.9900,18.4700],[-33.9000,18.6000]])
assignments=rng.integers(0,len(centers),SAMPLE_SIZE)
coordinates=centers[assignments]+rng.normal(0,[0.008,0.010],(SAMPLE_SIZE,2))
points=pd.DataFrame(coordinates,columns=["latitude","longitude"])
points["event_id"]=np.arange(len(points))
points["demand"]=rng.integers(1,8,len(points))
facilities=pd.DataFrame({"facility":["central","south","east","north"],"latitude":[-33.925,-34.0,-33.905,-33.85],"longitude":[18.42,18.48,18.60,18.50],"capacity":[1800,1700,1900,1200]})
radians=np.radians(points[["latitude","longitude"]].to_numpy())
facility_radians=np.radians(facilities[["latitude","longitude"]].to_numpy())
tree=BallTree(facility_radians,metric="haversine")
distance,index=tree.query(radians,k=2)
points["nearest_facility"]=facilities.facility.iloc[index[:,0]].to_numpy()
points["distance_km"]=distance[:,0]*earth_radius_km
points["backup_facility"]=facilities.facility.iloc[index[:,1]].to_numpy()
points["backup_distance_km"]=distance[:,1]*earth_radius_km
clusterer=DBSCAN(eps=0.65/earth_radius_km,min_samples=12,metric="haversine",algorithm="ball_tree")
points["hotspot"]=clusterer.fit_predict(radians)
hotspots=points.loc[points.hotspot.ge(0)].groupby("hotspot").agg(events=("event_id","size"),demand=("demand","sum"),latitude=("latitude","mean"),longitude=("longitude","mean"),mean_distance_km=("distance_km","mean")).reset_index()
loads=points.groupby("nearest_facility").agg(events=("event_id","size"),demand=("demand","sum"),p95_distance_km=("distance_km",lambda values:values.quantile(0.95))).reset_index().rename(columns={"nearest_facility":"facility"})
loads=facilities.merge(loads,on="facility",how="left").fillna({"events":0,"demand":0,"p95_distance_km":0})
loads["utilization"]=loads.demand/loads.capacity
coverage=[]
for radius in [1,2,3,5,8]:
    covered=points.distance_km<=radius
    coverage.append({"radius_km":radius,"event_coverage":float(covered.mean()),"demand_coverage":float(points.loc[covered,"demand"].sum()/points.demand.sum())})
coverage=pd.DataFrame(coverage)
lat1,lon1=facility_radians[:,0,None],facility_radians[:,1,None]
lat2,lon2=facility_radians[:,0][None,:],facility_radians[:,1][None,:]
haversine=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
facility_distances=2*earth_radius_km*np.arcsin(np.sqrt(np.clip(haversine,0,1)))
save_table(points,"event_assignments")


save_table(hotspots,"hotspots")
save_table(loads,"facility_loads")
save_table(coverage,"coverage_curve")
save_table(pd.DataFrame(facility_distances,columns=facilities.facility).reset_index(names="facility_index"),"facility_distances")
save_model({"tree":tree,"facilities":facilities,"earth_radius_km":earth_radius_km},"facility_index")
assert points.distance_km.ge(0).all()
assert np.allclose(facility_distances,facility_distances.T)
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].scatter(points.longitude,points.latitude,c=points.hotspot,cmap="tab10",s=8,alpha=0.5)
axes[0].scatter(facilities.longitude,facilities.latitude,marker="*",s=200,color="black")
axes[0].set(xlabel="Longitude",ylabel="Latitude")
coverage.plot(x="radius_km",y=["event_coverage","demand_coverage"],ax=axes[1])
save_figure(fig,"geospatial_analysis")
result=finish({"events":len(points),"hotspots":len(hotspots),"mean_distance_km":float(points.distance_km.mean()),"noise_events":int(points.hotspot.eq(-1).sum()),"overloaded_facilities":int(loads.utilization.gt(1).sum())},[hotspots,loads])
