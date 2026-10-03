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
WORKFLOW = "171_multiresolution_laplacian_image_blending"


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


from scipy.ndimage import gaussian_filter

height=width=192
y,x=np.mgrid[:height,:width]
image_a=np.stack([.2+.6*x/width,.3+.3*np.sin(x/18)**2,.2+.4*y/height],axis=-1)
image_b=np.stack([.7-.4*y/height,.2+.5*x/width,.4+.4*np.cos(y/20)**2],axis=-1)
image_a[((x-70)**2+(y-100)**2)<32**2]=[.9,.3,.2]
image_b[((x-120)**2+(y-90)**2)<36**2]=[.2,.4,.95]
mask=np.zeros((height,width,1))
mask[:,:width//2]=1

def reduce_image(image):
    sigma=(1.,1.,0.)
    return gaussian_filter(image,sigma=sigma,mode='reflect')[::2,::2]

def expand_image(image,target_shape):
    expanded=np.zeros((image.shape[0]*2,image.shape[1]*2,image.shape[2]))
    expanded[::2,::2]=image
    expanded=gaussian_filter(expanded,sigma=(1.,1.,0.),mode='reflect')*4
    return expanded[:target_shape[0],:target_shape[1]]

def gaussian_pyramid(image,levels):
    pyramid=[image]
    for level in range(levels):
        pyramid.append(reduce_image(pyramid[-1]))
    return pyramid

def laplacian_pyramid(gaussian):
    return [gaussian[index]-expand_image(gaussian[index+1],gaussian[index].shape) for index in range(len(gaussian)-1)]+[gaussian[-1]]

def collapse(pyramid):
    output=pyramid[-1]
    for layer in reversed(pyramid[:-1]):
        output=expand_image(output,layer.shape)+layer
    return output



pyramid_a=laplacian_pyramid(gaussian_pyramid(image_a,5))
pyramid_b=laplacian_pyramid(gaussian_pyramid(image_b,5))
pyramid_mask=gaussian_pyramid(mask,5)
blended_levels=[weight*left+(1-weight)*right for left,right,weight in zip(pyramid_a,pyramid_b,pyramid_mask)]
blended=np.clip(collapse(blended_levels),0,1)
hard=mask*image_a+(1-mask)*image_b
np.testing.assert_allclose(collapse(pyramid_a),image_a,atol=1e-12)
np.testing.assert_allclose(collapse(pyramid_b),image_b,atol=1e-12)
metrics=[]
for name,image in [('hard_seam',hard),('pyramid_blend',blended)]:
    metrics.append({'method':name,'mean_center_seam_jump':float(np.mean(abs(image[:,width//2]-image[:,width//2-1]))),'total_gradient_magnitude':float(np.mean(abs(np.diff(image,axis=0)))+np.mean(abs(np.diff(image,axis=1))))})
save_table(pd.DataFrame(metrics),'blending_seam_metrics')
array_path=output_dir/'pyramid_blending_arrays.npz'
np.savez_compressed(array_path,image_a=image_a,image_b=image_b,mask=mask,blended=blended)
record_artifact(array_path)
fig,axes=plt.subplots(1,4,figsize=(14,4))
for ax,image in zip(axes,[image_a,image_b,hard,blended]):
    ax.imshow(image)
    ax.axis('off')
save_figure(fig,'multiscale_image_blending')
finish({'pyramid_levels':len(pyramid_a),'source_reconstruction_verified':True,'output_min':float(blended.min()),'output_max':float(blended.max())},[pd.DataFrame(metrics)])
