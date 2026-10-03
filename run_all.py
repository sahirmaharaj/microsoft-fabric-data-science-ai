import os
import sys
import site

if site.ENABLE_USER_SITE:
    site.addsitedir(site.getusersitepackages())

import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()

def load_catalog():
    data = json.loads((ROOT / "catalog.json").read_text())
    required = {"id", "workflow", "script", "notebook", "category"}
    if not all(required.issubset(item) for item in data):
        raise ValueError("invalid_catalog")
    identifiers = [item["id"] for item in data]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate_workflow_id")
    return data

def select_workflows(catalog, filters):
    if not filters:
        return catalog
    requested = set(filters)
    selected = []
    for item in catalog:
        identifiers = {str(item["id"]), f'{item["id"]:02d}', item["workflow"], item["category"]}
        if identifiers & requested:
            selected.append(item)
    if not selected:
        raise ValueError("no_matching_workflows")
    return selected

def ensure_dependencies(selected):
    import importlib.util
    import importlib.metadata
    if importlib.util.find_spec("packaging") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging>=23,<27"])
    from packaging.requirements import Requirement
    required = sorted({value for item in selected for value in item["dependencies"]})
    missing = []
    for value in required:
        requirement = Requirement(value)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or installed not in requirement.specifier:
            missing.append(value)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing])
    return required

def execute_workflow(item, destination, timeout):
    script = ROOT / item["script"]
    logs = destination / "execution_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / (item["workflow"] + ".log")
    environment = os.environ.copy()
    environment.update({
        "FABRIC_STARTER_OUTPUT": str(destination),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2"
    })
    started = time.perf_counter()
    status = "failed"
    return_code = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False
            )
            return_code = result.returncode
            status = "passed" if return_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
    return {
        "workflow": item["workflow"],
        "status": status,
        "return_code": return_code,
        "seconds": round(time.perf_counter() - started, 3),
        "log": str(log_path)
    }

def main():
    arguments = parse_arguments()
    catalog = load_catalog()
    selected = select_workflows(catalog, arguments.only)
    if arguments.list:
        print(json.dumps(selected, indent=2))
        return 0
    if arguments.workers < 1 or arguments.timeout < 1:
        raise ValueError("invalid_execution_limits")
    ensure_dependencies(selected)
    destination = arguments.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=min(arguments.workers, 8)) as executor:
        futures = [executor.submit(execute_workflow, item, destination, arguments.timeout) for item in selected]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda item: item["workflow"])
    summary = {
        "python": sys.version.split()[0],
        "selected": len(selected),
        "passed": sum(result["status"] == "passed" for result in results),
        "failed": sum(result["status"] != "passed" for result in results),
        "results": results
    }
    report_path = destination / "execution_report.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({"report": str(report_path), "passed": summary["passed"], "failed": summary["failed"]}, indent=2))
    return int(summary["failed"] > 0)

if __name__ == "__main__":
    raise SystemExit(main())
