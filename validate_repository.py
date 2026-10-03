import sys
import site

if site.ENABLE_USER_SITE:
    site.addsitedir(site.getusersitepackages())

import subprocess
import importlib.util

DEPENDENCIES = {"nbformat": "nbformat>=5.10,<6", "nbclient": "nbclient>=0.10,<1", "ipykernel": "ipykernel>=6.29,<8"}
missing = [requirement for module, requirement in DEPENDENCIES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

import ast
import io
import os
import json
import time
import hashlib
import argparse
import tokenize
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import nbformat
from run_all import ensure_dependencies
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent

def inspect_code(source):
    tree = ast.parse(source)
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    comments = sum(token.type == tokenize.COMMENT for token in tokens)
    docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            value = ast.get_docstring(node)
            if value is not None:
                docstrings.append(value)
    return {"comments": comments, "docstrings": len(docstrings), "em_dashes": source.count(chr(8212)), "lines": len(source.splitlines())}

def validate_workflow(item, execute, output, engine):
    started = time.perf_counter()
    script_path = ROOT / item["script"]
    notebook_path = ROOT / item["notebook"]
    source = script_path.read_text(encoding="utf-8")
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    if not all(cell.cell_type == "code" for cell in notebook.cells):
        raise ValueError("non_code_cell")
    notebook_source = "\n\n".join(cell.source for cell in notebook.cells)
    if source != notebook_source:
        raise ValueError("script_notebook_mismatch")
    inspection = inspect_code(source)
    if inspection["comments"] or inspection["docstrings"] or inspection["em_dashes"]:
        raise ValueError("code_only_constraint_violation")
    if inspection["lines"] < 150:
        raise ValueError("workflow_too_short")
    if "DEPENDENCIES" not in notebook.cells[0].source:
        raise ValueError("missing_dependency_setup")
    result = {"workflow": item["workflow"], "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(), "cells": len(notebook.cells), **inspection, "static_validation": "passed", "execution": "not_requested"}
    if execute:
        environment = os.environ.copy()
        environment.update({"FABRIC_STARTER_OUTPUT": str(output / "artifacts"), "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        if engine == "jupyter":
            client = NotebookClient(notebook, timeout=300, kernel_name="python3", allow_errors=False, resources={"metadata": {"path": str(ROOT)}})
            executed = client.execute(env=environment)
            errors = [entry for cell in executed.cells for entry in cell.get("outputs", []) if entry.output_type == "error"]
            if errors:
                raise ValueError("notebook_error_output")
        else:
            log_path = output / (item["workflow"] + ".log")
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, str(ROOT / "tools" / "execute_notebook_cells.py"), str(notebook_path)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            if process.returncode != 0:
                raise RuntimeError(log_path.read_text()[-2500:])
        workflow_runs = sorted((output / "artifacts" / item["workflow"]).glob("*/run_manifest.json"), key=lambda path: path.stat().st_mtime)
        if not workflow_runs:
            raise ValueError("missing_run_manifest")
        manifest = json.loads(workflow_runs[-1].read_text())
        artifact_dir = workflow_runs[-1].parent
        for artifact in manifest["artifacts"]:
            path = artifact_dir / artifact["name"]
            if not path.is_file():
                raise ValueError("missing_output_artifact")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact["sha256"]:
                raise ValueError("output_hash_mismatch")
        result["execution"] = "passed"
        result["output_artifacts"] = len(manifest["artifacts"])
        result["metrics"] = manifest["metrics"]
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-notebooks", action="store_true")
    parser.add_argument("--engine", choices=["ipython", "jupyter"], default="ipython")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "validation_output")
    parser.add_argument("--only", nargs="*", default=[])
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT / "catalog.json").read_text())
    if len(catalog) != 60:
        raise ValueError("expected_sixty_workflows")
    if arguments.only:
        catalog = [item for item in catalog if str(item["id"]) in arguments.only or f'{item["id"]:02d}' in arguments.only]
    if not catalog:
        raise ValueError("no_matching_workflows")
    if arguments.execute_notebooks:
        ensure_dependencies(catalog)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(arguments.workers, 6))) as executor:
        futures = {executor.submit(validate_workflow, item, arguments.execute_notebooks, output, arguments.engine): item for item in catalog}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"workflow": item["workflow"], "execution": "failed", "error_type": type(error).__name__, "error": str(error)}
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "metrics"}), flush=True)
    report = {"python": sys.version.split()[0], "execution_environment": "local_" + arguments.engine, "live_fabric_execution_verified": False, "workflows": len(results), "passed": sum(result.get("execution") != "failed" for result in results), "failed": sum(result.get("execution") == "failed" for result in results), "results": sorted(results, key=lambda item: item["workflow"])}
    (output / "validation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return int(report["failed"] > 0)

if __name__ == "__main__":
    raise SystemExit(main())
