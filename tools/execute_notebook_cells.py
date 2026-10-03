import sys
import site

if site.ENABLE_USER_SITE:
    site.addsitedir(site.getusersitepackages())

import subprocess
import importlib.util

DEPENDENCIES = {"nbformat": "nbformat>=5.10,<6", "IPython": "ipython>=8,<10"}
missing = [requirement for module, requirement in DEPENDENCIES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

import json
import time
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

path = Path(sys.argv[1]).resolve()
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
shell = InteractiveShell.instance()
shell.ast_node_interactivity = "none"
execution = []
for index, cell in enumerate(notebook.cells):
    if cell.cell_type != "code":
        raise ValueError("non_code_cell")
    started = time.perf_counter()
    result = shell.run_cell(cell.source, store_history=True)
    if result.error_before_exec is not None:
        raise result.error_before_exec
    if result.error_in_exec is not None:
        raise result.error_in_exec
    execution.append({"cell": index, "status": "passed", "seconds": round(time.perf_counter() - started, 3)})
print(json.dumps({"notebook": path.name, "engine": "ipython_cells", "cells": execution}, indent=2))
