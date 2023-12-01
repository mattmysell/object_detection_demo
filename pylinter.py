#!/usr/bin/env python3
"""
Code for linting the python files in a project including all sub-folders.

To be run from inside the objection_detection_demo_linter_local docker container.
"""
## Standard Libraries
from fnmatch import fnmatch
from os import walk, PathLike
from os.path import dirname, join, realpath
from typing import List

## Installed Libraries
from pylint import lint

## Local Files

def get_py_files() -> List[PathLike]:
    """
    Get all python files in the project.
    """
    this_dir = dirname(realpath(__file__))

    file_paths = []
    for root, _, files in walk(this_dir):
        for file_name in files:
            if fnmatch(file_name, "*.py"):
                file_paths.append(join(this_dir, root, file_name))
                print(f"    {join(root.replace(this_dir, ''), file_name)}")
    return file_paths

def run_pylinting():
    """
    Run pylint on all files in this project.
    """
    totals = {
        "statement": 0,
        "info": 0,
        "convention": 0,
        "refactor": 0,
        "warning": 0,
        "error": 0,
        "fatal": 0
    }

    print("Running pylint check on the following files:")
    file_paths = get_py_files()

    run = lint.Run(["--rcfile=.pylintrc"] + file_paths, exit=False)
    for _, module in run.linter.stats.by_module.items():
        for code in totals:
            totals[code] += module[code]

    print(f"Total statements in  code:  {totals['statement']}")
    print(f"Total info       messages:  {totals['info']}")
    print(f"Total convention messages:  {totals['convention']}")
    print(f"Total refactor   messages:  {totals['refactor']}")
    print(f"Total warning    messages:  {totals['warning']}")
    print(f"Total error      messages:  {totals['error']}")
    print(f"Total fatal      messages:  {totals['fatal']}\n")

if __name__ == "__main__":
    run_pylinting()
