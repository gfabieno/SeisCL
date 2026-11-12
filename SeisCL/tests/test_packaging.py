"""setup.py must list every subpackage of SeisCL that users import (no GPU needed).

A subpackage missing from `packages=` is silently left out of a normal (non-editable) install: until
2026-11-12, `SeisCL.analytical` was missing, so code importing the analytical solutions worked from a clone but
failed after `pip install .`. The test suite folder (`SeisCL/tests`) is not installed on purpose.

Run with:
    python SeisCL/tests/test_packaging.py
"""
import ast
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NOT_INSTALLED = {"SeisCL.tests"}


def listed_packages():
    tree = ast.parse(open(os.path.join(ROOT, "setup.py")).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "packages":
            return set(ast.literal_eval(node.value))
    raise AssertionError("no packages= in setup.py")


def found_packages():
    found = set()
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "SeisCL")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if "__init__.py" in filenames:
            found.add(os.path.relpath(dirpath, ROOT).replace(os.sep, "."))
    return found


def test_every_subpackage_is_listed():
    missing = found_packages() - NOT_INSTALLED - listed_packages()
    assert not missing, "subpackages missing from setup.py packages=: %s" % sorted(missing)


if __name__ == "__main__":
    try:
        test_every_subpackage_is_listed()
    except AssertionError as e:
        print("FAIL test_every_subpackage_is_listed:", e)
        sys.exit(1)
    print("PASS test_every_subpackage_is_listed (%s)" % sorted(listed_packages()))
