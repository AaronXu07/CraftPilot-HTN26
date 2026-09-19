"""Subprocess entry point for the `run_script` sandbox.

Reads {"scene": <scene json>, "script": <python>} from stdin, runs the script with a restricted
set of builtins against a SceneEditor, and prints {"ops": [[name, kwargs, message], ...],
"output": <captured prints>, "error": <str|null>} as JSON on stdout. No file or network access:
`open`, `exec`, `eval`, `compile`, `globals`, `getattr` and friends are absent; `__import__` only
resolves the pure-Python helpers already in the namespace (`math`, `random`, `json`, `itertools`), so
`import math` / `from math import pi` work and everything else raises ImportError.
"""
from __future__ import annotations

import io
import itertools
import json
import math
import random
import sys
import traceback

SAFE_MODULES = {"math": math, "random": random, "json": json, "itertools": itertools}

MAX_OPS = 2000


def _safe_builtins(out: io.StringIO):
    import builtins as _b

    allowed = [
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int",
        "isinstance", "len", "list", "map", "max", "min", "next", "pow", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
        "ValueError", "TypeError", "KeyError", "IndexError", "Exception", "RuntimeError", "StopIteration",
        "chr", "ord", "hash", "iter", "callable",
    ]
    safe = {k: getattr(_b, k) for k in allowed if hasattr(_b, k)}

    def _print(*args, sep=" ", end="\n", **_):
        out.write(sep.join(str(a) for a in args) + end)

    safe["print"] = _print
    safe["ImportError"] = ImportError

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002 — builtin signature
        if level == 0 and name in SAFE_MODULES:
            return SAFE_MODULES[name]
        raise ImportError(f"scripts may only import {', '.join(sorted(SAFE_MODULES))} — {name!r} is not available")

    safe["__import__"] = _import
    return safe


def main() -> None:
    raw = sys.stdin.read()
    payload = json.loads(raw)
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from copilot.engine.scene import Scene, SceneEditor, SceneError

    scene = Scene.from_dict(payload["scene"])
    out = io.StringIO()
    ops = []

    def on_op(name, kwargs, msg):
        ops.append([name, json.loads(json.dumps(kwargs, default=_jsonable)), msg])
        if len(ops) > MAX_OPS:
            raise RuntimeError(f"script exceeded {MAX_OPS} ops")

    editor = SceneEditor(scene, on_op=on_op)
    env = {"__builtins__": _safe_builtins(out), "scene": editor, "math": math, "random": random, "json": json}
    error = None
    try:
        code = compile(payload["script"], "<script>", "exec")
        exec(code, env)  # noqa: S102 - sandboxed by restricted builtins + subprocess timeout
    except SceneError as e:
        error = f"{e}"
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(sys.exc_info()[2])
        line = next((f.lineno for f in reversed(tb) if f.filename == "<script>"), None)
        error = f"{type(e).__name__}: {e}" + (f" (line {line})" if line else "")
    sys.stdout.write(json.dumps({"ops": ops, "output": out.getvalue()[-4000:], "error": error}))
    sys.stdout.flush()


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    main()
