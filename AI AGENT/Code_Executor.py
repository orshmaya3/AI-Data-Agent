"""
Code_Executor.py — Sandboxed Python execution engine for the AI Data Agent.

Features:
- Persistent session namespace: variables defined in one call survive to the next
- Captures stdout, stderr, and return values
- Detects and captures matplotlib figures as base64-encoded PNGs
- Structured logging of every execution (input, output, duration, errors)
- Thread-safe chart buffer per executor instance
- Execution timeout via subprocess isolation (terminates hung/infinite-loop code)
"""

import time
import logging
import pickle
import multiprocessing

# Force non-interactive backend BEFORE any matplotlib import elsewhere
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger("CodeExecutor")

DEFAULT_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------------------------
# Module blocklist — checked via AST before any subprocess is spawned
# ---------------------------------------------------------------------------

_BLOCKED_MODULES = frozenset({
    "os",            # filesystem access, process execution (os.system etc.)
    "subprocess",    # shell command execution
    "shutil",        # filesystem write/delete operations
    "socket",        # raw network access / exfiltration
    "ctypes",        # native code execution, memory manipulation
    "importlib",     # dynamic import — bypasses this blocklist
    "multiprocessing",  # could spawn grandchild procs that outlive proc.terminate()
})


def _check_blocked_imports(code: str) -> str | None:
    """
    Parse *code* with the AST and return an error string if it attempts to
    import a blocked module, or ``None`` if the code looks safe.

    Catches all static import forms::

        import os
        import os, sys
        from os import path
        from os.path import join
        __import__("os")            # dynamic import via builtin
        importlib.import_module(…)  # dynamic import via importlib

    Returns ``None`` on ``SyntaxError`` — let ``exec()`` surface that naturally.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        # import X  /  import X, Y
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    return f"SecurityError: import of '{alias.name}' is not allowed."

        # from X import Y  /  from X.Y import Z
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_MODULES:
                    return f"SecurityError: import of '{node.module}' is not allowed."

        # __import__("os")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "__import__":
                return "SecurityError: __import__() is not allowed."
            # importlib.import_module("os")
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            ):
                return "SecurityError: importlib.import_module() is not allowed."

    return None


# ---------------------------------------------------------------------------
# Chart style — applied once when the module loads
# ---------------------------------------------------------------------------
_STYLE_APPLIED = False

def _apply_chart_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Fall back to matplotlib defaults

    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.autolayout": True,
    })
    _STYLE_APPLIED = True


_apply_chart_style()


# ---------------------------------------------------------------------------
# Subprocess worker — must be module-level for pickle (required on Windows)
# ---------------------------------------------------------------------------

def _subprocess_worker(
    code: str,
    namespace_data: dict,
    result_queue: "multiprocessing.Queue",
) -> None:
    """
    Runs in a child process. Reconstructs the namespace from *namespace_data*,
    executes *code*, captures output/charts, and pushes a result dict onto
    *result_queue*.

    Libraries are re-imported here rather than pickled — only plain data
    (DataFrames, dicts, scalars, etc.) crosses the process boundary.
    """
    import sys, io, traceback, base64 as _b64, pickle as _pickle  # noqa: E401
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import pandas as _pd
    import numpy as _np

    # ---- Rebuild namespace ------------------------------------------------
    namespace: dict = {
        "df":  namespace_data.get("df"),
        "dfs": namespace_data.get("dfs", {}),
        "pd":  _pd,
        "np":  _np,
        "plt": _plt,
    }
    for name, frame in namespace_data.get("dfs", {}).items():
        namespace[name] = frame
    for k, v in namespace_data.get("extra", {}).items():
        namespace[k] = v

    try:
        import seaborn as _sns
        namespace["sns"] = _sns
    except ImportError:
        pass
    try:
        import scipy.stats as _stats
        namespace["stats"] = _stats
    except ImportError:
        pass

    # ---- Execute ----------------------------------------------------------
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    error = ""
    charts: list[str] = []

    _plt.close("all")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    try:
        exec(compile(code, "<analyst_code>", "exec"), namespace)  # noqa: S102
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    output = stdout_buf.getvalue()
    stderr_out = stderr_buf.getvalue().strip()
    if stderr_out:
        output = (output + f"\n[warnings/stderr]:\n{stderr_out}").strip()

    # ---- Capture charts ---------------------------------------------------
    try:
        for fig_num in _plt.get_fignums():
            fig = _plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", bbox_inches="tight",
                dpi=150, facecolor="white", edgecolor="none",
            )
            buf.seek(0)
            charts.append(_b64.b64encode(buf.read()).decode("utf-8"))
        _plt.close("all")
    except Exception:
        pass

    # ---- Extract updated picklable namespace data -------------------------
    _SKIP_LIBS = {"pd", "np", "plt", "sns", "stats"}
    updated_data: dict = {}
    for k, v in namespace.items():
        if k in _SKIP_LIBS or k.startswith("_"):
            continue
        try:
            _pickle.dumps(v)
            updated_data[k] = v
        except Exception:
            pass

    result_queue.put({
        "output":       output.strip(),
        "error":        error,
        "charts":       charts,
        "updated_data": updated_data,
    })


# ---------------------------------------------------------------------------
# CodeExecutor
# ---------------------------------------------------------------------------

class CodeExecutor:
    """
    Persistent Python execution sandbox for one agent session.

    Each call to ``execute()`` runs in a child process so that hung or
    infinite-loop code can be hard-killed after *timeout* seconds without
    taking down the host process.  On success the child's namespace changes
    are merged back into the persistent session namespace.

    Usage::

        executor = CodeExecutor(df)
        result = executor.execute("print(df.shape)")
        charts  = executor.get_pending_charts()   # list of base64 PNG strings
    """

    def __init__(self, df=None, dfs: dict | None = None):
        self._charts: list[str] = []   # base64 PNGs waiting to be sent to UI
        self._namespace: dict = {}
        self._reset_namespace(df, dfs)

    # ------------------------------------------------------------------
    # Namespace management
    # ------------------------------------------------------------------

    def _reset_namespace(self, df, dfs: dict | None = None) -> None:
        """Populate the execution namespace with standard libraries + the dataframe(s).

        ``df``  — the primary dataframe (backward-compatible shorthand).
        ``dfs`` — optional dict of named dataframes, e.g. {"orders": df1, "customers": df2}.
                  Each key is also unpacked into the namespace directly so generated code
                  can reference ``orders`` or ``dfs["orders"]`` interchangeably.
        """
        import pandas as pd
        import numpy as np

        self._namespace = {
            # Data
            "df":  df,
            "dfs": dict(dfs) if dfs else {},
            # Libraries
            "pd":  pd,
            "np":  np,
            "plt": plt,
        }

        # Unpack named dataframes as top-level variables for ergonomic access
        if dfs:
            for name, frame in dfs.items():
                self._namespace[name] = frame
        try:
            import seaborn as sns
            self._namespace["sns"] = sns
        except ImportError:
            pass
        try:
            import scipy.stats as stats
            self._namespace["stats"] = stats
        except ImportError:
            pass

    def update_df(self, df) -> None:
        """Refresh the primary dataframe reference without losing other session variables."""
        self._namespace["df"] = df

    def update_dfs(self, dfs: dict) -> None:
        """Add or replace named dataframes without resetting the session.

        Each key is written both into ``dfs`` dict and as a top-level variable.
        """
        self._namespace["dfs"].update(dfs)
        for name, frame in dfs.items():
            self._namespace[name] = frame

    def _extract_picklable_namespace(self) -> dict:
        """Snapshot the picklable parts of the namespace for subprocess transfer.

        Libraries (pd, np, plt, …) are excluded — they are re-imported inside
        the worker.  Everything else that survives pickle.dumps() is included.
        """
        _SKIP_LIBS = {"pd", "np", "plt", "sns", "stats"}
        extra: dict = {}
        for k, v in self._namespace.items():
            if k in _SKIP_LIBS or k in {"df", "dfs"} or k.startswith("_"):
                continue
            try:
                pickle.dumps(v)
                extra[k] = v
            except Exception:
                pass
        return {
            "df":    self._namespace.get("df"),
            "dfs":   dict(self._namespace.get("dfs", {})),
            "extra": extra,
        }

    def _merge_namespace(self, updated_data: dict) -> None:
        """Write the worker's namespace changes back into the persistent session."""
        _SKIP_LIBS = {"pd", "np", "plt", "sns", "stats"}
        for k, v in updated_data.items():
            if k not in _SKIP_LIBS:
                self._namespace[k] = v
        # Keep top-level dfs variables in sync with the dfs dict
        for name, frame in self._namespace.get("dfs", {}).items():
            self._namespace[name] = frame

    # ------------------------------------------------------------------
    # Chart retrieval
    # ------------------------------------------------------------------

    def get_pending_charts(self) -> list[str]:
        """
        Return all charts generated since the last call, then clear the buffer.
        Each item is a base64-encoded PNG string suitable for st.image() or <img>.
        """
        charts = self._charts.copy()
        self._charts.clear()
        return charts

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def execute(self, code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
        """
        Execute *code* in a child process with a hard timeout.

        The child is terminated if it does not finish within *timeout* seconds.
        On success, namespace changes from the child are merged back into the
        persistent session.  On error or timeout, the session namespace is
        left unchanged (atomic semantics).

        Returns a dict::

            {
                "output":      str,   # everything printed to stdout/stderr
                "error":       str,   # exception message + traceback, or ""
                "charts":      list,  # base64 PNGs produced during this call
                "duration_ms": int,
                "success":     bool,
            }
        """
        _apply_chart_style()

        # ---- Blocklist check (fast, no subprocess needed) -----------------
        block_error = _check_blocked_imports(code)
        if block_error:
            logger.warning(
                "[CodeExecutor] Blocked: %s | code_preview=%r", block_error,
                (code[:80] + "...") if len(code) > 80 else code,
            )
            return {
                "output":      "",
                "error":       block_error,
                "charts":      [],
                "duration_ms": 0,
                "success":     False,
            }

        namespace_data = self._extract_picklable_namespace()
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_subprocess_worker,
            args=(code, namespace_data, result_queue),
            daemon=True,
        )

        start = time.perf_counter()
        proc.start()
        proc.join(timeout=timeout)
        duration_ms = int((time.perf_counter() - start) * 1000)

        # ---- Timeout path -------------------------------------------------
        if proc.is_alive():
            proc.terminate()
            proc.join()
            error = (
                f"TimeoutError: code execution exceeded {timeout}s "
                "and was terminated."
            )
            logger.warning(
                "[CodeExecutor] %s | code_preview=%r", error,
                (code[:80] + "...") if len(code) > 80 else code,
            )
            return {
                "output":      "",
                "error":       error,
                "charts":      [],
                "duration_ms": duration_ms,
                "success":     False,
            }

        # ---- Worker crashed without writing to the queue ------------------
        if result_queue.empty():
            error = (
                "ExecutionError: worker process exited without returning "
                f"a result (exit code {proc.exitcode})."
            )
            logger.warning("[CodeExecutor] %s", error)
            return {
                "output":      "",
                "error":       error,
                "charts":      [],
                "duration_ms": duration_ms,
                "success":     False,
            }

        # ---- Normal path --------------------------------------------------
        result = result_queue.get_nowait()
        charts = result["charts"]
        self._charts.extend(charts)

        # Only merge namespace on clean success to preserve atomic semantics
        if not result["error"]:
            self._merge_namespace(result["updated_data"])

        logger.info(
            "[CodeExecutor] success=%s | %d ms | %d chart(s) | "
            "code_preview=%r | error=%s",
            not bool(result["error"]), duration_ms, len(charts),
            (code[:80] + "...") if len(code) > 80 else code,
            (result["error"][:120] + "...") if len(result["error"]) > 120
            else (result["error"] or "none"),
        )

        return {
            "output":      result["output"],
            "error":       result["error"],
            "charts":      charts,
            "duration_ms": duration_ms,
            "success":     not bool(result["error"]),
        }
