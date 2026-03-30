"""Pipeline runner for executing CLI commands from the Streamlit dashboard.

Uses subprocess to run bist CLI commands, avoiding SQLite locking issues
and Rich library conflicts with Streamlit.
"""

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Regex to strip ANSI escape codes from Rich output
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")

# Force unbuffered output + pipe-friendly progress in child processes
_UNBUFFERED_ENV = {
    **os.environ,
    "PYTHONUNBUFFERED": "1",
    "PIPE_MODE": "1",         # Tell fetcher to use print() instead of Rich progress
}


@dataclass
class PipelineResult:
    """Result from a pipeline command execution."""
    success: bool
    command: str
    stdout: str
    stderr: str
    return_code: int


def _clean_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    return _ANSI_RE.sub("", text).strip()


def _build_cmd(
    stage: str,
    ticker: Optional[str] = None,
    prices_only: bool = False,
    dry_run: bool = False,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Build the CLI command list for a pipeline stage."""
    cmd = [sys.executable, "-m", "bist_picker", stage]

    if dry_run:
        cmd.insert(3, "--dry-run")

    if ticker and stage == "fetch":
        cmd.extend(["--ticker", ticker])

    if prices_only and stage == "fetch":
        cmd.append("--prices-only")

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_command(
    stage: str,
    ticker: Optional[str] = None,
    prices_only: bool = False,
    dry_run: bool = False,
    extra_args: Optional[list[str]] = None,
) -> PipelineResult:
    """Run a bist CLI command as a subprocess (blocking).

    Args:
        stage: Pipeline stage name (fetch, clean, score, pick, report, run).
        ticker: Optional ticker filter for fetch.
        prices_only: Only fetch prices (fetch stage only).
        dry_run: Run in dry-run mode (no DB writes).
        extra_args: Additional CLI arguments.

    Returns:
        PipelineResult with stdout, stderr, success flag.
    """
    cmd = _build_cmd(stage, ticker, prices_only, dry_run, extra_args)
    cmd_str = " ".join(cmd)
    logger.info("Running pipeline command: %s", cmd_str)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(PROJECT_ROOT),
            env=_UNBUFFERED_ENV,
        )

        return PipelineResult(
            success=result.returncode == 0,
            command=cmd_str,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return PipelineResult(
            success=False,
            command=cmd_str,
            stdout="",
            stderr="Command timed out after 10 minutes.",
            return_code=-1,
        )
    except Exception as e:
        return PipelineResult(
            success=False,
            command=cmd_str,
            stdout="",
            stderr=str(e),
            return_code=-1,
        )


def stream_command(
    stage: str,
    ticker: Optional[str] = None,
    prices_only: bool = False,
    dry_run: bool = False,
    extra_args: Optional[list[str]] = None,
) -> Generator[str, None, PipelineResult]:
    """Run a bist CLI command, yielding stdout+stderr lines in real-time.

    Key differences from run_command:
      - Uses PYTHONUNBUFFERED=1 so child output is not buffered.
      - Merges stderr into stdout so errors appear immediately.
      - Yields each line as it arrives.

    Usage::

        gen = stream_command("fetch")
        try:
            while True:
                line = next(gen)
        except StopIteration as e:
            result = e.value  # PipelineResult

    Yields:
        Each output line as it is produced.

    Returns:
        PipelineResult (accessible via StopIteration.value).
    """
    cmd = _build_cmd(stage, ticker, prices_only, dry_run, extra_args)
    cmd_str = " ".join(cmd)
    logger.info("Streaming pipeline command: %s", cmd_str)

    all_output: list[str] = []

    try:
        # Merge stderr into stdout so we see errors in real-time
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,           # Fully unbuffered at OS level
            env=_UNBUFFERED_ENV,
            cwd=str(PROJECT_ROOT),
        )

        # Read raw bytes and split on both \n and \r
        # Rich progress uses \r to update the same line — we want those too
        buf = ""
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                # EOF — process finished
                if buf.strip():
                    clean = _clean_ansi(buf.strip())
                    if clean:
                        all_output.append(clean)
                        yield clean
                break

            char = chunk.decode("utf-8", errors="replace")

            if char in ("\n", "\r"):
                stripped = _clean_ansi(buf.strip())
                if stripped:
                    all_output.append(stripped)
                    yield stripped
                buf = ""
            else:
                buf += char

        proc.wait(timeout=30)

        return PipelineResult(
            success=proc.returncode == 0,
            command=cmd_str,
            stdout="\n".join(all_output),
            stderr="",
            return_code=proc.returncode,
        )

    except subprocess.TimeoutExpired:
        proc.kill()
        return PipelineResult(
            success=False,
            command=cmd_str,
            stdout="\n".join(all_output),
            stderr="Command timed out.",
            return_code=-1,
        )
    except Exception as e:
        return PipelineResult(
            success=False,
            command=cmd_str,
            stdout="\n".join(all_output),
            stderr=str(e),
            return_code=-1,
        )


# Backtesting was intentionally removed per user request.
# Keep the old pipeline helper disabled so it does not reappear in the
# dashboard surface unless the user explicitly asks for it again.
#
# def run_backtest(
#     start_date: Optional[str] = None,
#     end_date: Optional[str] = None,
#     initial_capital: Optional[float] = None,
# ) -> PipelineResult:
#     ...
