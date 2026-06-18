"""Maintenance helpers for the sessions the Copilot CLI persists per call.

Every ``copilot -p`` invocation writes one session into a ``session-store.db``
(and a ``session-state/<id>`` folder) under its ``COPILOT_HOME``. A full
benchmark therefore creates ~130 throwaway sessions. Going forward
``CopilotCliClient`` isolates these in a dedicated home so they never reach the
user's real ``~/.copilot`` resume list -- but runs from *before* that isolation
left their sessions in the real store. These helpers prune them safely.

Two cleanup paths:

* :func:`purge_home` -- delete an *isolated* benchmark home wholesale (the
  simplest, zero-risk option once isolation is in use).
* :func:`clean_sessions` -- surgically remove only the benchmark sessions (matched
  by their ``cwd``) from a shared store, backing the database up first and never
  touching sessions created from any other working directory.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from dataclasses import dataclass

# Child tables that carry a ``session_id`` foreign key (verified via PRAGMA).
_CHILD_TABLES = (
    "turns",
    "checkpoints",
    "session_files",
    "session_refs",
    "forge_trajectory_events",
)


def default_real_home() -> str:
    return os.environ.get("COPILOT_HOME") or os.path.join(
        os.path.expanduser("~"), ".copilot"
    )


@dataclass(slots=True)
class CleanupResult:
    matched: int
    deleted: int
    backup_path: str | None
    folders_removed: int
    dry_run: bool


def purge_home(home_dir: str) -> bool:
    """Delete an entire isolated benchmark ``COPILOT_HOME`` directory.

    Safe only for a *dedicated* benchmark home (the default
    ``<temp>/model-judging-copilot-home``); never point this at ``~/.copilot``.
    """
    real = os.path.realpath(default_real_home())
    target = os.path.realpath(home_dir)
    if target == real:
        raise ValueError(
            f"Refusing to purge the real Copilot home: {home_dir}. "
            "Use clean_sessions() for the shared store."
        )
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
        return True
    return False


def _existing_child_tables(con: sqlite3.Connection) -> list[str]:
    have = {
        row[0]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    return [t for t in _CHILD_TABLES if t in have]


def clean_sessions(
    *,
    home_dir: str | None = None,
    cwd_match: str,
    dry_run: bool = False,
    backup: bool = True,
) -> CleanupResult:
    """Remove sessions whose ``cwd`` equals ``cwd_match`` from a store.

    Parameters
    ----------
    home_dir:
        Copilot home holding ``session-store.db`` (default: the real ``~/.copilot``).
    cwd_match:
        Only sessions created with exactly this working directory are removed.
        The benchmark runs every ``copilot`` call from the system temp dir, so
        passing that path targets benchmark sessions and nothing the user opened
        from a project directory.
    dry_run:
        Count matches but change nothing.
    backup:
        Copy the database to ``session-store.db.bak-<timestamp>`` before deleting
        (recommended -- the store may be in use by a live CLI session).
    """
    home_dir = home_dir or default_real_home()
    db_path = os.path.join(home_dir, "session-store.db")
    if not os.path.exists(db_path):
        return CleanupResult(0, 0, None, 0, dry_run)

    con = sqlite3.connect(db_path, timeout=10.0)
    try:
        con.execute("PRAGMA busy_timeout=10000")
        ids = [
            row[0]
            for row in con.execute(
                "SELECT id FROM sessions WHERE cwd = ?", (cwd_match,)
            )
        ]
        matched = len(ids)
        if dry_run or not ids:
            return CleanupResult(matched, 0, None, 0, dry_run)

        backup_path = None
        if backup:
            backup_path = f"{db_path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
            # Use SQLite's online backup so a live writer can't corrupt the copy.
            with sqlite3.connect(backup_path) as bak:
                con.backup(bak)

        placeholders = ",".join("?" * len(ids))
        for table in _existing_child_tables(con):
            con.execute(
                f"DELETE FROM {table} WHERE session_id IN ({placeholders})", ids
            )
        con.execute(
            f"DELETE FROM sessions WHERE id IN ({placeholders})", ids
        )
        con.commit()
        deleted = matched
    finally:
        con.close()

    # Remove the on-disk session-state folders too.
    folders_removed = 0
    state_dir = os.path.join(home_dir, "session-state")
    for sid in ids:
        folder = os.path.join(state_dir, sid)
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
            folders_removed += 1

    return CleanupResult(matched, deleted, backup_path, folders_removed, dry_run)
