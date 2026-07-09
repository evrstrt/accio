"""SQLite store: review decisions layered over immutable pipeline output.

The manifest.csv a walk's pipeline run produces is never edited. This store
holds what humans decided on top of it: pick overrides (swap the auto-pick
for another member of the same duplicate group) and dropped groups. Resetting
a walk is deleting its decision rows; the auto-picks come back untouched.

Every decision is logged, not just its latest state: a pattern in overrides
is evidence the auto-pick rule needs changing (see the labelling doc, item 4).
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS walks (
    walk_id TEXT PRIMARY KEY,
    video_file TEXT NOT NULL,
    site TEXT NOT NULL DEFAULT '',
    building TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL DEFAULT '',
    operator TEXT NOT NULL DEFAULT '',
    mount_height_cm INTEGER,
    shot_date TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    walk_id TEXT NOT NULL,
    anchor_idx INTEGER NOT NULL,     -- the group's auto-pick (manifest face idx)
    action TEXT NOT NULL CHECK (action IN ('pick', 'drop', 'restore')),
    pick_idx INTEGER,                -- for 'pick': the member now chosen
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_decisions_walk ON decisions (walk_id, anchor_idx, id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


WALK_FIELDS = ("site", "building", "stage", "operator", "mount_height_cm",
               "shot_date")


def save_walk_meta(conn: sqlite3.Connection, walk_id: str, video_file: str,
                   **fields) -> None:
    """Insert or update the item-8 capture metadata for a walk."""
    unknown = set(fields) - set(WALK_FIELDS)
    if unknown:
        raise ValueError(f"unknown walk fields: {sorted(unknown)}")
    cols = ["walk_id", "video_file", *fields]
    conn.execute(
        f"INSERT INTO walks ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))}) "
        "ON CONFLICT (walk_id) DO UPDATE SET "
        + ", ".join(f"{c} = excluded.{c}" for c in cols[1:]),
        (walk_id, video_file, *fields.values()))
    conn.commit()


def walk_meta(conn: sqlite3.Connection, walk_id: str) -> dict | None:
    row = conn.execute(
        "SELECT walk_id, video_file, site, building, stage, operator, "
        "mount_height_cm, shot_date FROM walks WHERE walk_id = ?",
        (walk_id,)).fetchone()
    if row is None:
        return None
    cols = ["walkId", "videoFile", "site", "building", "stage", "operator",
            "mountHeightCm", "shotDate"]
    return dict(zip(cols, row))


def log_decision(conn: sqlite3.Connection, walk_id: str, anchor_idx: int,
                 action: str, pick_idx: int | None = None) -> None:
    conn.execute(
        "INSERT INTO decisions (walk_id, anchor_idx, action, pick_idx) "
        "VALUES (?, ?, ?, ?)",
        (walk_id, anchor_idx, action, pick_idx))
    conn.commit()


def effective_state(conn: sqlite3.Connection, walk_id: str) -> dict[int, dict]:
    """Latest decision per group: {anchor_idx: {pick, dropped}}.

    Replays the log in order, so state is always derivable and the full
    history stays queryable for override-pattern analysis.
    """
    state: dict[int, dict] = {}
    rows = conn.execute(
        "SELECT anchor_idx, action, pick_idx FROM decisions "
        "WHERE walk_id = ? ORDER BY id", (walk_id,))
    for anchor_idx, action, pick_idx in rows:
        s = state.setdefault(anchor_idx, {"pick": None, "dropped": False})
        if action == "pick":
            s["pick"] = pick_idx
            s["dropped"] = False
        elif action == "drop":
            s["dropped"] = True
        elif action == "restore":
            s["dropped"] = False
    return state


def override_log(conn: sqlite3.Connection, walk_id: str | None = None) -> list[dict]:
    """Full decision history, for the override-pattern review."""
    q = ("SELECT walk_id, anchor_idx, action, pick_idx, created_at "
         "FROM decisions")
    args: tuple = ()
    if walk_id is not None:
        q += " WHERE walk_id = ?"
        args = (walk_id,)
    cols = ["walkId", "anchorIdx", "action", "pickIdx", "createdAt"]
    return [dict(zip(cols, r)) for r in conn.execute(q + " ORDER BY id", args)]
