"""SQLite store: review decisions layered over immutable pipeline output.

The manifest.csv a walk's pipeline run produces is never edited. This store
holds what humans decided on top of it: pick overrides (swap the auto-pick
for another member of the same duplicate group) and dropped groups. Resetting
a walk is deleting its decision rows; the auto-picks come back untouched.

Every decision is logged, not just its latest state: a pattern in overrides
is evidence the auto-pick rule needs changing (see the labelling doc, item 4).
"""

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS walks (
    walk_id TEXT PRIMARY KEY,
    video_file TEXT NOT NULL,
    site TEXT NOT NULL DEFAULT '',
    building TEXT NOT NULL DEFAULT '',
    floor TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL DEFAULT '',
    operator TEXT NOT NULL DEFAULT '',
    mount_height_cm INTEGER,
    shot_date TEXT NOT NULL DEFAULT '',
    shot_time TEXT NOT NULL DEFAULT '',
    camera TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    walk_id TEXT NOT NULL,
    anchor TEXT NOT NULL,            -- the group's anchor, by face file name
    action TEXT NOT NULL CHECK (action IN ('pick', 'drop', 'restore')),
    pick TEXT,                       -- for 'pick': the member now chosen
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_decisions_anchor ON decisions (walk_id, anchor, id);
"""


# Connections are per thread and the server opens one lazily on each thread's
# first request, so a redeploy that adds a column used to race itself: two
# threads read the same PRAGMA table_info, both decided to ALTER, and the
# second died with "duplicate column name" as a bare 500. The check-then-act
# below is only sound one connect at a time.
_migrate_lock = threading.Lock()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    with _migrate_lock:
        _retire_index_decisions(conn)
        conn.executescript(SCHEMA)
        _add_missing_walk_columns(conn)
    return conn


def _add_missing_walk_columns(conn: sqlite3.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS does not widen a table that already exists,
    so fields added later are added here rather than by rebuilding the row."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(walks)")}
    for col in ("floor", "shot_time", "camera"):
        if col not in have:
            conn.execute(f"ALTER TABLE walks ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
    conn.commit()


def _retire_index_decisions(conn: sqlite3.Connection) -> None:
    """Drop a pre-face-name decisions table so the new one can be made.

    Decisions used to key on the manifest row number, which only means
    anything while the manifest never changes. Re-running a stage renumbers
    the rows, so a stored index would silently point at a different frame.
    Those rows are review clicks on walks that predate this, not worth
    translating; deleting them restores the auto-picks.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
    if "anchor_idx" in cols:
        conn.execute("DROP TABLE decisions")
        conn.commit()


WALK_FIELDS = ("site", "building", "floor", "stage", "operator",
               "mount_height_cm", "shot_date", "shot_time", "camera")


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


def update_walk_meta(conn: sqlite3.Connection, walk_id: str, **fields) -> None:
    """Change some of the capture metadata on a walk that already exists.

    Metadata is the one thing about a walk that is corrected rather than
    re-derived: nothing downstream is computed from it, it is copied into the
    EXIF at export. So it updates in place instead of re-running anything.
    """
    unknown = set(fields) - set(WALK_FIELDS)
    if unknown:
        raise ValueError(f"unknown walk fields: {sorted(unknown)}")
    if not fields:
        return
    cur = conn.execute(
        f"UPDATE walks SET {', '.join(f'{c} = ?' for c in fields)} "
        "WHERE walk_id = ?", (*fields.values(), walk_id))
    conn.commit()
    if cur.rowcount == 0:
        raise KeyError(walk_id)


def walk_meta(conn: sqlite3.Connection, walk_id: str) -> dict | None:
    row = conn.execute(
        "SELECT walk_id, video_file, site, building, floor, stage, operator, "
        "mount_height_cm, shot_date, shot_time, camera FROM walks "
        "WHERE walk_id = ?",
        (walk_id,)).fetchone()
    if row is None:
        return None
    cols = ["walkId", "videoFile", "site", "building", "floor", "stage",
            "operator", "mountHeightCm", "shotDate", "shotTime", "camera"]
    return dict(zip(cols, row))


def forget_walk(conn: sqlite3.Connection, walk_id: str) -> None:
    """Drop a walk's metadata and every decision made on it."""
    conn.execute("DELETE FROM decisions WHERE walk_id = ?", (walk_id,))
    conn.execute("DELETE FROM walks WHERE walk_id = ?", (walk_id,))
    conn.commit()


def log_decision(conn: sqlite3.Connection, walk_id: str, anchor: str,
                 action: str, pick: str | None = None) -> None:
    """Anchor and pick are face file names, so a decision keeps meaning the
    same frame after a stage re-runs and renumbers the manifest."""
    conn.execute(
        "INSERT INTO decisions (walk_id, anchor, action, pick) "
        "VALUES (?, ?, ?, ?)",
        (walk_id, anchor, action, pick))
    conn.commit()


def effective_state(conn: sqlite3.Connection, walk_id: str) -> dict[str, dict]:
    """Latest decision per group: {anchor face name: {pick, dropped}}.

    Replays the log in order, so state is always derivable and the full
    history stays queryable for override-pattern analysis.
    """
    state: dict[str, dict] = {}
    rows = conn.execute(
        "SELECT anchor, action, pick FROM decisions "
        "WHERE walk_id = ? ORDER BY id", (walk_id,))
    for anchor, action, pick in rows:
        s = state.setdefault(anchor, {"pick": None, "dropped": False})
        if action == "pick":
            s["pick"] = pick
            s["dropped"] = False
        elif action == "drop":
            s["dropped"] = True
        elif action == "restore":
            s["dropped"] = False
    return state


def override_log(conn: sqlite3.Connection, walk_id: str | None = None) -> list[dict]:
    """Full decision history, for the override-pattern review."""
    q = ("SELECT walk_id, anchor, action, pick, created_at "
         "FROM decisions")
    args: tuple = ()
    if walk_id is not None:
        q += " WHERE walk_id = ?"
        args = (walk_id,)
    cols = ["walkId", "anchor", "action", "pick", "createdAt"]
    return [dict(zip(cols, r)) for r in conn.execute(q + " ORDER BY id", args)]
