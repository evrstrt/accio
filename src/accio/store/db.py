"""SQLite store: walk metadata and review decisions.

The pipeline's manifest.csv is never edited. Decisions (pick overrides and
dropped groups) are logged on top of it, keyed by face file name so they
survive a re-run that renumbers the manifest. Effective state is a replay of
the log.
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
CREATE INDEX IF NOT EXISTS ix_decisions_walk ON decisions (walk_id, id);
"""

# connections are per thread; two threads migrating at once both see the
# column missing and the second ALTER fails
_migrate_lock = threading.Lock()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    with _migrate_lock:
        _retire_index_decisions(conn)
        # (walk_id, anchor, id) served nothing: every read orders by id alone
        conn.execute("DROP INDEX IF EXISTS ix_decisions_anchor")
        conn.executescript(SCHEMA)
        _add_missing_walk_columns(conn)
    return conn


def _add_missing_walk_columns(conn: sqlite3.Connection) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(walks)")}
    for col in ("floor", "shot_time", "camera"):
        if col not in have:
            conn.execute(f"ALTER TABLE walks ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
    conn.commit()


def _retire_index_decisions(conn: sqlite3.Connection) -> None:
    """Drop the old decisions table that keyed on manifest row number.

    Those rows pointed at whatever frame carried that index at the time;
    deleting them restores the auto-picks.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
    if "anchor_idx" in cols:
        conn.execute("DROP TABLE decisions")
        conn.commit()


WALK_FIELDS = ("site", "building", "floor", "stage", "operator",
               "mount_height_cm", "shot_date", "shot_time", "camera")


def _check_fields(fields: dict) -> None:
    unknown = set(fields) - set(WALK_FIELDS)
    if unknown:
        raise ValueError(f"unknown walk fields: {sorted(unknown)}")


def save_walk_meta(conn: sqlite3.Connection, walk_id: str, video_file: str,
                   **fields) -> None:
    _check_fields(fields)
    cols = ["walk_id", "video_file", *fields]
    conn.execute(
        f"INSERT INTO walks ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))}) "
        "ON CONFLICT (walk_id) DO UPDATE SET "
        + ", ".join(f"{c} = excluded.{c}" for c in cols[1:]),
        (walk_id, video_file, *fields.values()))
    conn.commit()


def update_walk_meta(conn: sqlite3.Connection, walk_id: str, **fields) -> None:
    _check_fields(fields)
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
    conn.execute("DELETE FROM decisions WHERE walk_id = ?", (walk_id,))
    conn.execute("DELETE FROM walks WHERE walk_id = ?", (walk_id,))
    conn.commit()


def log_decision(conn: sqlite3.Connection, walk_id: str, anchor: str,
                 action: str, pick: str | None = None) -> None:
    conn.execute(
        "INSERT INTO decisions (walk_id, anchor, action, pick) "
        "VALUES (?, ?, ?, ?)",
        (walk_id, anchor, action, pick))
    conn.commit()


def effective_state(conn: sqlite3.Connection, walk_id: str) -> dict[str, dict]:
    """Latest decision per group: {anchor face name: {pick, dropped}}."""
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
    q = ("SELECT walk_id, anchor, action, pick, created_at "
         "FROM decisions")
    args: tuple = ()
    if walk_id is not None:
        q += " WHERE walk_id = ?"
        args = (walk_id,)
    cols = ["walkId", "anchor", "action", "pick", "createdAt"]
    return [dict(zip(cols, r)) for r in conn.execute(q + " ORDER BY id", args)]
