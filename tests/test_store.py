import sqlite3

import pytest

from accio.store.db import (connect, effective_state, forget_walk, log_decision,
                            override_log, save_walk_meta, update_walk_meta,
                            walk_meta)


def make_conn(tmp_path):
    return connect(tmp_path / "accio.db")


A, B, C = "y045_00005.jpg", "y135_00009.jpg", "y225_00012.jpg"


def test_pick_override_and_drop(tmp_path):
    conn = make_conn(tmp_path)
    log_decision(conn, "walk1", anchor=A, action="pick", pick=B)
    log_decision(conn, "walk1", anchor=C, action="drop")
    state = effective_state(conn, "walk1")
    assert state[A] == {"pick": B, "dropped": False}
    assert state[C] == {"pick": None, "dropped": True}


def test_latest_decision_wins_and_history_is_kept(tmp_path):
    conn = make_conn(tmp_path)
    log_decision(conn, "w", A, "pick", B)
    log_decision(conn, "w", A, "pick", C)
    log_decision(conn, "w", A, "drop")
    log_decision(conn, "w", A, "restore")
    state = effective_state(conn, "w")
    assert state[A] == {"pick": C, "dropped": False}
    assert [d["action"] for d in override_log(conn, "w")] == [
        "pick", "pick", "drop", "restore"]


def test_pick_on_dropped_group_restores_it(tmp_path):
    conn = make_conn(tmp_path)
    log_decision(conn, "w", A, "drop")
    log_decision(conn, "w", A, "pick", B)
    assert effective_state(conn, "w")[A] == {"pick": B, "dropped": False}


def test_walks_are_isolated(tmp_path):
    conn = make_conn(tmp_path)
    log_decision(conn, "a", A, "drop")
    assert effective_state(conn, "b") == {}
    assert len(override_log(conn)) == 1


def test_walk_meta_roundtrip_and_upsert(tmp_path):
    conn = make_conn(tmp_path)
    save_walk_meta(conn, "w1", "w1.insv", site="GCMR", building="T2",
                   stage="bare-rcc", operator="ram", mount_height_cm=178,
                   shot_date="2026-07-01")
    m = walk_meta(conn, "w1")
    assert m["building"] == "T2" and m["mountHeightCm"] == 178
    save_walk_meta(conn, "w1", "w1.insv", building="T3")
    assert walk_meta(conn, "w1")["building"] == "T3"
    assert walk_meta(conn, "nope") is None


def seeded(tmp_path):
    conn = make_conn(tmp_path)
    save_walk_meta(conn, "w1", "w1.insv", site="GCMR", building="T2",
                   stage="bare-rcc", operator="ram", mount_height_cm=178,
                   shot_date="2026-07-01")
    return conn


def test_update_one_field(tmp_path):
    conn = seeded(tmp_path)
    update_walk_meta(conn, "w1", site="ASHV")
    m = walk_meta(conn, "w1")
    assert m["site"] == "ASHV"
    assert (m["building"], m["operator"], m["mountHeightCm"]) == ("T2", "ram", 178)
    assert m["videoFile"] == "w1.insv"


def test_update_clears_field(tmp_path):
    conn = seeded(tmp_path)
    update_walk_meta(conn, "w1", operator="")
    assert walk_meta(conn, "w1")["operator"] == ""


def test_update_nothing_noop(tmp_path):
    conn = seeded(tmp_path)
    update_walk_meta(conn, "w1")
    assert walk_meta(conn, "w1")["site"] == "GCMR"


def test_update_missing_walk_raises(tmp_path):
    conn = seeded(tmp_path)
    with pytest.raises(KeyError):
        update_walk_meta(conn, "nope", site="X")


def test_update_unknown_field_refused(tmp_path):
    conn = seeded(tmp_path)
    with pytest.raises(ValueError, match="video_file"):
        update_walk_meta(conn, "w1", video_file="elsewhere.insv")


def test_forget_walk_drops_decisions(tmp_path):
    conn = seeded(tmp_path)
    save_walk_meta(conn, "w2", "w2.insv", site="ASHV")
    log_decision(conn, "w1", A, "drop")
    log_decision(conn, "w2", B, "drop")

    forget_walk(conn, "w1")

    assert walk_meta(conn, "w1") is None
    assert effective_state(conn, "w1") == {}
    assert walk_meta(conn, "w2")["site"] == "ASHV"
    assert [d["walkId"] for d in override_log(conn)] == ["w2"]


def test_forget_missing_walk_noop(tmp_path):
    forget_walk(make_conn(tmp_path), "nope")


def test_decisions_index_covers_replay(tmp_path):
    """(walk_id, anchor, id) leaves ORDER BY id to a temp b-tree."""
    conn = make_conn(tmp_path)
    names = {r[1] for r in conn.execute("PRAGMA index_list(decisions)")}
    assert names == {"ix_decisions_walk"}
    plan = " ".join(r[3] for r in conn.execute(
        "EXPLAIN QUERY PLAN SELECT anchor, action, pick FROM decisions "
        "WHERE walk_id = ? ORDER BY id", ("w",)))
    assert "ix_decisions_walk" in plan and "TEMP B-TREE" not in plan


def test_stale_index_dropped_on_connect(tmp_path):
    raw = sqlite3.connect(tmp_path / "accio.db")
    raw.executescript("""
        CREATE TABLE decisions (id INTEGER PRIMARY KEY, walk_id TEXT NOT NULL,
            anchor TEXT NOT NULL, action TEXT NOT NULL, pick TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE INDEX ix_decisions_anchor ON decisions (walk_id, anchor, id);
    """)
    raw.close()
    conn = make_conn(tmp_path)
    names = {r[1] for r in conn.execute("PRAGMA index_list(decisions)")}
    assert names == {"ix_decisions_walk"}
