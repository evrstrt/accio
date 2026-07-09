from accio.store.db import connect, effective_state, log_decision, override_log


def make_conn(tmp_path):
    return connect(tmp_path / "accio.db")


def test_pick_override_and_drop(tmp_path):
    conn = make_conn(tmp_path)
    log_decision(conn, "walk1", anchor_idx=5, action="pick", pick_idx=9)
    log_decision(conn, "walk1", anchor_idx=7, action="drop")
    state = effective_state(conn, "walk1")
    assert state[5] == {"pick": 9, "dropped": False}
    assert state[7] == {"pick": None, "dropped": True}


def test_latest_decision_wins_and_history_is_kept(tmp_path):
    conn = make_conn(tmp_path)
    log_decision(conn, "w", 5, "pick", 9)
    log_decision(conn, "w", 5, "pick", 12)
    log_decision(conn, "w", 5, "drop")
    log_decision(conn, "w", 5, "restore")
    state = effective_state(conn, "w")
    assert state[5] == {"pick": 12, "dropped": False}
    assert [d["action"] for d in override_log(conn, "w")] == [
        "pick", "pick", "drop", "restore"]


def test_pick_on_dropped_group_restores_it(tmp_path):
    conn = make_conn(tmp_path)
    log_decision(conn, "w", 3, "drop")
    log_decision(conn, "w", 3, "pick", 4)
    assert effective_state(conn, "w")[3] == {"pick": 4, "dropped": False}


def test_walks_are_isolated(tmp_path):
    conn = make_conn(tmp_path)
    log_decision(conn, "a", 1, "drop")
    assert effective_state(conn, "b") == {}
    assert len(override_log(conn)) == 1


def test_walk_meta_roundtrip_and_upsert(tmp_path):
    conn = make_conn(tmp_path)
    from accio.store.db import save_walk_meta, walk_meta
    save_walk_meta(conn, "w1", "w1.insv", site="GCMR", building="T2",
                   stage="bare-rcc", operator="ram", mount_height_cm=178,
                   shot_date="2026-07-01")
    m = walk_meta(conn, "w1")
    assert m["building"] == "T2" and m["mountHeightCm"] == 178
    save_walk_meta(conn, "w1", "w1.insv", building="T3")
    assert walk_meta(conn, "w1")["building"] == "T3"
    assert walk_meta(conn, "nope") is None
