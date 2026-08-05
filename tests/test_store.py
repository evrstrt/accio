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


def test_updating_one_field_leaves_the_others_alone(tmp_path):
    conn = seeded(tmp_path)
    update_walk_meta(conn, "w1", site="ASHV")
    m = walk_meta(conn, "w1")
    assert m["site"] == "ASHV"
    assert (m["building"], m["operator"], m["mountHeightCm"]) == ("T2", "ram", 178)
    assert m["videoFile"] == "w1.insv"     # not something the edit can reach


def test_a_field_can_be_cleared(tmp_path):
    conn = seeded(tmp_path)
    update_walk_meta(conn, "w1", operator="")
    assert walk_meta(conn, "w1")["operator"] == ""


def test_updating_nothing_is_not_an_error(tmp_path):
    conn = seeded(tmp_path)
    update_walk_meta(conn, "w1")
    assert walk_meta(conn, "w1")["site"] == "GCMR"


def test_updating_a_walk_that_is_not_there_says_so(tmp_path):
    conn = seeded(tmp_path)
    with pytest.raises(KeyError):
        update_walk_meta(conn, "nope", site="X")


def test_unknown_fields_are_refused_rather_than_ignored(tmp_path):
    conn = seeded(tmp_path)
    with pytest.raises(ValueError, match="video_file"):
        update_walk_meta(conn, "w1", video_file="elsewhere.insv")


def test_forgetting_a_walk_takes_its_decisions_with_it(tmp_path):
    conn = seeded(tmp_path)
    save_walk_meta(conn, "w2", "w2.insv", site="ASHV")
    log_decision(conn, "w1", A, "drop")
    log_decision(conn, "w2", B, "drop")

    forget_walk(conn, "w1")

    assert walk_meta(conn, "w1") is None
    assert effective_state(conn, "w1") == {}
    # the other walk is untouched
    assert walk_meta(conn, "w2")["site"] == "ASHV"
    assert [d["walkId"] for d in override_log(conn)] == ["w2"]


def test_forgetting_a_walk_that_was_never_there_is_not_an_error(tmp_path):
    forget_walk(make_conn(tmp_path), "nope")
