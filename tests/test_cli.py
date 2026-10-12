"""The tgmem command line, run in-process against a temporary database and vault."""

import json
import subprocess
import sys

import pytest

from tgmem.cli import main


@pytest.fixture
def env(tmp_path):
    return {"TGMEM_DB": str(tmp_path / "memory.db"), "TGMEM_VAULT": str(tmp_path / "vault"),
            "TGMEM_EMBEDDER": "hash"}


@pytest.fixture
def run(env, capsys):
    def invoke(*args, expect=0):
        code = main(list(args), env=env)
        out, err = capsys.readouterr()
        assert code == expect, (out, err)
        return out if expect == 0 else out + err
    return invoke


@pytest.fixture
def shop(run):
    run("init")
    run("add-contact", "Rosa Delgado", "+1 555 000 0100", "--id", "rosa", "--owner")
    run("add-contact", "Priya Shah", "+1 555 000 0101", "priya@example.com", "--id", "priya")
    run("add-contact", "Mateo Ruiz", "+1 555 000 0102", "--id", "mateo")
    run("add-entity", "Lakeside Print Shop", "--id", "shop", "--kind", "org")
    hours = run("remember", "shop", "opening_hours", "9am-5pm Monday to Saturday").split()[0]
    budget = run("remember", "priya", "budget", "$480 for wedding invitations").split()[0]
    return {"hours": hours, "budget": budget}


def test_init_and_stats(run):
    assert "initialised" in run("init")
    stats = json.loads(run("--json", "stats"))
    assert stats["facts"] == 0 and stats["embedder"] == "hashing-v1-256"


def test_entities_and_contacts(run):
    run("init")
    assert run("add-entity", "Acme Logistics", "--kind", "org", "--alias", "Acme").strip() == (
        "acme-logistics")
    assert run("add-contact", "Priya Shah", "+1 555 000 0101").strip() == "priya-shah"
    err = run("add-contact", "Someone", "not a phone", expect=1)
    assert err.startswith("error:")


def test_remember_and_read_back(run, shop):
    line = run("facts", "shop")
    assert shop["hours"] in line and "opening_hours = 9am-5pm Monday to Saturday" in line
    [fact] = json.loads(run("--json", "facts", "priya"))
    assert (fact["id"], fact["scope"], fact["status"]) == (shop["budget"], "private:priya",
                                                           "active")


def test_time_travel_and_history(run, shop):
    run("remember", "shop", "opening_hours", "10am-6pm Monday to Saturday",
        "--from", "2030-01-01")
    assert "9am-5pm" in run("facts", "shop", "--as-of", "2029-12-31")
    assert "10am-6pm" in run("facts", "shop", "--as-of", "2030-06-01")
    statuses = [row["status"] for row in json.loads(run("--json", "history", "shop"))]
    assert sorted(statuses) == ["active", "active", "superseded"]
    assert len(json.loads(run("--json", "timeline", "shop"))) == 2


def test_search_is_scoped_to_the_sender(run, shop):
    owner = json.loads(run("--json", "search", "wedding invitations budget"))
    assert owner["hits"][0]["id"] == shop["budget"]
    mateo = json.loads(run("--json", "search", "wedding invitations budget",
                           "--as", "+1 555 000 0102"))
    assert shop["budget"] not in [hit["id"] for hit in mateo["hits"]]
    as_priya = run("search", "budget", "--as", "priya")  # contact ids work too
    assert shop["budget"] in as_priya


def test_tell_corrections_and_context(run, shop):
    assert "recorded 1" in run("tell", "priya", shop["hours"], "--channel", "sms")
    assert "nothing owed" in run("corrections", "priya")
    run("correct", shop["hours"], "9am-6pm Monday to Saturday")
    owed = json.loads(run("--json", "corrections", "priya"))
    assert [o["reason"] for o in owed] == ["corrected"]
    block = run("context", "+1 555 000 0101", "when are you open?")
    assert "Corrections to make first" in block and "9am-6pm" in block


def test_guard_exit_code_signals_a_leak(run, shop):
    assert run("guard", "+1 555 000 0102", "We open at 9am.").strip() == "ok"
    leak = run("guard", "+1 555 000 0102", "Priya's budget is $480.", expect=1)
    assert shop["budget"] in leak


def test_retract_purge_and_errors(run, shop):
    assert "retracted" in run("retract", shop["hours"])
    assert "--yes" in run("purge", shop["budget"], expect=1)
    assert shop["budget"] in run("purge", shop["budget"], "--yes")
    assert "unknown fact" in run("retract", "f-000000000000", expect=1)


def test_vault_export_check_rebuild(run, shop, env):
    assert "wrote" in run("vault", "export")
    assert "ok" in run("vault", "check")
    run("remember", "shop", "fax", "none")
    assert "not in the vault" in run("vault", "rebuild", expect=1)
    assert "rebuilt" in run("vault", "rebuild", "--force")
    assert "fax" not in run("facts", "shop")


def test_consolidate_and_reindex(run, shop):
    run("define", "note", "many")
    run("remember", "priya", "note", "prefers matte finish")
    run("remember", "priya", "note", "Prefers matte finish.")
    report = json.loads(run("--json", "consolidate", "--dry-run"))
    assert report["dry_run"] is True and len(report["duplicates"]) == 1
    assert "duplicates 1" in run("consolidate", "--export")
    assert "embedded 0" in run("reindex")


def test_env_file_is_read(tmp_path, capsys):
    env_file = tmp_path / "custom.env"
    env_file.write_text(f"TGMEM_DB={tmp_path / 'from-file.db'}\nTGMEM_EMBEDDER=none\n",
                        encoding="utf-8")
    assert main(["--env-file", str(env_file), "init"], env={}) == 0
    assert (tmp_path / "from-file.db").exists()
    capsys.readouterr()


def test_module_entry_point_runs(tmp_path):
    result = subprocess.run([sys.executable, "-m", "tgmem", "--help"], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0
    assert "remember" in result.stdout and "vault" in result.stdout
