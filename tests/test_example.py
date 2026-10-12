"""The runnable example is part of the product: it must keep working."""

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replies(tmp_path, lines):
    chat = load("chat_loop")
    clock = chat.ManualClock(chat.DEMO_START)
    memory = chat.build_demo(tmp_path / "demo.db", clock)
    out = io.StringIO()
    chat.run(lines, memory, clock, tmp_path / "vault", out)
    memory.close()
    transcript = out.getvalue().splitlines()
    return transcript, [line.removeprefix("assistant > ") for line in transcript
                        if line.startswith("assistant > ")]


def test_demo_script_shows_every_feature(tmp_path):
    script = (EXAMPLES / "demo_script.txt").read_text(encoding="utf-8").splitlines()
    transcript, said = replies(tmp_path, script)
    assert "9am-5pm Monday to Saturday" in said[0]
    assert said[1].startswith("As I mentioned earlier")  # no verbatim repeat
    assert "480" not in said[2]  # Mateo cannot get Priya's budget
    assert "500 flyers due Friday" in said[3]  # but he can get his own order
    assert said[4].startswith("Saved: Lakeside Print Shop opening hours 10am-6pm")
    assert said[5] == ('Quick correction: earlier I said "9am-5pm Monday to Saturday"; it is '
                       'now "10am-6pm Monday to Saturday".')
    assert said[6].endswith("(private:priya).")
    assert "10am-6pm Monday to Saturday" in said[7]  # a stranger sees public facts
    assert any(line.startswith("-- consolidate: expired") for line in transcript)
    assert (tmp_path / "vault" / "entities" / "priya.md").exists()


def test_a_correction_is_delivered_once(tmp_path):
    _, said = replies(tmp_path, ["priya: opening hours?",
                                 "rosa: remember shop opening_hours = 8am-4pm",
                                 "/wait 1h", "priya: hello", "priya: hello again"])
    assert "Quick correction" in said[2]
    assert "Quick correction" not in said[3]


def test_the_example_runs_as_a_script(tmp_path):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "chat_loop.py"), "--script",
         str(EXAMPLES / "demo_script.txt"), "--db", str(tmp_path / "demo.db")],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "Quick correction" in result.stdout
