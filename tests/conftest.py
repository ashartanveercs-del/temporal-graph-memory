import pytest
from fakes import MATEO, PRIYA, ROSA, T0

from tgmem.ids import SequentialIds
from tgmem.memory import Memory
from tgmem.timeutil import ManualClock


@pytest.fixture
def clock():
    return ManualClock(T0)


@pytest.fixture
def mem(tmp_path, clock):
    memory = Memory.open(tmp_path / "memory.db", clock=clock, ids=SequentialIds())
    seed(memory)
    yield memory
    memory.close()


def seed(memory: Memory) -> None:
    memory.upsert_entity("shop", "Lakeside Print Shop", kind="org")
    memory.add_contact("Rosa Delgado", [ROSA], role="owner", id="rosa")
    memory.add_contact("Priya Shah", [PRIYA, "priya@example.com"], id="priya")
    memory.add_contact("Mateo Ruiz", [MATEO], id="mateo")
