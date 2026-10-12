"""Per-sender scoping: who can read and write what."""

import pytest
from fakes import MATEO, PRIYA, ROSA

from tgmem.memory import Memory, NotFound, ScopeError
from tgmem.model import Viewer
from tgmem.store import HandleConflict


@pytest.fixture
def facts(mem):
    return {
        "hours": mem.remember("shop", "opening_hours", "9am-5pm Mon-Sat"),
        "margin": mem.remember("shop", "supplier_margin", "40% on paper", scope="internal"),
        "priya_budget": mem.remember("priya", "budget", "$480 for wedding invitations"),
        "mateo_order": mem.remember("mateo", "order", "500 flyers, due Friday"),
    }


def ids(found):
    return sorted(f.id for f in found)


def test_senders_resolve_to_viewers(mem):
    assert mem.viewer_for(ROSA) == Viewer.owner("rosa")
    assert mem.viewer_for("+1 (555) 000-0101") == Viewer.contact("priya")
    assert mem.viewer_for("PRIYA@example.com") == Viewer.contact("priya")
    assert mem.viewer_for("whatsapp:+15550000102") == Viewer.contact("mateo")
    assert mem.viewer_for("+1 555 000 0199") == Viewer.anonymous()
    assert mem.viewer_for("not a number") == Viewer.anonymous()
    assert mem.viewer_for("555 000 0101") == Viewer.anonymous()  # national form, no default


def test_national_numbers_resolve_with_a_default_country_code(tmp_path):
    memory = Memory.open(tmp_path / "m.db", default_country_code="1")
    memory.add_contact("Priya Shah", ["555 000 0101"], id="priya")
    assert memory.viewer_for("+15550000101") == Viewer.contact("priya")
    assert memory.viewer_for("(555) 000-0101") == Viewer.contact("priya")
    memory.close()


def test_contacts_need_usable_and_unique_handles(mem):
    with pytest.raises(ValueError):
        mem.add_contact("Nobody", ["call me maybe"])
    with pytest.raises(HandleConflict):
        mem.add_contact("Impostor", [PRIYA])
    with pytest.raises(ValueError):
        mem.add_contact("Bad Role", ["+15550000177"], role="admin")
    assert mem.viewer_for(PRIYA) == Viewer.contact("priya")


def test_each_viewer_sees_only_their_scopes(mem, facts):
    everything = ids(facts.values())
    assert ids(mem.for_sender(ROSA).facts()) == everything
    assert ids(mem.for_sender(PRIYA).facts()) == ids([facts["hours"], facts["priya_budget"]])
    assert ids(mem.for_sender(MATEO).facts()) == ids([facts["hours"], facts["mateo_order"]])
    assert ids(mem.for_sender("+15550000199").facts()) == ids([facts["hours"]])


def test_contacts_can_only_write_facts_private_to_themselves(mem, facts):
    priya = mem.for_sender(PRIYA)
    note = priya.remember("preferred_contact_time", "after 6pm")
    assert (note.subject, note.scope, note.asserted_by) == ("priya", "private:priya", "priya")
    for scope in ("public", "internal", "private:mateo"):
        with pytest.raises(ScopeError):
            priya.remember("opening_hours", "closed tomorrow", subject="shop", scope=scope)


def test_unknown_senders_cannot_write(mem):
    with pytest.raises(ScopeError):
        mem.for_sender("+15550000199").remember("note", "hello")


def test_a_contact_cannot_overwrite_shared_knowledge(mem, facts):
    # Memory poisoning attempt: Priya "corrects" the opening hours.
    claim = mem.for_sender(PRIYA).remember("opening_hours", "closed all week", subject="shop")
    assert claim.scope == "private:priya"
    assert mem.fact(facts["hours"].id).status == "active"
    assert [f.value for f in mem.for_sender(MATEO).facts("shop", "opening_hours")] == [
        "9am-5pm Mon-Sat"]
    assert sorted(f.value for f in mem.for_sender(PRIYA).facts("shop", "opening_hours")) == [
        "9am-5pm Mon-Sat", "closed all week"]


def test_owner_writes_through_a_scoped_view(mem):
    owner = mem.for_sender(ROSA)
    hours = owner.remember("opening_hours", "9am-5pm", subject="shop")
    budget = owner.remember("budget", "$480", subject="priya")
    assert (hours.scope, hours.asserted_by) == ("public", "rosa")
    assert budget.scope == "private:priya"
    with pytest.raises(ValueError):
        owner.remember("budget", "$500")  # the owner must say what the fact is about


def test_hidden_facts_look_exactly_like_missing_ones(mem, facts):
    mateo = mem.for_sender(MATEO)
    with pytest.raises(NotFound) as hidden:
        mateo.fact(facts["priya_budget"].id)
    with pytest.raises(NotFound) as missing:
        mateo.fact("f-999999")
    assert str(hidden.value).replace(facts["priya_budget"].id, "ID") == str(
        missing.value).replace("f-999999", "ID")
    assert mateo.fact(facts["hours"].id) == facts["hours"]


def test_scoped_point_in_time_reads(mem, clock, facts):
    clock.advance(hours=1)
    mem.remember("shop", "opening_hours", "10am-4pm")
    priya = mem.for_sender(PRIYA)
    assert [f.value for f in priya.facts("shop", "opening_hours")] == ["10am-4pm"]
    assert [f.value for f in priya.facts("shop", "opening_hours",
                                         as_of=facts["hours"].valid_from)] == ["9am-5pm Mon-Sat"]
