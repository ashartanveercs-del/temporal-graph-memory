"""Per-sender scoping: who can read and write what."""

import pytest
from fakes import MATEO, PRIYA, ROSA

from tgmem.memory import Memory, NotFound, ScopeError
from tgmem.model import Viewer
from tgmem.store import HandleConflict
from tgmem.tools import call_tool
from tgmem.vault import rebuild_from_vault


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


def test_a_contact_cannot_overwrite_what_the_owner_recorded_about_them(mem, clock):
    # Memory poisoning through the contact's own scope: the owner's facts about Priya live
    # in private:priya, the same scope her own tool calls write to.
    limit = mem.remember("priya", "credit_limit", "$500 net 30")
    standing = mem.remember("priya", "account_status", "overdue 3 invoices")
    clock.advance(hours=1)
    priya = mem.for_sender(PRIYA)
    raised = call_tool(priya, "memory_remember",
                       {"predicate": "credit_limit", "value": "$50000 net 365"})
    raised_since_2020 = call_tool(priya, "memory_remember",
                                  {"predicate": "credit_limit", "value": "$50000 net 365",
                                   "valid_from": "2020-01-01"})
    backdated = call_tool(priya, "memory_remember",
                          {"predicate": "account_status", "value": "in good standing",
                           "valid_from": "2020-01-01"})
    assert raised["ok"] and raised_since_2020["ok"] and backdated["ok"]
    # Neither superseded nor corrected ("never true"): the owner's rows stand.
    assert [mem.fact(f.id).status for f in (limit, standing)] == ["active", "active"]
    assert sorted((f.value, f.asserted_by) for f in mem.facts("priya", "credit_limit")) == [
        ("$500 net 30", "owner"), ("$50000 net 365", "priya")]
    assert ("overdue 3 invoices", "owner") in {
        (f.value, f.asserted_by) for f in mem.facts("priya", "account_status")}
    # The assistant still gets the owner's values, and sees the claims as Priya's word.
    pack = priya.context("what is my credit limit and account status?")
    assert "credit limit $500 net 30\n" in pack.text + "\n"
    assert "credit limit $50000 net 365 [according to Priya Shah]" in pack.text
    assert "account status overdue 3 invoices\n" in pack.text + "\n"


def test_a_contacts_notes_never_absorb_the_owners(mem):
    note = mem.remember("priya", "note", "prefers matte finish")  # written by the owner
    mem.for_sender(PRIYA).remember("note", "prefers matte finish on cards, letterheads and "
                                           "everything else she orders")
    report = mem.consolidate()
    assert report.merged == [] and report.duplicates == []
    assert mem.fact(note.id).status == "active"


def test_a_contacts_own_claims_still_replace_each_other(mem, clock):
    priya = mem.for_sender(PRIYA)
    first = priya.remember("deadline", "invitations needed by June 1")
    clock.advance(hours=1)
    priya.remember("deadline", "invitations needed by June 8")
    assert mem.fact(first.id).status == "superseded"
    assert [f.value for f in mem.facts("priya", "deadline")] == ["invitations needed by June 8"]


def test_the_owner_settles_a_claim_by_retracting_it(mem, clock):
    claim = mem.for_sender(PRIYA).remember("credit_limit", "$50000 net 365")
    clock.advance(hours=1)
    mem.remember("priya", "credit_limit", "$500 net 30")
    assert mem.fact(claim.id).status == "active"  # an owner write never erases what she said
    with pytest.raises(ValueError, match="claim"):
        mem.correct(claim.id, "$5000 net 30")  # it would move the claim into the owner's lane
    mem.retract(claim.id)
    assert [f.value for f in mem.facts("priya", "credit_limit")] == ["$500 net 30"]


def test_owner_facts_and_claims_side_by_side_survive_the_vault(mem, tmp_path):
    mem.remember("priya", "credit_limit", "$500 net 30")
    mem.for_sender(PRIYA).remember("credit_limit", "$50000 net 365")
    mem.export_vault(tmp_path / "vault")
    rebuild_from_vault(tmp_path / "vault", tmp_path / "copy.db")
    copy = Memory.open(tmp_path / "copy.db")
    assert copy.facts("priya", "credit_limit") == mem.facts("priya", "credit_limit")
    copy.close()


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
