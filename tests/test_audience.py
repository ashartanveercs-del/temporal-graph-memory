"""The audience ledger: what was told to whom, repetition, corrections owed and leaks."""

from datetime import timedelta

import pytest
from fakes import MATEO, PRIYA, ROSA

from tgmem.memory import Memory, NotFound, ScopeError

HOUR, DAY = timedelta(hours=1), timedelta(days=1)


@pytest.fixture
def shop(mem, clock):
    clock.advance(HOUR)
    return {
        "hours": mem.remember("shop", "opening_hours", "9am-5pm Monday to Saturday"),
        "turnaround": mem.remember("shop", "turnaround", "business cards ship in 3 days"),
        "margin": mem.remember("shop", "supplier_margin", "40 percent on paper stock",
                               scope="internal"),
        "budget": mem.remember("priya", "budget", "$480 for wedding invitations"),
        "order": mem.remember("mateo", "order", "500 flyers due Friday"),
    }


# -- the ledger ---------------------------------------------------------------------------

def test_disclosures_are_recorded_per_contact_and_fact(mem, clock, shop):
    clock.advance(HOUR)
    told = mem.record_disclosure("priya", [shop["hours"].id, shop["budget"].id], channel="sms")
    assert [(d.contact, d.fact_id, d.channel) for d in told] == [
        ("priya", shop["hours"].id, "sms"), ("priya", shop["budget"].id, "sms")]
    assert told[0].told_at == told[1].told_at >= clock.now()
    assert mem.told("priya") == told
    assert mem.told("mateo") == []


def test_the_ledger_refuses_cross_person_disclosures(mem, shop):
    with pytest.raises(ScopeError):
        mem.record_disclosure("mateo", [shop["budget"].id])  # Priya's private fact
    with pytest.raises(ScopeError):
        mem.record_disclosure("mateo", [shop["margin"].id])  # internal
    with pytest.raises(ValueError):
        mem.record_disclosure("shop", [shop["hours"].id])  # not a contact
    with pytest.raises(NotFound):
        mem.record_disclosure("priya", ["f-999999"])
    assert mem.told("mateo") == []


def test_told_at_recognises_the_same_value_in_a_newer_row(mem, clock, shop):
    clock.advance(HOUR)
    [told] = mem.record_disclosure("priya", [shop["hours"].id])
    clock.advance(DAY)
    # A change scheduled for next month: today's value is now carried by a remainder row.
    mem.remember("shop", "opening_hours", "8am-8pm", valid_from=clock.now() + 30 * DAY)
    [current] = mem.facts("shop", "opening_hours")
    assert current.id != shop["hours"].id
    assert mem.told_at("priya", current) == told.told_at


def test_repeat_window(mem, clock, shop):
    [told] = mem.record_disclosure("priya", [shop["hours"].id])
    clock.advance(100 * DAY)
    assert mem.told_at("priya", shop["hours"]) == told.told_at
    assert mem.told_at("priya", shop["hours"], since=clock.now() - 90 * DAY) is None


# -- corrections owed -----------------------------------------------------------------------

def test_a_changed_value_is_owed_until_the_new_one_is_told(mem, clock, shop):
    mem.record_disclosure("priya", [shop["hours"].id])
    clock.advance(DAY)
    new = mem.remember("shop", "opening_hours", "10am-6pm Monday to Saturday")
    [owed] = mem.corrections_owed("priya")
    assert (owed.told.id, owed.current.id, owed.reason) == (shop["hours"].id, new.id, "changed")
    assert mem.corrections_owed("mateo") == []  # Mateo was never told the old hours
    mem.record_disclosure("priya", [new.id])
    assert mem.corrections_owed("priya") == []


def test_a_wrong_value_is_owed_as_a_correction(mem, clock, shop):
    mem.record_disclosure("mateo", [shop["turnaround"].id])
    clock.advance(HOUR)
    fixed = mem.correct(shop["turnaround"].id, "business cards ship in 5 days")
    [owed] = mem.corrections_owed("mateo")
    assert (owed.current.id, owed.reason) == (fixed.id, "corrected")


def test_a_retracted_value_is_owed_with_nothing_to_replace_it(mem, clock, shop):
    promo = mem.remember("shop", "promo", "20% off letterpress cards")
    mem.record_disclosure("mateo", [promo.id])
    clock.advance(HOUR)
    mem.retract(promo.id)
    [owed] = mem.corrections_owed("mateo")
    assert (owed.told.id, owed.current, owed.reason) == (promo.id, None, "retracted")


def test_a_scheduled_change_is_not_owed_before_it_takes_effect(mem, clock, shop):
    mem.record_disclosure("priya", [shop["hours"].id])
    effective = clock.now() + 7 * DAY
    mem.remember("shop", "opening_hours", "8am-8pm", valid_from=effective)
    assert mem.corrections_owed("priya") == []
    clock.set(effective)
    assert [o.reason for o in mem.corrections_owed("priya")] == ["changed"]


def test_temporary_overrides_are_owed_only_while_they_last(mem, clock, shop):
    mem.record_disclosure("priya", [shop["hours"].id])
    start, end = clock.now() + 5 * DAY, clock.now() + 8 * DAY
    holiday = mem.remember("shop", "opening_hours", "closed for the holidays",
                           valid_from=start, valid_to=end)
    clock.set(start + DAY)
    [owed] = mem.corrections_owed("priya")
    assert owed.current.id == holiday.id
    clock.set(end + DAY)
    assert mem.corrections_owed("priya") == []  # regular hours are back: nothing to fix


def test_a_value_that_lapsed_naturally_is_not_owed(mem, clock, shop):
    promo = mem.remember("shop", "promo", "free delivery this week", ttl=7 * DAY)
    mem.record_disclosure("mateo", [promo.id])
    clock.advance(10 * DAY)
    mem.remember("shop", "promo", "10% off flyers")
    assert mem.corrections_owed("mateo") == []


def test_many_valued_facts_owe_corrections_for_retractions_only_when_needed(mem, clock):
    a = mem.remember("priya", "note", "prefers matte finish on all cards")
    b = mem.remember("priya", "note", "pays by bank transfer every month")
    mem.record_disclosure("priya", [a.id, b.id])
    clock.advance(HOUR)
    mem.retract(a.id)
    assert [(o.told.id, o.reason) for o in mem.corrections_owed("priya")] == [(a.id, "retracted")]


# -- mentions and the outbound leak guard ---------------------------------------------------

def test_mentions_finds_facts_whose_values_appear_in_a_reply(mem, shop):
    reply = "We're open 9am-5pm, Monday to Saturday, and cards ship in 3 days!"
    found = mem.mentions(reply, [shop["hours"], shop["turnaround"], shop["order"]])
    assert [f.id for f in found] == [shop["hours"].id]  # "ship in 3 days" is not the value


def test_numbers_alone_count_as_mentions(mem, shop):
    assert mem.mentions("Your budget of $480 is noted.", [shop["budget"]]) == [shop["budget"]]


def test_the_guard_flags_another_persons_private_values(mem, shop):
    mateo = mem.for_sender(MATEO)
    assert [f.id for f in mateo.guard("Priya's budget is $480, so yours is fine.")] == [
        shop["budget"].id]
    assert [f.id for f in mateo.guard("Our paper margin is 40 percent on paper stock.")] == [
        shop["margin"].id]
    assert mateo.guard("We're open 9am-5pm Monday to Saturday.") == []
    assert mateo.guard("Your 500 flyers are due Friday.") == []  # his own order


def test_the_guard_ignores_phrases_the_viewer_may_see_elsewhere(mem, shop):
    mem.remember("shop", "flyer_minimum", "minimum order is 500 flyers")
    priya = mem.for_sender(PRIYA)
    # "500" is in Mateo's private order, but the shop's public minimum says it too.
    assert priya.guard("Flyers come in batches of 500.") == []


def test_the_owner_has_nothing_to_leak(mem, shop):
    assert mem.for_sender(ROSA).guard("Priya's budget is $480; margin 40 percent.") == []


# -- the scoped view used by an assistant ---------------------------------------------------

def test_mark_told_hides_invisible_facts_behind_not_found(mem, shop):
    mateo = mem.for_sender(MATEO)
    with pytest.raises(NotFound) as hidden:
        mateo.mark_told([shop["budget"].id])
    assert "unknown fact" in str(hidden.value)
    assert mem.told("mateo") == []
    with pytest.raises(ScopeError):
        mem.for_sender(ROSA).mark_told([shop["hours"].id])
    with pytest.raises(ScopeError):
        mem.for_sender("+15550000199").mark_told([shop["hours"].id])


def test_context_pack_marks_repeats_corrections_and_attribution(mem, clock, shop):
    priya = mem.for_sender(PRIYA)
    priya.remember("deadline", "invitations needed by June 1")
    mem.record_disclosure("priya", [shop["hours"].id])
    clock.advance(DAY)
    mem.remember("shop", "turnaround", "business cards ship in 2 days")
    pack = priya.context("when are you open and how fast do business cards ship?")
    told = {cf.fact.predicate: cf.told_at for cf in pack.facts}
    assert told["opening_hours"] is not None
    assert told["turnaround"] is None
    assert pack.corrections == ()  # she was never told the old turnaround
    assert "already told" in pack.text
    assert "9am-5pm Monday to Saturday" in pack.text
    assert "500 flyers" not in pack.text  # Mateo's order never reaches Priya's prompt
    own = priya.context("what deadline did I give you?")
    assert "according to Priya Shah" in own.text


def test_context_pack_lists_corrections_first(mem, clock, shop):
    mem.record_disclosure("priya", [shop["turnaround"].id])
    clock.advance(DAY)
    mem.correct(shop["turnaround"].id, "business cards ship in 5 days")
    pack = mem.for_sender(PRIYA).context("hello")
    assert [c.reason for c in pack.corrections] == ["corrected"]
    assert pack.text.index("Corrections") < pack.text.index("Facts")


def test_record_reply_marks_what_the_reply_actually_said(mem, shop):
    priya = mem.for_sender(PRIYA)
    pack = priya.context("opening hours and turnaround?")
    recorded = priya.record_reply("We're open 9am-5pm Monday to Saturday.", pack)
    assert [d.fact_id for d in recorded] == [shop["hours"].id]
    again = priya.context("opening hours?")
    assert next(cf for cf in again.facts if cf.fact.id == shop["hours"].id).told_at is not None


def test_the_guard_sees_facts_written_after_its_last_check(mem, shop):
    mateo = mem.for_sender(MATEO)
    assert mateo.guard("Our rush fee is $275 for weekend jobs.") == []
    secret = mem.remember("priya", "rush_fee", "$275 for weekend jobs")
    assert [f.id for f in mateo.guard("Our rush fee is $275 for weekend jobs.")] == [secret.id]


def test_the_guard_sees_facts_written_by_another_process(mem, shop, tmp_path):
    other = Memory.open(mem.store.path)
    try:
        mateo = mem.for_sender(MATEO)
        assert mateo.guard("code 4417 opens the side door") == []
        secret = other.remember("priya", "door_code", "4417")
        assert [f.id for f in mateo.guard("code 4417 opens the side door")] == [secret.id]
    finally:
        other.close()


def test_the_guard_catches_the_start_of_a_very_long_private_value(mem, shop):
    long_value = " ".join(f"clause{i}" for i in range(200))
    contract = mem.remember("priya", "contract_terms", long_value)
    leaked = mem.for_sender(MATEO).guard("FYI: " + " ".join(long_value.split()[:40]))
    assert [f.id for f in leaked] == [contract.id]
