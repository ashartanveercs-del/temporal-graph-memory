"""Valid-time behaviour: supersession, corrections, temporary overrides and history."""

from datetime import timedelta

import pytest
from fakes import T0

from tgmem.memory import NotFound

HOUR, DAY = timedelta(hours=1), timedelta(days=1)


def values(facts):
    return [f.value for f in facts]


def test_remember_records_an_active_fact_valid_from_now(mem, clock):
    clock.advance(HOUR)
    fact = mem.remember("shop", "Opening Hours", "9am-5pm Mon-Sat")
    assert fact.id == "f-000001"
    assert (fact.predicate, fact.value, fact.scope, fact.status) == (
        "opening_hours", "9am-5pm Mon-Sat", "public", "active")
    assert fact.valid_from == fact.recorded_at == T0 + HOUR
    assert fact.valid_to is None
    assert fact.text == "Lakeside Print Shop opening hours 9am-5pm Mon-Sat"
    assert mem.facts("shop") == [fact]


def test_remembering_the_same_value_again_changes_nothing(mem, clock):
    first = mem.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(HOUR)
    assert mem.remember("shop", "opening_hours", "  9am-5pm ") == first
    assert mem.history("shop") == [first]


def test_a_new_value_supersedes_the_old_one_without_deleting_it(mem, clock):
    clock.advance(HOUR)
    old = mem.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(10 * DAY)
    changed_at = clock.now()
    new = mem.remember("shop", "opening_hours", "9am-6pm")

    ended = mem.fact(old.id)
    assert (ended.status, ended.superseded_by, ended.ended_at) == (
        "superseded", new.id, changed_at)
    assert ended.value == "9am-5pm" and ended.valid_to is None  # the row itself is untouched

    [remainder] = [f for f in mem.history("shop") if f.derived_from == old.id]
    assert (remainder.value, remainder.valid_from, remainder.valid_to) == (
        "9am-5pm", old.valid_from, changed_at)
    assert (remainder.status, remainder.recorded_at) == ("expired", changed_at)

    assert mem.facts("shop", "opening_hours") == [new]
    assert values(mem.facts("shop", "opening_hours", as_of=T0 + 2 * DAY)) == ["9am-5pm"]


def test_a_scheduled_change_takes_effect_on_its_date(mem, clock):
    clock.advance(HOUR)
    mem.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(DAY)
    effective = clock.now() + 7 * DAY
    mem.remember("shop", "opening_hours", "8am-8pm", valid_from=effective)
    assert values(mem.facts("shop", "opening_hours")) == ["9am-5pm"]
    clock.advance(7 * DAY)
    assert values(mem.facts("shop", "opening_hours")) == ["8am-8pm"]


def test_a_correction_replaces_the_old_value_for_all_of_time(mem, clock):
    clock.advance(HOUR)
    wrong = mem.remember("shop", "card_price", "$0.45 per card")
    clock.advance(2 * HOUR)
    fixed = mem.remember("shop", "card_price", "$0.40 per card", valid_from=wrong.valid_from)
    assert mem.fact(wrong.id).status == "corrected"
    assert mem.fact(wrong.id).superseded_by == fixed.id
    assert not [f for f in mem.history("shop") if f.derived_from == wrong.id]
    for when in (wrong.valid_from, clock.now()):
        assert values(mem.facts("shop", "card_price", as_of=when)) == ["$0.40 per card"]


def test_correct_defaults_to_a_full_correction(mem, clock):
    clock.advance(HOUR)
    wrong = mem.remember("shop", "card_price", "$0.45 per card")
    clock.advance(2 * HOUR)
    fixed = mem.correct(wrong.id, "$0.40 per card")
    assert fixed.valid_from == wrong.valid_from
    assert (mem.fact(wrong.id).status, mem.fact(wrong.id).superseded_by) == (
        "corrected", fixed.id)


def test_correct_with_a_later_start_is_a_change(mem, clock):
    clock.advance(HOUR)
    old = mem.remember("shop", "card_price", "$0.45 per card")
    from_date = old.valid_from + 3 * DAY
    clock.advance(5 * DAY)
    mem.correct(old.id, "$0.50 per card", valid_from=from_date)
    assert mem.fact(old.id).status == "superseded"
    assert values(mem.facts("shop", "card_price", as_of=from_date - HOUR)) == ["$0.45 per card"]
    assert values(mem.facts("shop", "card_price", as_of=from_date)) == ["$0.50 per card"]


def test_a_temporary_override_ends_and_the_regular_value_resumes(mem, clock):
    clock.advance(HOUR)
    regular = mem.remember("shop", "opening_hours", "9am-5pm")
    start, end = T0 + 20 * DAY, T0 + 23 * DAY
    mem.remember("shop", "opening_hours", "closed for the holidays", valid_from=start,
                 valid_to=end)
    assert values(mem.facts("shop", "opening_hours", as_of=start - HOUR)) == ["9am-5pm"]
    assert values(mem.facts("shop", "opening_hours", as_of=start + DAY)) == [
        "closed for the holidays"]
    assert values(mem.facts("shop", "opening_hours", as_of=end)) == ["9am-5pm"]
    assert values(mem.facts("shop", "opening_hours", as_of=end + 300 * DAY)) == ["9am-5pm"]
    pieces = [f for f in mem.history("shop") if f.derived_from == regular.id]
    assert [(p.valid_from, p.valid_to) for p in pieces] == [
        (regular.valid_from, start), (end, None)]


def test_known_at_answers_what_we_believed_back_then(mem, clock):
    clock.advance(HOUR)
    t1 = clock.now()
    mem.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(5 * DAY)
    learned = clock.now()
    mem.remember("shop", "opening_hours", "9am-6pm", valid_from=t1 + 2 * DAY)  # retroactive
    probe = t1 + 3 * DAY
    assert values(mem.facts("shop", "opening_hours", as_of=probe)) == ["9am-6pm"]
    before = learned - timedelta(seconds=1)
    assert values(mem.facts("shop", "opening_hours", as_of=probe, known_at=before)) == [
        "9am-5pm"]


def test_retracting_removes_a_fact_from_every_point_in_time(mem, clock):
    promo = mem.remember("shop", "promo", "20% off business cards")
    clock.advance(HOUR)
    retracted = mem.retract(promo.id)
    assert (retracted.status, retracted.ended_at) == ("retracted", clock.now())
    assert mem.facts("shop", "promo") == []
    assert mem.facts("shop", "promo", as_of=promo.valid_from) == []
    assert [f.status for f in mem.history("shop", "promo")] == ["retracted"]
    with pytest.raises(ValueError):
        mem.retract(promo.id)


def test_many_valued_predicates_accumulate(mem):
    a = mem.remember("priya", "note", "prefers matte finish")
    b = mem.remember("priya", "note", "pays by bank transfer")
    assert mem.remember("priya", "note", "prefers matte finish") == a
    assert sorted(values(mem.facts("priya", "note"))) == sorted([a.value, b.value])
    mem.define_predicate("likes", "many")
    mem.remember("priya", "likes", "linen paper")
    mem.remember("priya", "likes", "letterpress")
    assert len(mem.facts("priya", "likes")) == 2


def test_ttl_sets_the_end_of_validity(mem, clock):
    promo = mem.remember("shop", "promo", "free delivery", ttl=7 * DAY)
    assert promo.valid_to == promo.valid_from + 7 * DAY
    clock.set(promo.valid_to - timedelta(microseconds=1))
    assert mem.facts("shop", "promo") == [promo]
    clock.set(promo.valid_to)  # half-open: gone exactly at valid_to
    assert mem.facts("shop", "promo") == []


def test_facts_entirely_in_the_past_are_recorded_as_expired(mem):
    past = mem.remember("shop", "closure", "closed for inventory",
                        valid_from=T0 - 10 * DAY, valid_to=T0 - 8 * DAY)
    assert past.status == "expired"
    assert past.ended_at == past.recorded_at
    assert values(mem.facts("shop", "closure", as_of=T0 - 9 * DAY)) == ["closed for inventory"]


def test_timeline_shows_belief_and_history_shows_every_row(mem, clock):
    clock.advance(HOUR)
    mem.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(DAY)
    mem.remember("shop", "opening_hours", "9am-6pm")
    assert values(mem.timeline("shop", "opening_hours")) == ["9am-5pm", "9am-6pm"]
    assert [f.status for f in mem.history("shop", "opening_hours")] == [
        "superseded", "expired", "active"]


def test_facts_can_point_at_other_nodes(mem):
    mem.upsert_entity("acme", "Acme Logistics", kind="org")
    works = mem.remember("priya", "works_at", object="acme", scope="public")
    assert works.text == "Priya Shah works at Acme Logistics"
    assert [(f.id, n.id) for f, n in mem.neighbors("acme")] == [(works.id, "priya")]
    assert [(f.id, n.id) for f, n in mem.neighbors("priya")] == [(works.id, "acme")]


def test_renaming_a_node_re_renders_the_text_of_its_facts(mem):
    hours = mem.remember("shop", "opening_hours", "9am-5pm")
    works = mem.remember("priya", "works_at", object="shop", scope="public")
    mem.upsert_entity("shop", "Lakeside Prints", kind="org", aliases=("LPS",))
    assert mem.fact(hours.id).text == "Lakeside Prints (LPS) opening hours 9am-5pm"
    assert mem.fact(works.id).text == "Priya Shah works at Lakeside Prints (LPS)"


def test_facts_about_a_contact_default_to_private(mem):
    assert mem.remember("priya", "budget", "$480").scope == "private:priya"
    assert mem.remember("shop", "opening_hours", "9am-5pm").scope == "public"
    assert mem.remember("priya", "nickname", "P", scope="internal").scope == "internal"


def test_invalid_writes_are_rejected(mem):
    with pytest.raises(NotFound):
        mem.remember("nobody", "colour", "blue")
    with pytest.raises(ValueError):
        mem.remember("shop", "colour", "blue", object="priya")
    with pytest.raises(ValueError):
        mem.remember("shop", "colour")
    with pytest.raises(ValueError):
        mem.remember("shop", "colour", "blue", valid_from=T0, valid_to=T0)
    with pytest.raises(ValueError):
        mem.remember("shop", "colour", "blue", valid_to=T0 + DAY, ttl=DAY)
    with pytest.raises(ValueError):
        mem.remember("shop", "colour", "blue", scope="private:shop")  # not a contact
    with pytest.raises(NotFound):
        mem.remember("shop", "colour", "blue", scope="private:ghost")
    with pytest.raises(ValueError):
        mem.remember("shop", "colour", "blue", source="has spaces")
    assert mem.history("shop") == []


def test_recorded_time_never_goes_backwards(mem, clock):
    clock.advance(5 * HOUR)
    a = mem.remember("shop", "p1", "x")
    clock.set(T0 - 5 * HOUR)  # the wall clock jumps back
    b = mem.remember("shop", "p2", "y")
    assert b.recorded_at > a.recorded_at
    assert mem.now() >= b.recorded_at
    assert mem.facts("shop", "p2") == [b]


def test_purge_erases_a_fact_and_every_row_carrying_its_value(mem, clock):
    clock.advance(HOUR)
    old = mem.remember("shop", "opening_hours", "9am-5pm")
    clock.advance(DAY)
    new = mem.remember("shop", "opening_hours", "9am-6pm")
    [remainder] = [f for f in mem.history("shop") if f.derived_from == old.id]
    assert sorted(mem.purge(remainder.id)) == sorted([old.id, remainder.id])
    assert mem.history("shop", "opening_hours") == [new]
    with pytest.raises(NotFound):
        mem.fact(old.id)


def test_bulk_writes_share_one_transaction(mem):
    with mem.bulk():
        for i in range(50):
            mem.remember("shop", f"attribute_{i}", f"value {i}")
    assert len(mem.facts("shop")) == 50
    with pytest.raises(RuntimeError), mem.bulk():
        mem.remember("shop", "doomed", "never stored")
        raise RuntimeError("abort the batch")
    assert mem.facts("shop", "doomed") == []
