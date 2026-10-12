from datetime import datetime, timedelta

import pytest

from tgmem.model import (
    ACTIVE,
    CORRECTED,
    EXPIRED,
    INTERNAL,
    PUBLIC,
    SUPERSEDED,
    Fact,
    Viewer,
    normalize_predicate,
    normalize_value,
    private_scope,
    scope_audience,
    validate_scope,
)
from tgmem.timeutil import UTC

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def make_fact(**overrides) -> Fact:
    fields = dict(id="f-1", subject="shop", predicate="opening_hours", value="9-5", object=None,
                  scope=PUBLIC, valid_from=T0, valid_to=None, recorded_at=T0, asserted_by="owner")
    fields.update(overrides)
    return Fact(**fields)


def test_scopes_visible_to_each_kind_of_viewer():
    assert Viewer.owner("rosa").scopes() is None  # everything
    assert Viewer.contact("priya").scopes() == (PUBLIC, "private:priya")
    assert Viewer.anonymous().scopes() == (PUBLIC,)


def test_can_see_follows_scopes():
    priya, mateo = Viewer.contact("priya"), Viewer.contact("mateo")
    assert priya.can_see("private:priya")
    assert not priya.can_see("private:mateo")
    assert not mateo.can_see(INTERNAL)
    assert Viewer.owner().can_see("private:mateo")
    assert not Viewer.anonymous().can_see("private:priya")


def test_contact_viewer_requires_a_node():
    with pytest.raises(ValueError):
        Viewer.contact("")


def test_scope_helpers():
    assert private_scope("priya-shah") == "private:priya-shah"
    assert scope_audience("private:priya-shah") == "priya-shah"
    assert scope_audience(PUBLIC) is None
    assert validate_scope("internal") == INTERNAL
    for bad in ("secret", "private:", "private:Bad Id", "PUBLIC"):
        with pytest.raises(ValueError):
            validate_scope(bad)


def test_fact_validity_is_half_open():
    fact = make_fact(valid_to=T0 + timedelta(days=1))
    assert fact.valid_at(T0)
    assert fact.valid_at(T0 + timedelta(hours=23))
    assert not fact.valid_at(T0 + timedelta(days=1))
    assert not fact.valid_at(T0 - timedelta(microseconds=1))
    assert make_fact().valid_at(T0 + timedelta(days=10_000))


def test_believed_statuses():
    assert make_fact(status=ACTIVE).believed
    assert make_fact(status=EXPIRED).believed
    assert not make_fact(status=SUPERSEDED).believed
    assert not make_fact(status=CORRECTED).believed


def test_fact_key_groups_by_subject_predicate_scope_and_claimant():
    assert make_fact().key == ("shop", "opening_hours", PUBLIC, None)
    claim = make_fact(subject="priya", scope="private:priya", asserted_by="priya")
    assert claim.claimant == "priya"
    assert claim.key == ("priya", "opening_hours", "private:priya", "priya")
    assert make_fact(subject="priya", scope="private:priya").claimant is None  # the owner's
    assert make_fact(scope="private:priya", asserted_by="mateo").claimant is None
    assert make_fact(asserted_by="priya").claimant is None  # public: only the owner publishes


def test_fact_rejects_value_and_object_together_or_neither():
    with pytest.raises(ValueError):
        make_fact(value="x", object="acme")
    with pytest.raises(ValueError):
        make_fact(value=None, object=None)


def test_fact_rejects_empty_or_inverted_intervals():
    with pytest.raises(ValueError):
        make_fact(valid_to=T0)
    with pytest.raises(ValueError):
        make_fact(valid_to=T0 - timedelta(seconds=1))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Opening Hours", "opening_hours"), ("prefers-finish", "prefers_finish"),
     ("  budget ", "budget"), ("Due In (Days)", "due_in_days")],
)
def test_normalize_predicate(raw, expected):
    assert normalize_predicate(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", "123abc", "!!!"])
def test_normalize_predicate_rejects_unusable_names(raw):
    with pytest.raises(ValueError):
        normalize_predicate(raw)


def test_normalize_value_cleans_whitespace_and_control_characters():
    assert normalize_value("  9am-6pm\r\nMon-Sat \x00") == "9am-6pm\nMon-Sat"
    assert normalize_value("tab\tkept") == "tab\tkept"
    with pytest.raises(ValueError):
        normalize_value("   ")
    with pytest.raises(ValueError):
        normalize_value("x" * 50_001)
