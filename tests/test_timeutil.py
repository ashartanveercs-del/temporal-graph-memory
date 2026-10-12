from datetime import datetime, timedelta, timezone

import pytest

from tgmem.timeutil import (
    UTC,
    ManualClock,
    from_db,
    parse_duration,
    parse_time,
    to_db,
    to_text,
)

T0 = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)


def test_db_format_is_fixed_width_utc_with_microseconds():
    plus_two = timezone(timedelta(hours=2))
    assert to_db(T0) == "2026-03-01T09:30:00.000000Z"
    assert to_db(datetime(2026, 3, 1, 11, 30, tzinfo=plus_two)) == "2026-03-01T09:30:00.000000Z"
    assert to_db(T0.replace(microsecond=7)) == "2026-03-01T09:30:00.000007Z"


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError, match="timezone"):
        to_db(datetime(2026, 3, 1))


def test_db_round_trip_preserves_the_instant():
    value = datetime(2026, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
    assert from_db(to_db(value)) == value
    assert from_db(to_db(value)).tzinfo is not None


def test_string_order_matches_time_order_across_offsets():
    zones = [timezone(timedelta(hours=h)) for h in (-8, 0, 5, 14)]
    instants = [T0 + timedelta(minutes=37 * i, microseconds=i) for i in range(40)]
    rendered = [to_db(t.astimezone(zones[i % 4])) for i, t in enumerate(instants)]
    assert sorted(rendered) == [to_db(t) for t in sorted(instants)]


def test_text_format_is_short_when_there_are_no_microseconds():
    assert to_text(T0) == "2026-03-01T09:30:00Z"
    assert to_text(T0.replace(microsecond=120)) == "2026-03-01T09:30:00.000120Z"
    assert parse_time(to_text(T0.replace(microsecond=120))) == T0.replace(microsecond=120)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-01", datetime(2026, 3, 1, tzinfo=UTC)),
        ("2026-03-01T09:30", T0),
        ("2026-03-01T09:30:00Z", T0),
        ("2026-03-01T11:30:00+02:00", T0),
        ("2026-03-01 09:30", T0),
    ],
)
def test_parse_time_accepts_common_forms(text, expected):
    assert parse_time(text) == expected


def test_parse_time_interprets_naive_input_in_the_given_zone():
    minus_five = timezone(timedelta(hours=-5))
    assert parse_time("2026-03-01T04:30", tz=minus_five) == T0


def test_parse_time_supports_now_and_relative_offsets():
    assert parse_time("now", now=T0) == T0
    assert parse_time("+3d", now=T0) == T0 + timedelta(days=3)
    assert parse_time("-90m", now=T0) == T0 - timedelta(minutes=90)


def test_parse_time_rejects_garbage():
    with pytest.raises(ValueError):
        parse_time("next tuesday-ish")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("90m", timedelta(minutes=90)),
        ("36h", timedelta(hours=36)),
        ("7d", timedelta(days=7)),
        ("2w", timedelta(weeks=2)),
        ("45s", timedelta(seconds=45)),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "7", "d7", "-3d", "1.5h", "3y"])
def test_parse_duration_rejects_invalid(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_manual_clock_advances_and_sets():
    clock = ManualClock(T0)
    assert clock.now() == T0
    clock.advance(days=1, hours=2)
    assert clock.now() == T0 + timedelta(days=1, hours=2)
    clock.set(T0)
    assert clock.now() == T0
    with pytest.raises(ValueError):
        ManualClock(datetime(2026, 1, 1))
