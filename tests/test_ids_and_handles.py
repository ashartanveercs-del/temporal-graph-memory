import re

import pytest

from tgmem.handles import canonical_handle
from tgmem.ids import RandomIds, SequentialIds, slugify

SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Lakeside Print Shop", "lakeside-print-shop"),
        ("  Priya   Shah ", "priya-shah"),
        ("Mateo Núñez", "mateo-nunez"),
        ("Invoice #1042 (Q3)", "invoice-1042-q3"),
        ("ACME, Inc.", "acme-inc"),
    ],
)
def test_slugify_produces_readable_ascii_ids(name, expected):
    assert slugify(name) == expected


@pytest.mark.parametrize("name", ["CON", "nul", "Com1", "lpt9", "aux"])
def test_slugify_avoids_windows_reserved_device_names(name):
    slug = slugify(name)
    assert slug != name.lower()
    assert SLUG.match(slug)


@pytest.mark.parametrize("name", ["", "   ", "!!!", "東京", "🙂🙂"])
def test_slugify_never_returns_an_empty_id(name):
    slug = slugify(name)
    assert SLUG.match(slug)
    assert slugify(name) == slug  # deterministic


def test_slugify_keeps_ids_short():
    slug = slugify("word " * 40)
    assert len(slug) <= 48
    assert SLUG.match(slug)


def test_sequential_ids_count_per_prefix():
    ids = SequentialIds()
    assert [ids.new("f"), ids.new("f"), ids.new("d"), ids.new("f")] == [
        "f-000001", "f-000002", "d-000001", "f-000003"]


def test_random_ids_are_prefixed_and_unique():
    ids = RandomIds()
    made = {ids.new("f") for _ in range(500)}
    assert len(made) == 500
    assert all(re.match(r"^f-[0-9a-f]{12}$", value) for value in made)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+1 555 000 0101", "+15550000101"),
        ("+1 (555) 000-0101", "+15550000101"),
        ("+1.555.000.0103", "+15550000103"),
        ("001 555 000 0104", "+15550000104"),
        ("  Priya@Example.COM ", "priya@example.com"),
        ("whatsapp:+15550000101", "+15550000101"),
        ("mailto:Priya@example.com", "priya@example.com"),
    ],
)
def test_canonical_handle(raw, expected):
    assert canonical_handle(raw) == expected


def test_national_numbers_need_a_default_country_code():
    assert canonical_handle("555 000 0101") is None
    assert canonical_handle("555 000 0101", default_country_code="1") == "+15550000101"
    assert canonical_handle("0555 000 0105", default_country_code="1") == "+15550000105"


def test_digits_with_a_country_code_but_no_plus_are_accepted():
    assert canonical_handle("15550000101", default_country_code="1") == "+15550000101"


@pytest.mark.parametrize(
    "raw", ["", "hello", "+1 555 CALL NOW", "+12", "+1 555 000 0101 23456", "a@b", "@example.com"])
def test_malformed_handles_are_rejected(raw):
    assert canonical_handle(raw, default_country_code="1") is None


def test_numbers_sharing_a_long_tail_stay_distinct():
    # Same last ten digits, different country codes: these are different people.
    us = canonical_handle("+1 555 000 0101")
    ru = canonical_handle("+7 555 000 0101")
    assert us != ru
