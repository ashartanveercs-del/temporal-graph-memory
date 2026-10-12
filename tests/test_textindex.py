import sqlite3

import pytest

from tgmem.textindex import fts_expression, like_pattern, query_terms, tokenize


def test_tokenize_matches_the_fts_tokenizer_rules():
    assert tokenize("Priya's order_1042: 9am-6pm, Mon–Sat!") == [
        "priya", "s", "order", "1042", "9am", "6pm", "mon", "sat"]


def test_tokenize_normalizes_unicode_compatibility_forms():
    assert tokenize("ＡＢＣ ﬁle") == ["abc", "file"]


def test_query_terms_drop_stopwords_and_duplicates():
    assert query_terms("What are the opening hours of the shop? hours!") == [
        "opening", "hours", "shop"]


def test_query_terms_keep_stopwords_when_nothing_else_is_left():
    assert query_terms("what is it") == ["what", "is", "it"]


def test_query_terms_are_capped():
    assert len(query_terms(" ".join(f"word{i}" for i in range(100)))) == 32


HOSTILE = [
    'invoice "due',
    "budget) OR (1=1",
    "NEAR(budget due, 2)",
    "value: secret",
    "-budget *",
    "AND OR NOT",
    "'); DROP TABLE edges; --",
    "^start + end",
]


@pytest.mark.parametrize("text", HOSTILE)
def test_fts_expression_is_always_valid_fts5_syntax(text):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(text, other)")
    db.execute("INSERT INTO t VALUES ('the invoice is due friday; budget 480', 'secret')")
    expression = fts_expression(query_terms(text))
    assert expression  # every hostile string still has searchable words
    db.execute("SELECT rowid FROM t WHERE t MATCH ?", (expression,)).fetchall()


def test_fts_expression_quotes_every_term_so_operators_and_columns_are_inert():
    assert fts_expression(["value", "secret", "and"]) == '"value" OR "secret" OR "and"'
    assert fts_expression([]) is None


def test_like_pattern_escapes_wildcards():
    assert like_pattern("100%") == "%100\\%%"
    assert like_pattern("a_b") == "%a\\_b%"
    assert like_pattern("back\\slash") == "%back\\\\slash%"
