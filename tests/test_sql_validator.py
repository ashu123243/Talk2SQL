"""Tests for SQLValidator."""

import pytest

from app.validation.sql_validator import (
    REASON_DDL,
    REASON_DML_WRITE,
    REASON_EXPLAIN_ROWS,
    REASON_EXPLAIN_UNAVAILABLE,
    REASON_PARSE_FAILURE,
    REASON_SUBQUERY_DEPTH,
    SQLValidator,
)

def test_schema_hash_is_stable_for_equivalent_dictionaries():
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}

    assert SQLValidator._schema_hash(first) == SQLValidator._schema_hash(second)

def test_schema_hash_returns_empty_string_for_missing_schema():
    assert SQLValidator._schema_hash(None) == ""

def test_valid_select_is_allowed_and_limit_is_injected():
    validator = SQLValidator(
        limit_cap=100,
        explain_runner=lambda sql: 50,
    )

    result = validator.validate("SELECT id FROM users")

    assert result.allowed is True
    assert result.blocked_reason is None
    assert "LIMIT 100" in result.sql
    assert result.warnings

def test_existing_limit_is_preserved():
    validator = SQLValidator(
        limit_cap=100,
        explain_runner=lambda sql: 50,
    )

    result = validator.validate("SELECT id FROM users LIMIT 20")

    assert result.allowed is True
    assert "LIMIT 20" in result.sql
    assert "LIMIT 100" not in result.sql
    assert result.warnings == []

def test_empty_sql_is_blocked():
    validator = SQLValidator(explain_runner=lambda sql: 10)

    result = validator.validate("")

    assert result.allowed is False
    assert result.blocked_reason == REASON_PARSE_FAILURE


def test_multiple_statements_are_blocked():
    validator = SQLValidator(explain_runner=lambda sql: 10)

    result = validator.validate(
        "SELECT id FROM users; SELECT id FROM orders"
    )

    assert result.allowed is False
    assert result.blocked_reason == REASON_PARSE_FAILURE


def test_write_statement_is_blocked():
    validator = SQLValidator(explain_runner=lambda sql: 10)

    result = validator.validate("DELETE FROM users")

    assert result.allowed is False
    assert result.blocked_reason == REASON_PARSE_FAILURE


def test_prohibited_ddl_keyword_is_blocked():
    validator = SQLValidator(explain_runner=lambda sql: 10)

    result = validator.validate("SELECT drop FROM users")

    assert result.allowed is False
    assert result.blocked_reason == REASON_DDL


def test_prohibited_write_keyword_is_blocked():
    validator = SQLValidator(explain_runner=lambda sql: 10)

    result = validator.validate("SELECT update FROM users")

    assert result.allowed is False
    assert result.blocked_reason == REASON_DML_WRITE

def test_excessive_subquery_depth_is_blocked():
    validator = SQLValidator(
        max_subquery_depth=1,
        explain_runner=lambda sql: 10,
    )

    sql = (
        "SELECT id FROM users "
        "WHERE id IN (SELECT user_id FROM orders "
        "WHERE user_id IN (SELECT id FROM accounts))"
    )

    result = validator.validate(sql)

    assert result.allowed is False
    assert result.blocked_reason == REASON_SUBQUERY_DEPTH


def test_subquery_depth_within_limit_is_allowed():
    validator = SQLValidator(
        max_subquery_depth=2,
        explain_runner=lambda sql: 10,
    )

    sql = (
        "SELECT id FROM users "
        "WHERE id IN (SELECT user_id FROM orders)"
    )

    result = validator.validate(sql)

    assert result.allowed is True

def test_estimate_above_threshold_is_blocked():
    validator = SQLValidator(
        max_explain_rows=100,
        explain_runner=lambda sql: 101,
    )

    result = validator.validate("SELECT id FROM users")

    assert result.allowed is False
    assert result.blocked_reason == REASON_EXPLAIN_ROWS


def test_estimate_at_threshold_is_allowed():
    validator = SQLValidator(
        max_explain_rows=100,
        explain_runner=lambda sql: 100,
    )

    result = validator.validate("SELECT id FROM users")

    assert result.allowed is True


@pytest.mark.parametrize("invalid_estimate", [-1, 1.5, True, "100", None])
def test_invalid_estimate_fails_closed(invalid_estimate):
    validator = SQLValidator(
        explain_runner=lambda sql: invalid_estimate,
    )

    result = validator.validate("SELECT id FROM users")

    assert result.allowed is False
    assert result.blocked_reason == REASON_EXPLAIN_UNAVAILABLE


def test_missing_explain_runner_fails_closed():
    validator = SQLValidator()

    result = validator.validate("SELECT id FROM users")

    assert result.allowed is False
    assert result.blocked_reason == REASON_EXPLAIN_UNAVAILABLE

@pytest.mark.parametrize(
    "configuration",
    [
        {"limit_cap": 0},
        {"limit_cap": -1},
        {"max_subquery_depth": 0},
        {"max_explain_rows": 0},
        {"statement_timeout_ms": 0},
    ],
)
def test_invalid_configuration_raises_value_error(configuration):
    with pytest.raises(ValueError):
        SQLValidator(**configuration)