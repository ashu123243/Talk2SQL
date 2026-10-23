"""Validate generated SQL using layered safety guardrails."""

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable

import sqlparse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.logging_config import logger


REASON_PARSE_FAILURE = "parse_failure"
REASON_DDL = "ddl"
REASON_DML_WRITE = "dml_write"
REASON_SUBQUERY_DEPTH = "subquery_depth"
REASON_EXPLAIN_ROWS = "explain_rows"
REASON_EXPLAIN_UNAVAILABLE = "explain_unavailable"

_TOKEN_RE = re.compile(r"\b[A-Z_][A-Z_0-9]*\b", re.IGNORECASE)
_STRING_RE = re.compile(r"'(?:''|[^'])*'")

_DDL_TOKENS = frozenset({"CREATE", "ALTER", "DROP", "TRUNCATE"})
_DML_WRITE_TOKENS = frozenset(
    {"INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "GRANT", "REVOKE"}
)


class ValidationResult(BaseModel):
    """Represent the outcome of SQL validation."""

    sql: str
    allowed: bool
    warnings: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    def to_block_record(self, *, provider: str = "unknown", schema_hash: str = "",) -> dict[str, Any]:
        """Return structured metadata for a blocked-query log."""

        return {
            "sql": self.sql, "reason": self.blocked_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider, "schema_hash": schema_hash,
        }


class SQLValidator:
    """Validate generated SQL before it reaches the execution layer."""

    def __init__(
        self, *, limit_cap: int = 1000, max_subquery_depth: int = 3,
        max_explain_rows: int = 100_000, statement_timeout_ms: int = 30_000,
        explain_runner: Callable[[str], int] | None = None, log_blocked: bool = True,) -> None:
        """Initialize configurable SQL guardrails."""

        if limit_cap <= 0:
            raise ValueError("limit_cap must be > 0")

        if max_subquery_depth < 1:
            raise ValueError("max_subquery_depth must be >= 1")

        if max_explain_rows <= 0:
            raise ValueError("max_explain_rows must be > 0")

        if statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms must be > 0")

        self.limit_cap = limit_cap
        self.max_subquery_depth = max_subquery_depth
        self.max_explain_rows = max_explain_rows
        self.statement_timeout_ms = statement_timeout_ms
        self._explain_runner = explain_runner
        self.log_blocked = log_blocked

    @staticmethod
    def _schema_hash(schema: Any) -> str:
        """Return a stable short hash for schema metadata."""

        if schema is None:
            return ""

        if hasattr(schema, "model_dump"):
            schema_data = schema.model_dump(mode="json")
        elif isinstance(schema, dict):
            schema_data = schema
        else:
            schema_data = str(schema)

        serialized_schema = json.dumps(
            schema_data, sort_keys=True,
            separators=(",", ":"), default=str,
        )

        return hashlib.sha256(
            serialized_schema.encode("utf-8")
        ).hexdigest()[:16]

    def _block(
        self, sql: str, reason: str,
        message: str, provider: str,
        schema: Any,
    ) -> ValidationResult:
        """Create a rejection result and log safe metadata."""

        schema_hash = self._schema_hash(schema)

        if self.log_blocked:
            # Avoid logging raw SQL because it may contain sensitive data.
            logger.warning(
                "sql_guardrail_block reason=%s message=%s "
                "provider=%s schema_hash=%s",
                reason, message, provider, schema_hash,
            )

        return ValidationResult(
            sql=sql, allowed=False, warnings=[message], blocked_reason=reason,
        )
        
    @staticmethod
    def _strip_string_literals(sql: str) -> str:
        """Remove simple single-quoted literals for token checks."""

        return _STRING_RE.sub("", sql)

    def _validate_statement(self, sql: str) -> str | None:
        """Check that SQL contains exactly one SELECT statement."""

        try:
            statements = [
                statement
                for statement in sqlparse.parse(sql)
                if str(statement).strip()
            ]
        except Exception:
            return "SQL could not be parsed."

        if len(statements) != 1:
            return "SQL must contain exactly one parseable statement."

        if statements[0].get_type() != "SELECT":
            return "Only SELECT queries are supported."

        return None

    def _check_forbidden_operations(self, sql: str) -> tuple[str, str] | None:
        """Reject SQL containing prohibited operation keywords."""

        comment_free_sql = sqlparse.format(sql, strip_comments=True)
        stripped_sql = self._strip_string_literals(comment_free_sql)

        tokens = {
            token.upper()
            for token in _TOKEN_RE.findall(stripped_sql)
        }

        ddl_matches = sorted(tokens & _DDL_TOKENS)

        if ddl_matches:
            return (
                REASON_DDL,
                f"Prohibited DDL operation(s): {ddl_matches}",
            )

        write_matches = sorted(tokens & _DML_WRITE_TOKENS)

        if write_matches:
            return (
                REASON_DML_WRITE,
                f"Prohibited write operation(s): {write_matches}",
            )

        return None

    @staticmethod
    def _walk_subquery_depth(sql: str) -> int:
        """Estimate maximum nested parenthesized SELECT/WITH depth."""

        comment_free_sql = re.sub(r"--[^\n]*", "", sql)
        upper_sql = comment_free_sql.upper()

        max_depth = 0
        subquery_stack: list[bool] = []

        for index, character in enumerate(comment_free_sql):
            if character == "(":
                remaining = upper_sql[index + 1:index + 64].lstrip()

                is_subquery = (
                    remaining.startswith("SELECT")
                    or remaining.startswith("WITH")
                )

                subquery_stack.append(is_subquery)

                if is_subquery:
                    current_depth = sum(subquery_stack)
                    max_depth = max(max_depth, current_depth)

            elif character == ")" and subquery_stack:
                subquery_stack.pop()

        return max_depth

    def _check_subquery_depth(self, sql: str) -> str | None:
        """Return a rejection message if subquery depth is too high."""

        depth = self._walk_subquery_depth(sql)

        if depth > self.max_subquery_depth:
            return (
                f"Subquery depth {depth} exceeds "
                f"the maximum of {self.max_subquery_depth}."
            )

        return None
    
    @staticmethod
    def _has_limit_clause(sql: str) -> bool:
        """Check for an existing numeric LIMIT clause."""

        return bool(
            re.search(r"\bLIMIT\s+\d+\b", sql, re.IGNORECASE)
        )

    def _maybe_inject_limit(
        self,
        sql: str,
    ) -> tuple[str, str | None]:
        """Add the configured LIMIT when no recognized LIMIT exists."""

        formatted_sql = sqlparse.format(sql, strip_comments=True)
        stripped_sql = self._strip_string_literals(formatted_sql)

        if self._has_limit_clause(stripped_sql):
            return sql, None

        body = sql.rstrip()
        trailing_comment = ""

        comment_match = re.search(r"(\s*--[^\n]*)$", body)

        if comment_match:
            trailing_comment = comment_match.group(1)
            body = body[:comment_match.start()].rstrip()

        if body.endswith(";"):
            limited_sql = (
                body[:-1].rstrip()
                + f" LIMIT {self.limit_cap};"
                + trailing_comment
            )
        else:
            limited_sql = (
                body
                + f" LIMIT {self.limit_cap};"
                + trailing_comment
            )

        return (
            limited_sql,
            f"No LIMIT clause was found; LIMIT {self.limit_cap} was added.",
        )

    def _default_explain_runner(
        self,
        engine: Engine,
    ) -> Callable[[str], int]:
        """Create a PostgreSQL EXPLAIN runner with a statement timeout."""

        if engine.dialect.name != "postgresql":
            raise RuntimeError(
                "EXPLAIN row estimation is currently supported only "
                "for PostgreSQL."
            )

        def run_explain(sql: str) -> int:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "SELECT set_config("
                        "'statement_timeout', :timeout, true)"
                    ),
                    {"timeout": f"{self.statement_timeout_ms}ms"},
                )

                result = connection.exec_driver_sql(
                    f"EXPLAIN (FORMAT JSON) {sql}"
                )

                row = result.fetchone()

                if row is None:
                    raise RuntimeError(
                        "PostgreSQL EXPLAIN returned no query plan."
                    )

                plan_data = row[0]

                if isinstance(plan_data, str):
                    plan_data = json.loads(plan_data)

                if (
                    not isinstance(plan_data, list)
                    or not plan_data
                    or not isinstance(plan_data[0], dict)
                ):
                    raise ValueError(
                        "Unexpected PostgreSQL EXPLAIN format."
                    )

                plan = plan_data[0].get("Plan")

                if not isinstance(plan, dict):
                    raise ValueError(
                        "PostgreSQL query plan is missing."
                    )

                estimated_rows = plan.get("Plan Rows")

                if (
                    isinstance(estimated_rows, bool)
                    or not isinstance(estimated_rows, (int, float))
                    or estimated_rows < 0
                    or int(estimated_rows) != estimated_rows
                ):
                    raise ValueError("Invalid EXPLAIN row estimate.")

                return int(estimated_rows)

        return run_explain

    def _check_query_cost(
        self, sql: str, engine: Engine | None,
    ) -> tuple[int | None, str | None]:
        """Obtain the estimated row count or return an error message."""

        try:
            if self._explain_runner is not None:
                estimate = self._explain_runner(sql)
            elif engine is not None:
                estimate = self._default_explain_runner(engine)(sql)
            else:
                raise RuntimeError(
                    "No query-cost estimation mechanism is configured."
                )

            if (
                isinstance(estimate, bool)
                or not isinstance(estimate, int)
                or estimate < 0
            ):
                raise ValueError("Invalid EXPLAIN row estimate.")

            return estimate, None

        except Exception as exc:
            logger.warning(
                "SQL validation cost check unavailable: %s",
                type(exc).__name__,
            )
            return None, (
                "Query-cost estimation failed; SQL execution is not permitted."
            )

    def validate(
        self, sql: str, *, schema: Any = None, engine: Engine | None = None,
        provider: str = "unknown",
    ) -> ValidationResult:
        """Apply guardrails and return a structured validation result."""

        if not isinstance(sql, str) or not sql.strip():
            return self._block(
                sql if isinstance(sql, str) else "",
                REASON_PARSE_FAILURE,
                "SQL is empty or invalid.",
                provider,
                schema,
            )

        statement_error = self._validate_statement(sql)

        if statement_error:
            return self._block(
                sql, REASON_PARSE_FAILURE, statement_error, provider, schema,
            )

        operation_error = self._check_forbidden_operations(sql)

        if operation_error:
            reason, message = operation_error
            return self._block(
                sql, reason, message, provider, schema,
            )

        comment_free_sql = sqlparse.format(sql, strip_comments=True)
        stripped_sql = self._strip_string_literals(comment_free_sql)

        depth_error = self._check_subquery_depth(stripped_sql)

        if depth_error:
            return self._block(
                sql, REASON_SUBQUERY_DEPTH, depth_error, provider, schema,
            )

        limited_sql, limit_warning = self._maybe_inject_limit(sql)
        warnings: list[str] = []

        if limit_warning:
            warnings.append(limit_warning)

        estimate, explain_error = self._check_query_cost(
            limited_sql,
            engine,
        )

        if explain_error:
            return self._block(
                limited_sql, REASON_EXPLAIN_UNAVAILABLE, explain_error, provider, schema,
            )

        assert estimate is not None

        if estimate > self.max_explain_rows:
            return self._block(
                limited_sql,
                REASON_EXPLAIN_ROWS,
                (
                    f"Estimated rows {estimate} exceed "
                    f"the configured maximum of {self.max_explain_rows}."
                ),
                provider, schema,
            )

        return ValidationResult(
            sql=limited_sql, allowed=True, warnings=warnings,
        )


__all__ = [
    "SQLValidator",
    "ValidationResult",
    "REASON_PARSE_FAILURE",
    "REASON_DDL",
    "REASON_DML_WRITE",
    "REASON_SUBQUERY_DEPTH",
    "REASON_EXPLAIN_ROWS",
    "REASON_EXPLAIN_UNAVAILABLE",
]