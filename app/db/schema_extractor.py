
"""Extract database schema metadata for Talk2SQL."""

from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, inspect
from sqlalchemy.engine.reflection import Inspector

from app.core.logging_config import logger
from app.exceptions.custom_exceptions import CustomException


class ColumnSchema(BaseModel):
    """Describe a database column."""

    model_config = ConfigDict(extra="forbid")

    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool


class ForeignKeySchema(BaseModel):
    """Describe a foreign-key relationship."""

    model_config = ConfigDict(extra="forbid")

    column_names: list[str]
    referenced_table: str
    referenced_columns: list[str]
    referenced_schema: str | None = None


class TableSchema(BaseModel):
    """Describe a database table."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    columns: list[ColumnSchema]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeySchema]


class DatabaseSchema(BaseModel):
    """Represent the extracted database schema."""

    model_config = ConfigDict(extra="forbid")

    dialect: str
    tables: list[TableSchema]


class SchemaExtractor:
    """Inspect a database and produce structured schema metadata."""

    def __init__(self, engine: Engine) -> None:
        """Initialize the extractor with an existing SQLAlchemy engine."""

        if engine is None:
            raise ValueError("A SQLAlchemy engine is required.")

        self.engine = engine
        self.inspector: Inspector = inspect(engine)

        logger.info("SchemaExtractor initialized.")

    def get_table_names(self) -> list[str]:
        """Return sorted table names from the default database schema."""

        try:
            table_names = sorted(self.inspector.get_table_names())

            logger.info(
                "Discovered %d database tables.",
                len(table_names),
            )

            return table_names

        except Exception as exc:
            logger.error(
                "Database table inspection failed: %s",
                type(exc).__name__,
            )
            raise CustomException(
                "Unable to retrieve database table names.",
                exc,
            ) from exc

    def _extract_table_schema(self, table_name: str) -> TableSchema:
        """Extract column, primary-key, and foreign-key metadata."""

        try:
            column_metadata: list[dict[str, Any]] = (
                self.inspector.get_columns(table_name)
            )

            primary_key_metadata = (
                self.inspector.get_pk_constraint(table_name)
            )
            primary_keys = (
                primary_key_metadata.get("constrained_columns") or []
            )
            primary_key_set = set(primary_keys)

            columns = [
                ColumnSchema(
                    name=column["name"],
                    data_type=str(column["type"]),
                    nullable=column["nullable"],
                    is_primary_key=column["name"] in primary_key_set,
                )
                for column in column_metadata
            ]

            foreign_key_metadata = (
                self.inspector.get_foreign_keys(table_name)
            )

            foreign_keys = [
                ForeignKeySchema(
                    column_names=foreign_key["constrained_columns"],
                    referenced_table=foreign_key["referred_table"],
                    referenced_columns=foreign_key["referred_columns"],
                    referenced_schema=foreign_key.get("referred_schema"),
                )
                for foreign_key in foreign_key_metadata
                if foreign_key.get("constrained_columns")
                and foreign_key.get("referred_table")
                and foreign_key.get("referred_columns")
            ]

            return TableSchema(
                table_name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
            )

        except Exception as exc:
            logger.error(
                "Schema inspection failed for table '%s': %s",
                table_name,
                type(exc).__name__,
            )
            raise CustomException(
                f"Unable to inspect table '{table_name}'.",
                exc,
            ) from exc

    def extract_schema(self) -> DatabaseSchema:
        """Extract the schema from the database's default schema."""

        try:
            tables = [
                self._extract_table_schema(table_name)
                for table_name in self.get_table_names()
            ]

            database_schema = DatabaseSchema(
                dialect=self.engine.dialect.name,
                tables=tables,
            )

            logger.info(
                "Schema extraction completed for %d tables.",
                len(tables),
            )

            return database_schema

        except CustomException:
            raise

        except Exception as exc:
            logger.error(
                "Database schema extraction failed: %s",
                type(exc).__name__,
            )
            raise CustomException(
                "Unable to extract the database schema.",
                exc,
            ) from exc

    @staticmethod
    def format_schema_for_llm( database_schema: DatabaseSchema,) -> str:
        """Format schema metadata as readable text for an LLM prompt."""

        lines = [
            f"Database dialect: {database_schema.dialect}",
            "",
            "Database schema:",
        ]

        if not database_schema.tables:
            lines.append("No tables were found.")
            return "\n".join(lines)

        for table in database_schema.tables:
            lines.extend(["", f"Table: {table.table_name}"])

            for column in table.columns:
                attributes = [column.data_type]

                if column.is_primary_key:
                    attributes.append("PRIMARY KEY")

                attributes.append(
                    "NOT NULL"
                    if column.is_primary_key or not column.nullable
                    else "NULLABLE"
                )

                lines.append(
                    f"  - {column.name}: {', '.join(attributes)}"
                )

            for foreign_key in table.foreign_keys:
                local_columns = ", ".join(foreign_key.column_names)
                remote_columns = ", ".join(
                    foreign_key.referenced_columns
                )

                referenced_table = foreign_key.referenced_table

                if foreign_key.referenced_schema:
                    referenced_table = (
                        f"{foreign_key.referenced_schema}."
                        f"{referenced_table}"
                    )

                lines.append(
                    f"  - FOREIGN KEY ({local_columns}) REFERENCES "
                    f"{referenced_table} ({remote_columns})"
                )

        return "\n".join(lines)
