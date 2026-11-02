
"""Database schema discovery and extraction for Talk2SQL."""

from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.reflection import Inspector

from backend.core.logging_config import get_logger
from backend.exceptions.custom_exceptions import DatabaseException


class ColumnSchema(BaseModel):
    """Represent a database column."""

    model_config = ConfigDict(extra='forbid')
    
    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool
    default_value: str | None = None
    comment: str | None = None


class ForeignKeySchema(BaseModel):
    """Represent a foreign-key relationship."""

    model_config = ConfigDict(extra="forbid")

    constraint_name: str | None = None
    column_names: list[str]
    referenced_table: str
    referenced_columns: list[str]
    referenced_schema: str | None = None
    
    
class UniqueConstraintSchema(BaseModel):
    """Represent a unique constraint."""

    model_config = ConfigDict(extra="forbid")

    constraint_name: str | None = None
    column_names: list[str]  


class CheckConstraintSchema(BaseModel):
    """Represent a check constraint."""

    model_config = ConfigDict(extra="forbid")

    constraint_name: str | None = None
    sql_expression: str
  
  
class IndexSchema(BaseModel):
    """Represent a database index."""

    model_config = ConfigDict(extra="forbid")

    index_name: str
    column_names: list[str]
    is_unique: bool
    index_type: str | None = None  


class TableSchema(BaseModel):
    """Represent the complete structure of a database table."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    columns: list[ColumnSchema]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeySchema]
    unique_constraints: list[UniqueConstraintSchema]
    check_constraints: list[CheckConstraintSchema]
    indexes: list[IndexSchema]  


class ViewSchema(BaseModel):
    """Represent a database view."""

    model_config = ConfigDict(extra="forbid")

    view_name: str
    columns: list[ColumnSchema]   


class DatabaseSchema(BaseModel):
    """Represent the extracted database structure."""

    model_config = ConfigDict(extra="forbid")

    dialect: str
    tables: list[TableSchema]
    views: list[ViewSchema]


class SchemaExtractor:
    """Inspect a SQLAlchemy engine and extract database metadata."""
    
    def __init__(self, engine:Engine)->None:
        """Initialize the schema extractor."""
        
        self.engine = engine
        self.inspector: Inspector = inspect(engine)
        self._logger = get_logger("database.schema_extractor")

        self._logger.info("SchemaExtractor initialized successfully.")
    
    def get_table_names(self) -> list[str]:
        """Return all table names in the inspected database."""
        
        try:
            table_names = self.inspector.get_table_names()
            
            self._logger.info(
                "Discovered %d database tables.",
                len(table_names),
            )
            
            return sorted(table_names)
        
        except SQLAlchemyError as exc:
            self._logger.info(
                "Table discovery failed. Exception type: %s",
                type(exc).__name__,
            )
            
            raise DatabaseException(
                "Unable to discover database tables.",
                error_details=exc,
                error_code="TABLE_DISCOVERY_ERROR",
            ) from exc
    
    def get_view_names(self) -> list[str]:
        """Return all view names in the inspected database."""

        try:
            view_names = self.inspector.get_view_names()

            self._logger.info(
                "Discovered %d database views.",
                len(view_names),
            )

            return sorted(view_names)

        except SQLAlchemyError as exc:
            self._logger.error(
                "View discovery failed. Exception type: %s",
                type(exc).__name__,
            )

            raise DatabaseException(
                "Unable to discover database views.",
                error_details=exc,
                error_code="VIEW_DISCOVERY_ERROR",
            ) from exc
    
    def _extract_table_schema(self, table_name: str) -> TableSchema:
        """Extract columns, keys, constraints, and indexes for a table."""

        try:
            raw_columns = self.inspector.get_columns(table_name)
            
            primary_key_data = self.inspector.get_pk_constraint(table_name)
            
            primary_keys = (primary_key_data.get('constrained_columns') or [])
            
            primary_key_set = set(primary_keys)
            
            columns = [
                ColumnSchema(
                    name=column["name"],
                    data_type=str(column["type"]),
                    nullable=bool(column.get("nullable", True)),
                    is_primary_key=(
                        column["name"] in primary_key_set
                    ),
                    default_value=(
                        str(column["default"])
                        if column.get("default") is not None
                        else None
                    ),
                    comment=column.get("comment"),
                )
                for column in raw_columns
            ]
            
            raw_foreign_keys = self.inspector.get_foreign_keys(table_name)
            
            foreign_keys = [
                ForeignKeySchema(
                    constraint_name=foreign_key.get("name"),
                    column_names=(
                        foreign_key.get("constrained_columns") or []
                    ),
                    referenced_table=foreign_key["referred_table"],
                    referenced_columns=(
                        foreign_key.get("referred_columns") or []
                    ),
                    referenced_schema=foreign_key.get(
                        "referred_schema"
                    ),
                )
                for foreign_key in raw_foreign_keys
            ]
            
            raw_unique_constraints = (
                self.inspector.get_unique_constraints(table_name)
            )
            
            unique_constraints = [
                UniqueConstraintSchema(
                    constraint_name=constraint.get("name"),
                    column_names=constraint.get("column_names") or [],
                )
                for constraint in raw_unique_constraints
            ]
            
            try:
                raw_check_constraints = (
                    self.inspector.get_check_constraints(table_name)
                )

            except NotImplementedError:
                self._logger.warning(
                    "Check constraints are not supported by the "
                    "current database dialect for table '%s'.",
                    table_name,
                )

                raw_check_constraints = []
            
            check_constraints = [
                CheckConstraintSchema(
                    constraint_name=constraint.get("name"),
                    sql_expression=str(
                        constraint.get("sqltext", "")
                    ),
                )
                for constraint in raw_check_constraints
            ]
            
            raw_indexes = self.inspector.get_indexes(table_name)

            indexes = []
            
            for index in raw_indexes:
                dialect_options = index.get("dialect_options") or {}

                index_type = index.get("type")

                if index_type is None:
                    for option_name, option_value in (
                        dialect_options.items()
                    ):
                        if option_name.endswith("_using"):
                            index_type = str(option_value)
                            break

                index_columns = [
                    column_name
                    for column_name in (
                        index.get("column_names") or []
                    )
                    if column_name is not None
                ]

                if index.get("name") is None:
                    continue

                indexes.append(
                    IndexSchema(
                        index_name=index["name"],
                        column_names=index_columns,
                        is_unique=bool(index.get("unique", False)),
                        index_type=index_type,
                    )
                )

            table_schema = TableSchema(
                table_name=table_name,
                columns=columns,
                primary_keys=list(primary_keys),
                foreign_keys=foreign_keys,
                unique_constraints=unique_constraints,
                check_constraints=check_constraints,
                indexes=indexes,
            )

            self._logger.debug(
                "Extracted schema for table '%s'.",
                table_name,
            )

            return table_schema

        except SQLAlchemyError as exc:
            self._logger.error(
                "Schema extraction failed for table '%s'. "
                "Exception type: %s",
                table_name,
                type(exc).__name__,
            )

            raise DatabaseException(
                f"Unable to extract schema for table '{table_name}'.",
                error_details=exc,
                error_code="TABLE_SCHEMA_EXTRACTION_ERROR",
            ) from exc
    
    def _extract_view_schema(self, view_name: str) -> ViewSchema:
        """Extract column metadata for a database view."""
        
        try:
            raw_columns = self.inspector.get_columns(view_name)

            columns = [
                ColumnSchema(
                    name=column["name"],
                    data_type=str(column["type"]),
                    nullable=bool(column.get("nullable", True)),
                    is_primary_key=False,
                    default_value=(
                        str(column["default"])
                        if column.get("default") is not None
                        else None
                    ),
                    comment=column.get("comment"),
                )
                for column in raw_columns
            ]

            return ViewSchema(
                view_name=view_name,
                columns=columns,
            )

        except SQLAlchemyError as exc:
            self._logger.error(
                "View schema extraction failed for '%s'. "
                "Exception type: %s",
                view_name,
                type(exc).__name__,
            )

            raise DatabaseException(
                f"Unable to extract schema for view '{view_name}'.",
                error_details=exc,
                error_code="VIEW_SCHEMA_EXTRACTION_ERROR",
            ) from exc
            
    def extract_schema(self) -> DatabaseSchema:
        """Extract and validate the complete database schema."""

        self._logger.info("Starting database schema extraction.")

        try:
            dialect = self.engine.dialect.name
            
            table_names = self.get_table_names()
            view_names = self.get_view_names()
            
            tables = [
                self._extract_table_schema(table_name)
                for table_name in table_names
            ]
            
            views = [
                self._extract_view_schema(view_name)
                for view_name in view_names
            ]
            
            database_schema = DatabaseSchema(
                dialect=dialect,
                tables=tables,
                views=views,
            )
            
            self._logger.info(
                "Database schema extraction completed. "
                "Tables: %d, Views: %d.",
                len(tables),
                len(views),
            )

            return database_schema

        except DatabaseException:
            raise

        except SQLAlchemyError as exc:
            self._logger.error(
                "Database schema extraction failed. "
                "Exception type: %s",
                type(exc).__name__,
            )

            raise DatabaseException(
                "Unable to extract the database schema.",
                error_details=exc,
                error_code="DATABASE_SCHEMA_EXTRACTION_ERROR",
            ) from exc

    def format_schema_for_llm(
        self,
        database_schema: DatabaseSchema,
    ) -> str:
        """Convert extracted schema metadata into readable text."""
        
        if not isinstance(database_schema, DatabaseSchema):
            raise TypeError(
                "database_schema must be a DatabaseSchema instance."
            )

        lines = [
            f"DATABASE DIALECT: {database_schema.dialect}",
            "",
        ]
        
        for table in database_schema.tables:
            lines.append(f"TABLE: {table.table_name}")
            lines.append("COLUMNS:")
            
            for column in table.columns:
                attributes = []

                if column.is_primary_key:
                    attributes.append("PRIMARY KEY")

                if not column.nullable:
                    attributes.append("NOT NULL")
                else:
                    attributes.append("NULLABLE")
                    
                column_description = (
                    f"- {column.name}: {column.data_type}"
                )

                if attributes:
                    column_description += (
                        f" [{', '.join(attributes)}]"
                    )

                if column.default_value is not None:
                    column_description += (
                        f" DEFAULT {column.default_value}"
                    )

                if column.comment:
                    column_description += (
                        f" COMMENT: {column.comment}"
                    )

                lines.append(column_description)
            
            if table.foreign_keys:
                lines.append("FOREIGN KEYS:")

                for foreign_key in table.foreign_keys:
                    source_columns = ", ".join(
                        foreign_key.column_names
                    )

                    target_columns = ", ".join(
                        foreign_key.referenced_columns
                    )

                    lines.append(
                        f"- {table.table_name}.{source_columns} "
                        f"REFERENCES "
                        f"{foreign_key.referenced_table}."
                        f"{target_columns}"
                    )
            
            if table.unique_constraints:
                lines.append("UNIQUE CONSTRAINTS:")

                for constraint in table.unique_constraints:
                    columns = ", ".join(constraint.column_names)

                    lines.append(
                        f"- {constraint.constraint_name or 'unnamed'}: "
                        f"({columns})"
                    )

            if table.check_constraints:
                lines.append("CHECK CONSTRAINTS:")

                for constraint in table.check_constraints:
                    lines.append(
                        f"- {constraint.constraint_name or 'unnamed'}: "
                        f"{constraint.sql_expression}"
                    )

            if table.indexes:
                lines.append("INDEXES:")

                for index in table.indexes:
                    columns = ", ".join(index.column_names)

                    uniqueness = (
                        "UNIQUE" if index.is_unique else "NON-UNIQUE"
                    )

                    index_description = (
                        f"- {index.index_name}: "
                        f"{uniqueness} ({columns})"
                    )

                    if index.index_type:
                        index_description += (
                            f", TYPE: {index.index_type}"
                        )

                    lines.append(index_description)

            lines.append("")

        for view in database_schema.views:
            lines.append(f"VIEW: {view.view_name}")
            lines.append("COLUMNS:")

            for column in view.columns:
                nullability = (
                    "NULLABLE" if column.nullable else "NOT NULL"
                )

                lines.append(
                    f"- {column.name}: "
                    f"{column.data_type} [{nullability}]"
                )

            lines.append("")

        formatted_schema = "\n".join(lines).strip()

        self._logger.debug(
            "Database schema formatted for LLM consumption."
        )

        return formatted_schema