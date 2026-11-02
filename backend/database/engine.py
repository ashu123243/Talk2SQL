
"""Database engine and connection lifecycle management for Talk2SQL."""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

from backend.config import Settings, settings
from backend.core.logging_config import get_logger
from backend.exceptions.custom_exceptions import (
    ConfigurationException,
    DatabaseException,
)


class DatabaseEngine:
    """Manage SQLAlchemy engine creation and database connections."""

    def __init__(
        self,
        config: Settings | None = None,
    ) -> None:
        """Initialize the database manager without connecting."""
        
        self._config = config if config is not None else settings
        
        self.engine: Engine |None=None
        
        self._logger = get_logger('database.engine')
        
        self._logger.info("DatabaseEngine manager initialized")
        
    def _get_database_url(self) -> str:
        """Retrieve and validate the configured MySQL URL."""
        
        database_url = self._config.database_url
        
        if database_url is None:
            raise ConfigurationException(
                "Database URL is not configured.",
                error_code="DATABASE_URL_MISSING",
            )
       
        raw_url = database_url.get_secret_value().strip()
        
        if not raw_url:
            raise ConfigurationException(
                "Database URL cannot be empty.",
                error_code="DATABASE_URL_EMPTY",
            )
            
        try:
            parsed_url = make_url(raw_url)
            
        except (SQLAlchemyError, ValueError, TypeError) as exc:
            raise ConfigurationException(
                "Database URL has an invalid format.",
                error_details=exc,
                error_code="INVALID_DATABASE_URL",
            ) from exc
        
        if parsed_url.drivername != "mysql+pymysql":
            raise ConfigurationException(
                "Database URL must use the mysql+pymysql driver.",
                error_code="UNSUPPORTED_DATABASE_DRIVER",
            )

        if not parsed_url.host or not parsed_url.database:
            raise ConfigurationException(
                "Database URL must include a host and database name.",
                error_code="INCOMPLETE_DATABASE_URL",
            )

        return raw_url
       
    def create(self) -> Engine:
        """Create an engine or return the existing engine."""
        
        if self.engine is not None:
               return self.engine
        
        database_url = self._get_database_url()
        
        try:
            self.engine = create_engine(
                database_url,
                echo = self._config.database_echo,
                pool_size = self._config.database_pool_size,
                max_overflow=self._config.database_max_overflow,
                pool_timeout=self._config.database_pool_timeout,
                pool_recycle=self._config.database_pool_recycle,
                pool_pre_ping=True, 
            )
            
            self._logger.info(
                "SQLAlchemy engine initialized successfully."
            )

            return self.engine
        except (SQLAlchemyError, ValueError, TypeError) as exc:

            self.engine = None

            self._logger.error(
                "SQLAlchemy engine initialization failed. "
                "Exception type: %s",
                type(exc).__name__,
            )

            raise DatabaseException(
                "Unable to initialize the database engine.",
                error_details=exc,
                error_code="DATABASE_ENGINE_INITIALIZATION_ERROR",
            ) from exc
      
    def get_engine(self) -> Engine:
        """Return the existing engine or create one if necessary."""

        if self.engine is None:
            return self.create()

        return self.engine

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        """Provide a managed database connection."""

        engine = self.get_engine()
        
        try:
            with engine.connect() as connection:
                   yield connection
        
        except ConfigurationException:
            raise
        
        except DatabaseException:
            raise

        except SQLAlchemyError as exc:

            self._logger.error(
                "Database connection operation failed. "
                "Exception type: %s",
                type(exc).__name__,
            )

            raise DatabaseException(
                "Unable to complete the database connection operation.",
                error_details=exc,
                error_code="DATABASE_CONNECTION_ERROR",
            ) from exc

    def check_connection(self) -> bool:
        """Check database connectivity using a harmless SELECT 1."""

        try:
            with self.connect() as connection:
                connection.execute(text("SELECT 1"))

            self._logger.info(
                "Database connectivity check succeeded."
            )

            return True

        except (SQLAlchemyError, DatabaseException) as exc:

            self._logger.error(
                "Database connectivity check failed. "
                "Exception type: %s",
                type(exc).__name__,
            )

            return False

        except ConfigurationException as exc:

            self._logger.error(
                "Database connectivity check could not start. "
                "Error code: %s",
                exc.error_code,
            )

            return False               

    def dispose(self) -> None:
        """Dispose of the engine and release its pooled connections."""

        if self.engine is None:
            return

        try:
            self.engine.dispose()

            self.engine = None

            self._logger.info(
                "SQLAlchemy engine disposed successfully."
            )

        except SQLAlchemyError as exc:

            self._logger.error(
                "SQLAlchemy engine disposal failed. "
                "Exception type: %s",
                type(exc).__name__,
            )

            raise DatabaseException(
                "Unable to dispose of the database engine.",
                error_details=exc,
                error_code="DATABASE_ENGINE_DISPOSAL_ERROR",
            ) from exc

    def __enter__(self) -> "DatabaseEngine":
        """Support context-manager initialization."""

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        """Dispose of the engine when leaving the context."""

        self.dispose()

        return False