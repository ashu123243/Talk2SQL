"""Manage SQLAlchemy engine creation and lifecycle."""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.core.logging_config import logger
from app.exceptions.custom_exceptions import CustomException

class DatabaseEngine:
    """Create, reuse, check, and dispose of a database engine."""
    
    def __init__(self)->None:
        """Initialize the manager without creating an engine."""
        
        self.engine: Engine | None = None
        logger.info("DatabaseEngine manager initialized.")
    
    def create(self)->Engine:
        """Create an engine if needed, or return the existing engine."""
        
        if self.engine is not None:
            return self.engine

        settings = get_settings()
        database_url = settings.database_url

        if not database_url or not database_url.strip():
            raise ValueError("Database URL must not be empty.")
        
        try:
            self.engine = create_engine(database_url)
            
            logger.info("SQLAlchemy engine initialized.")
            return self.engine

        except SQLAlchemyError as exc:
            logger.error(
                "SQLAlchemy engine initialization failed: %s",
                type(exc).__name__,
            )
            
            raise CustomException(
                "Unable to initialize the database engine.",
                exc,
            ) from exc
    
    def check_connection(self)->bool:
        """Check connectivity by executing a harmless SELECT statement."""

        try:
            engine = self.engine or self.create()     
            
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))

            logger.info("Database connectivity check succeeded.")
            return True
        
        except (SQLAlchemyError, CustomException, ValueError) as exc:
            logger.error(
                "Database connectivity check failed: %s",
                type(exc).__name__,
            )
            return False
        
    def dispose(self) -> None:
        """Dispose of the engine and clear its stored reference."""

        if self.engine is None:
            return

        self.engine.dispose()
        self.engine = None

        logger.info("SQLAlchemy engine disposed.") 
        
