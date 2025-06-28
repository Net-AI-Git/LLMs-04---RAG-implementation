import os
from typing import Dict, Any, Optional, List

import psycopg2
from dotenv import load_dotenv

from utils.logging_utils import get_logger, ConfigurationError, DatabaseError

# Setup colored logger for this module
logger = get_logger(__name__)

# Global configuration cache
_config: Optional[Dict[str, Any]] = None


def load_configuration() -> Dict[str, Any]:
    """
    Load and cache configuration from environment variables.

    This function is centralized to be used by any module needing configuration.

    Returns:
        Dict[str, Any]: Configuration dictionary containing API key, DB URL, and embedding model.

    Raises:
        ConfigurationError: If required configuration is missing.
    """
    global _config
    if _config is not None:
        return _config

    load_dotenv()

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ConfigurationError("GEMINI_API_KEY not found in environment")

    embedding_model = os.getenv('EMBEDDING_MODEL')
    if not embedding_model:
        raise ConfigurationError("EMBEDDING_MODEL not found in environment")

    # Ensure embedding model has proper format for Gemini API
    if not embedding_model.startswith(('models/', 'tunedModels/')):
        embedding_model = f"models/{embedding_model}"

    # Load the single PostgreSQL URL
    postgres_url = os.getenv('POSTGRES_URL')
    if not postgres_url:
        raise ConfigurationError("POSTGRES_URL not found in environment")

    _config = {
        'api_key': api_key,
        'postgres_url': postgres_url,
        'embedding_model': embedding_model
    }

    logger.info("Configuration loaded successfully")
    return _config


def get_db_connection() -> psycopg2.extensions.connection:
    """
    Establishes and returns a connection to the PostgreSQL database using a DSN.

    Returns:
        psycopg2.extensions.connection: An active database connection object.

    Raises:
        DatabaseError: If connection to the database fails.
    """
    try:
        config = load_configuration()
        # Use the postgres_url directly as the DSN (Data Source Name)
        connection = psycopg2.connect(dsn=config['postgres_url'])
        logger.info("Database connection established successfully.")
        return connection
    except (ConfigurationError, psycopg2.Error) as e:
        logger.error(f"Failed to get database connection: {e}")
        raise DatabaseError(f"Failed to get database connection: {e}")


def ensure_database_schema(connection: psycopg2.extensions.connection) -> None:
    """
    Ensure database schema exists with all required tables and functions.
    This function is idempotent and will not alter existing tables.

    Args:
        connection: Active PostgreSQL connection.
    """
    cursor = connection.cursor()
    try:
        # Create the main documents table with the full, correct schema.
        # The 'IF NOT EXISTS' clause prevents errors if the table already exists.
        create_table_query = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            chunk_text TEXT,
            split_strategy VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW(),
            embedding REAL[],
            embedding_norm REAL
        );
        """
        cursor.execute(create_table_query)

        # Create SQL functions for cosine similarity.
        # 'CREATE OR REPLACE' makes this operation safe to run multiple times.
        create_dot_product_function = """
        CREATE OR REPLACE FUNCTION dot_product(a REAL[], b REAL[])
        RETURNS REAL AS $$
            SELECT SUM(ai * bi)
            FROM unnest(a) WITH ORDINALITY AS t1(ai, i)
            JOIN unnest(b) WITH ORDINALITY AS t2(bi, i) ON t1.i = t2.i
        $$ LANGUAGE SQL IMMUTABLE;
        """
        cursor.execute(create_dot_product_function)

        create_vector_norm_function = """
        CREATE OR REPLACE FUNCTION vector_norm(a REAL[])
        RETURNS REAL AS $$
            SELECT SQRT(SUM(ai * ai))
            FROM unnest(a) AS ai
        $$ LANGUAGE SQL IMMUTABLE;
        """
        cursor.execute(create_vector_norm_function)

        connection.commit()
        logger.info("Database schema verified.")
    except Exception as e:
        connection.rollback()
        logger.error(f"Failed to ensure database schema: {e}")
        raise DatabaseError(f"Failed to ensure database schema: {e}")
    finally:
        cursor.close()


def delete_document_data(filename: Optional[str] = None) -> bool:
    """
    Deletes document data from the database.

    If a filename is provided, only data for that file is deleted.
    If no filename is provided, the entire documents table is cleared.

    Args:
        filename (Optional[str]): The specific filename to delete. Defaults to None.

    Returns:
        bool: True on success, False on failure.
    """
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        if filename:
            # Delete a specific document's data
            logger.info(f"Deleting data for filename: {filename}")
            cursor.execute("DELETE FROM documents WHERE filename = %s", (filename,))
            action = f"deleted {cursor.rowcount} rows for {filename}"
        else:
            # Clear the entire table
            logger.warning("Clearing all data from the documents table.")
            cursor.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
            action = "cleared all data"

        connection.commit()
        logger.info(f"Successfully {action}.")
        return True
    except (DatabaseError, psycopg2.Error) as e:
        if connection:
            connection.rollback()
        logger.error(f"Failed to delete document data: {e}")
        return False
    finally:
        if connection:
            connection.close()


def get_indexed_filenames() -> List[str]:
    """
    Retrieves a list of unique, sorted filenames currently in the database.

    Returns:
        List[str]: A list of indexed filenames.
    """
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT filename FROM documents ORDER BY filename;")
        # fetchall() returns a list of tuples, e.g., [('file1.pdf',), ('file2.docx',)]
        # We extract the first element from each tuple.
        filenames = [item[0] for item in cursor.fetchall()]
        logger.info(f"Retrieved {len(filenames)} unique filenames from the database.")
        return filenames
    except (DatabaseError, psycopg2.Error) as e:
        logger.error(f"Failed to retrieve indexed filenames: {e}")
        return []  # Return an empty list on error
    finally:
        if connection:
            connection.close()
