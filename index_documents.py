import os
import re
from pathlib import Path
from typing import List, Tuple

import PyPDF2
from docx import Document
import psycopg2
from psycopg2.extras import execute_values

from utils.logging_utils import get_logger, DocumentProcessingError, DatabaseError, EmbeddingGenerationError
from utils.db_utils import get_db_connection, ensure_database_schema, delete_document_data
from utils.shared_utils import chunk_by_paragraphs, generate_embeddings

# Setup colored logger for this module
logger = get_logger(__name__)


def load_document(file_path: str) -> str:
    """
    Load text content from PDF or DOCX files.

    Args:
        file_path (str): Path to the document file (.pdf or .docx only)

    Returns:
        str: Raw text content extracted from the document

    Raises:
        ValueError: If file extension is not supported or file is empty
        FileNotFoundError: If the specified file does not exist
        DocumentProcessingError: If file cannot be read or processed
    """
    logger.info(f"Processing file: {file_path}")

    file_extension = Path(file_path).suffix.lower()
    logger.info(f"File type: {file_extension}")

    if file_extension not in ['.pdf', '.docx']:
        logger.error(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}. Only PDF and DOCX are supported.")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if file_extension == '.pdf':
            text_content = _extract_pdf_text(file_path)
        else:  # .docx
            text_content = _extract_docx_text(file_path)
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {str(e)}")
        raise DocumentProcessingError(f"Failed to process file {file_path}: {str(e)}")

    if not text_content or not text_content.strip():
        logger.error(f"File is empty: {file_path}")
        raise ValueError(f"File is empty: {file_path}")

    logger.info(f"Text length: {len(text_content)} characters")
    return text_content


def _extract_pdf_text(file_path: str) -> str:
    """
    Extract and normalize text from a PDF file.
    This function handles inconsistent paragraph breaks common in PDF text extraction.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        str: Normalized and extracted text content
    """
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        raw_text = ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    if not raw_text:
        return ""

    # Normalize paragraph breaks. PDFs often have inconsistent newlines.
    # This regex replaces any sequence of 2 or more newlines (with optional whitespace between them)
    # with a clean, consistent double newline.
    normalized_text = re.sub(r'(\n\s*){2,}', '\n\n', raw_text)
    logger.info("Normalized paragraph breaks for PDF text.")

    return normalized_text


def _extract_docx_text(file_path: str) -> str:
    """
    Extract text from DOCX file.

    Args:
        file_path (str): Path to the DOCX file

    Returns:
        str: Extracted text content
    """
    document = Document(file_path)
    return '\n\n'.join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())


def _validate_data_for_indexing(chunks: List[str], embeddings: List[List[float]], filename: str) -> None:
    """
    Validates that the data intended for indexing is consistent and not empty.

    Args:
        chunks (List[str]): List of text chunks.
        embeddings (List[List[float]]): List of embedding vectors.
        filename (str): Source filename.

    Raises:
        ValueError: If any data validation fails.
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty for indexing.")
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty for indexing.")
    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunks count ({len(chunks)}) doesn't match embeddings count ({len(embeddings)}).")
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty for indexing.")


def store_chunks_to_db(chunks: List[str], embeddings: List[List[float]],
                       filename: str, split_strategy: str = "paragraph") -> bool:
    """
    Store text chunks and their embeddings in the PostgreSQL database using an efficient bulk insert operation.

    Args:
        chunks (List[str]): List of text chunks.
        embeddings (List[List[float]]): List of embedding vectors.
        filename (str): Source filename.
        split_strategy (str): The strategy used for chunking (e.g., "paragraph").

    Returns:
        bool: True if storage succeeded, False otherwise.
    """
    try:
        _validate_data_for_indexing(chunks, embeddings, filename)
    except ValueError as e:
        logger.error(f"Data validation failed for '{filename}': {e}")
        return False

    logger.info(f"Preparing to insert {len(chunks)} chunks for '{filename}' in a single batch.")

    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        ensure_database_schema(connection)

        cursor = connection.cursor()

        data_to_insert: List[Tuple[str, str, str, List[float], List[float]]] = [
            (filename, chunk, split_strategy, embedding, embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]

        template = '(%s, %s, %s, %s::REAL[], vector_norm(%s::REAL[]))'

        execute_values(
            cursor,
            "INSERT INTO documents (filename, chunk_text, split_strategy, embedding, embedding_norm) VALUES %s",
            data_to_insert,
            template=template
        )

        connection.commit()
        logger.info(f"Successfully stored {len(chunks)} chunks for '{filename}' in a single batch.")
        return True

    except (DatabaseError, psycopg2.Error) as e:
        if connection:
            connection.rollback()
        logger.error(f"Failed to store data to database for '{filename}': {str(e)}")
        return False

    finally:
        if connection:
            if cursor:
                cursor.close()
            connection.close()


def process_document(file_path: str) -> bool:
    """
    Process a document through the complete indexing pipeline.
    This is an idempotent operation: it first deletes any existing data for the file.

    Args:
        file_path (str): Path to document file (PDF or DOCX)

    Returns:
        bool: True if document was successfully processed and stored, False otherwise
    """
    logger.info(f"Starting document processing pipeline for: {file_path}")

    try:
        # Step 1: Delete any existing data for this file to prevent duplicates.
        logger.info(f"Clearing any existing data for '{file_path}' before indexing.")
        if not delete_document_data(filename=file_path):
            logger.error(f"Failed to clear old data for '{file_path}'. Aborting.")
            return False

        # Step 2: Load and process the new document data.
        text = load_document(file_path)
        chunks = chunk_by_paragraphs(text)

        # Proactive memory management: free the large 'text' object as it's no longer needed.
        del text
        logger.info("Freed 'text' object from memory.")

        embeddings = generate_embeddings(chunks)
        db_success = store_chunks_to_db(chunks, embeddings, file_path)

        # Proactive memory management: free the large lists after they are stored in the DB.
        del chunks, embeddings
        logger.info("Freed 'chunks' and 'embeddings' objects from memory.")

        if db_success:
            logger.info(f"Document processing completed successfully for: {file_path}")
            return True
        else:
            logger.error(f"Failed to store document in database: {file_path}")
            return False

    except (FileNotFoundError, ValueError, DocumentProcessingError, EmbeddingGenerationError) as e:
        logger.error(f"A known error occurred during document processing: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred processing document '{file_path}': {e}")
        return False
