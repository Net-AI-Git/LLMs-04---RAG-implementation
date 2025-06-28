import time
from typing import List, Dict, Union

import google.generativeai as genai

from utils.db_utils import load_configuration
from utils.logging_utils import get_logger, EmbeddingGenerationError

# Setup colored logger for this module
logger = get_logger(__name__)


def chunk_by_paragraphs(text: str) -> List[str]:
    """
    Split text into chunks by paragraphs.

    Args:
        text (str): Raw text content to split

    Returns:
        List[str]: List of paragraph chunks

    Raises:
        ValueError: If text is empty
    """
    if not text or not text.strip():
        logger.error("Text is empty")
        raise ValueError("Cannot chunk empty text")

    # Split by double newlines and filter out empty chunks
    cleaned_chunks = [
        chunk.strip()
        for chunk in text.split('\n\n')
        if chunk.strip()
    ]

    logger.info(f"Split into {len(cleaned_chunks)} paragraphs")
    return cleaned_chunks


def _embed_with_retry(model: str, content: Union[str, List[str]], max_retries: int = 3) -> Dict:
    """
    Call Gemini embed_content API with retry mechanism.

    Args:
        model: Embedding model name
        content: Text or list of texts to embed
        max_retries: Maximum number of retry attempts

    Returns:
        API response with embeddings

    Raises:
        EmbeddingGenerationError: If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            return genai.embed_content(model=model, content=content)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"All {max_retries} embedding attempts failed: {str(e)}")
                raise EmbeddingGenerationError(f"Failed after {max_retries} attempts: {str(e)}")

            # Calculate wait time with exponential backoff
            wait_time = 2 ** attempt  # 1, 2, 4 seconds
            logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)


def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for text chunks using Gemini API.

    Args:
        chunks (List[str]): List of text chunks to embed

    Returns:
        List[List[float]]: List of embedding vectors

    Raises:
        ValueError: If chunks is empty
        EmbeddingGenerationError: If API request fails
    """
    if not chunks:
        logger.error("Chunks list is empty")
        raise ValueError("Cannot generate embeddings for empty chunks list")

    try:
        config = load_configuration()
        api_key = config['api_key']
        embedding_model = config['embedding_model']
        genai.configure(api_key=api_key)
    except Exception as e:  # Catching generic exception from config loading
        logger.error(f"Configuration error: {e}")
        raise EmbeddingGenerationError(f"Configuration error: {e}")

    logger.info(f"Generating embeddings for {len(chunks)} chunks")

    BATCH_SIZE = 10
    embeddings = []

    try:
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            batch = chunks[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // BATCH_SIZE + 1}/{(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE} "
                f"(chunks {batch_start + 1}-{batch_end})")

            batch_result = _embed_with_retry(
                model=embedding_model,
                content=batch
            )

            # The API response is always a list of embeddings, so we can always use extend.
            embeddings.extend(batch_result['embedding'])

            if batch_end < len(chunks):
                time.sleep(0.1)

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise EmbeddingGenerationError(f"Failed to generate embeddings: {str(e)}")

    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings
