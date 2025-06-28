import numpy as np
import psycopg2
from typing import List, Dict, Tuple

from utils.logging_utils import get_logger, DatabaseSearchError, DatabaseError, EmbeddingError, EmbeddingGenerationError
from utils.db_utils import get_db_connection, ensure_database_schema
from utils.shared_utils import chunk_by_paragraphs, generate_embeddings

# Setup colored logger for this module
logger = get_logger(__name__)


def create_query_embeddings(query: str) -> Tuple[List[str], List[List[float]]]:
    """
    Generate embeddings for user query using paragraph-based chunking.

    Args:
        query (str): User search query text

    Returns:
        Tuple[List[str], List[List[float]]]: Tuple containing chunks and their embedding vectors

    Raises:
        ValueError: If query is empty
        EmbeddingError: If chunking or embedding generation fails
    """
    logger.info(f"Creating embeddings for query: '{query}'")

    try:
        chunks = chunk_by_paragraphs(query)
        embeddings = generate_embeddings(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks")
        return chunks, embeddings
    except ValueError as e:
        logger.error(f"Query validation error: {e}")
        raise ValueError(f"Invalid query: {e}")
    except EmbeddingGenerationError as e:
        logger.error(f"Failed to create query embeddings: {e}")
        raise EmbeddingError(f"Failed to create query embeddings: {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating query embeddings: {e}")
        raise EmbeddingError(f"Unexpected error: {e}")


def _merge_search_results(all_search_results: List[List[Dict]], top_k: int) -> List[Dict]:
    """
    Merge multiple search results using round-robin approach. (Private helper function)

    Args:
        all_search_results (List[List[Dict]]): Results from multiple searches
        top_k (int): Maximum number of final results

    Returns:
        List[Dict]: Merged results without duplicates
    """
    final_results = []
    seen_chunks: Dict[str, int] = {}  # Track duplicates by chunk_text -> index in final_results

    # Round-robin through all search results
    for round_index in range(top_k):
        if len(final_results) >= top_k:
            break

        for search_index, search_results in enumerate(all_search_results):
            if len(final_results) >= top_k:
                break
            if round_index >= len(search_results):
                continue

            result = search_results[round_index]
            chunk_text = result['chunk_text']

            if chunk_text in seen_chunks:
                existing_index = seen_chunks[chunk_text]
                if result['similarity_score'] > final_results[existing_index]['similarity_score']:
                    final_results[existing_index] = result
                continue

            seen_chunks[chunk_text] = len(final_results)
            final_results.append(result)

    logger.info(f"Merged {len(final_results)} unique results from {len(all_search_results)} searches")
    return final_results


def format_results(results: List[Dict]) -> str:
    """
    Format search results for user display.

    Args:
        results (List[Dict]): Search results from similarity_search

    Returns:
        str: Formatted text for user display
    """
    if not results:
        return "No results found for your search."

    formatted_text = f"Search Results ({len(results)} results):\n"
    formatted_text += "=" * 50 + "\n\n"

    for result_index, result in enumerate(results, 1):
        similarity_percentage = result['similarity_score'] * 100
        formatted_text += f"{result_index}. Source: {result['filename']} | Similarity: {similarity_percentage:.1f}%\n"
        formatted_text += f"{result['chunk_text']}\n\n"
        formatted_text += "-" * 30 + "\n\n"

    logger.info(f"Formatted {len(results)} results for display")
    return formatted_text


def _search_single_embedding(embedding: List[float], connection: psycopg2.extensions.connection, top_k: int) -> List[
    Dict]:
    """
    Search for similar chunks using a single embedding. (Private helper function)

    Args:
        embedding (List[float]): Single embedding vector
        connection (psycopg2.extensions.connection): PostgreSQL database connection
        top_k (int): Number of results to return

    Returns:
        List[Dict]: Search results for this embedding

    Raises:
        DatabaseSearchError: If database search fails
    """
    cursor = None
    try:
        cursor = connection.cursor()
        query_embedding_str = '{' + ','.join(map(str, embedding)) + '}'

        # Calculate the norm and immediately cast it to a standard Python float
        query_norm = float(np.linalg.norm(np.array(embedding, dtype=np.float32)))

        # Handle potential division by zero if query norm is zero.
        if query_norm == 0:
            logger.warning("Query vector norm is zero, cannot compute similarity.")
            return []

        search_query_sql = """
        SELECT
            chunk_text,
            filename,
            split_strategy,
            dot_product(embedding, %s::REAL[]) / (embedding_norm * %s) as similarity_score
        FROM documents
        WHERE embedding_norm > 0
        ORDER BY similarity_score DESC
        LIMIT %s
        """

        cursor.execute(search_query_sql, (query_embedding_str, query_norm, top_k))
        results = cursor.fetchall()

        similarities = [{
            'chunk_text': row[0],
            'filename': row[1],
            'split_strategy': row[2],
            'similarity_score': row[3]
        } for row in results]

        logger.info(f"Single embedding search returned {len(similarities)} results")
        return similarities

    except psycopg2.Error as e:
        logger.error(f"Single embedding search failed: {str(e)}")
        raise DatabaseSearchError(f"Failed to search database: {str(e)}")
    finally:
        if cursor:
            cursor.close()


def similarity_search(query_embeddings: List[List[float]], top_k: int = 5) -> List[Dict]:
    """
    Search for similar chunks in database using multiple embeddings.

    Args:
        query_embeddings (List[List[float]]): List of query embedding vectors.
        top_k (int): Maximum number of results to return (default: 5).

    Returns:
        List[Dict]: List of search results.

    Raises:
        DatabaseSearchError: If database search fails.
    """
    if not query_embeddings:
        logger.error("No embeddings provided for search")
        raise ValueError("Cannot search without embeddings")

    logger.info(f"Searching with {len(query_embeddings)} embeddings, top_k={top_k}")

    connection = None
    try:
        connection = get_db_connection()
        ensure_database_schema(connection)  # Ensures DB is ready for searching

        all_search_results = []
        for i, embedding in enumerate(query_embeddings):
            search_results = _search_single_embedding(embedding, connection, top_k)
            all_search_results.append(search_results)
            logger.info(f"Search for embedding #{i + 1}: Found {len(search_results)} results")

        final_results = _merge_search_results(all_search_results, top_k)

        if not final_results:
            logger.warning("No similar chunks found for any of the query embeddings.")

        return final_results

    except (DatabaseError, psycopg2.Error) as e:
        logger.error(f"A database error occurred during similarity search: {e}")
        raise DatabaseSearchError(f"A database error occurred: {e}")
    finally:
        if connection:
            connection.close()


def search_query(user_question: str) -> str:
    """
    Search for relevant documents using the complete search pipeline.

    Args:
        user_question (str): User's search query

    Returns:
        str: Formatted search results ready for display, or error message if search failed
    """
    logger.info(f"Starting search pipeline for query: '{user_question}'")

    try:
        chunks, embeddings = create_query_embeddings(user_question)

        # Proactive memory management: free the 'chunks' object as it's no longer needed.
        del chunks
        logger.info("Freed query 'chunks' object from memory.")

        results = similarity_search(embeddings)
        formatted_results = format_results(results)

        if results:
            logger.info(f"Search completed successfully: {len(results)} results found")
        else:
            logger.warning("Search completed but no results found")

        return formatted_results

    except (ValueError, EmbeddingError, DatabaseSearchError) as e:
        logger.error(f"A known error occurred during search: {e}")
        return f"Search failed: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during search: {e}")
        return "An unexpected error occurred. Please try again."
