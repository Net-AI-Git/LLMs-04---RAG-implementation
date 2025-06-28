# Advanced RAG Document Search System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-stable-green.svg)

### A sophisticated command-line tool for indexing and performing semantic searches on local documents using a Retrieval-Augmented Generation (RAG) architecture.

This project provides a robust framework for transforming a collection of local documents (`.pdf`, `.docx`) into a searchable knowledge base. It leverages the power of Google's Gemini language model to generate high-quality embeddings and a PostgreSQL database for efficient vector storage and similarity search. The system is designed with a clean, modular architecture, ensuring reliability, efficiency, and ease of use.

---

## 📋 Table of Contents
1.  [Key Features](#-key-features)
2.  [Tech Stack](#-tech-stack)
3.  [Project Architecture](#-project-architecture)
4.  [Installation Guide](#-installation-guide)
5.  [Usage](#-usage)
6.  [Results & Evaluation](#-results--evaluation)
7.  [Future Work](#-future-work)
8.  [Contact](#-contact)

---

## ✨ Key Features

* **📄 Multi-Format Document Support:** Ingests and processes both `.pdf` and `.docx` files.
* **🧠 Advanced Text Processing:**
    * Intelligent paragraph-based chunking.
    * Automatic normalization for inconsistent text layouts extracted from PDF files.
* **🚀 Efficient Indexing:**
    * Utilizes Google's `text-embedding-004` for high-quality semantic embeddings.
    * Employs bulk-insert operations for fast and efficient data storage.
    * Idempotent processing automatically handles updates by clearing old data before indexing new versions of a file.
* **🔍 Powerful Semantic Search:**
    * Calculates cosine similarity directly within PostgreSQL for maximum performance.
    * Handles multi-chunk queries by merging results for comprehensive answers.
* **🗄️ Full Database Management:**
    * Interactive CLI for listing indexed documents, deleting specific files, or clearing the entire database.
* **⚙️ Developer-Friendly:**
    * Includes a `--debug` mode for detailed console logging.
    * Proactive memory management for handling large files efficiently.
    * Clean, modular, and well-documented codebase.

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3.8+
* **LLM & Embeddings:** Google Gemini API (`google-generativeai`)
* **Database:** PostgreSQL
* **Core Libraries:**
    * `psycopg2-binary`: PostgreSQL driver for Python.
    * `numpy`: For numerical vector operations.
    * `PyPDF2`: For PDF text extraction.
    * `python-docx`: For DOCX text extraction.
    * `python-dotenv`: For managing environment variables.
* **Development Tools:**
    * `argparse`: For command-line argument parsing.

---

## 🏗️ Project Architecture

The project is built on a clean, modular architecture to ensure separation of concerns and maintainability.

* `document_search_cli.py`: The main entry point of the application. Handles all user interaction, menus, and commands. It orchestrates calls to the other modules but contains no core business logic.
* `index_documents.py`: Contains the logic for the indexing pipeline: loading files, extracting text, and storing data.
* `search_documents.py`: Contains the logic for the search pipeline: creating query embeddings, performing the similarity search, and formatting results.
* `shared_utils.py`: A utility module for shared functionalities like text chunking and embedding generation, used by both indexing and search processes.
* `db_utils.py`: A dedicated module for all database interactions, including connection management, schema validation, and data manipulation (delete/list).
* `logging_utils.py`: A centralized module for configuring logging and defining custom exceptions, allowing for consistent error handling and debug capabilities across the application.

---

## ⚙️ Installation Guide

Follow these steps to set up and run the project locally.

### 1. Prerequisites
* **Python 3.8 or higher.**
* **PostgreSQL Server:** Ensure you have a running instance of PostgreSQL.

### 2. Clone the Repository
```bash
git clone [https://github.com/Net-AI-Git/LLMs-04---RAG-implementation.git](https://github.com/Net-AI-Git/LLMs-04---RAG-implementation.git)
cd LLMs-04---RAG-implementation
```

### 3. Set Up the Database
You must manually create the database before running the application for the first time.
1.  Connect to your PostgreSQL server (using `psql`, `pgAdmin`, or another tool).
2.  Execute the following SQL command to create the database:
    ```sql
    CREATE DATABASE rag_database;
    ```
The application will automatically create the necessary tables and functions inside this database on its first run.

### 4. Configure Environment Variables
1.  In the project's root directory, create a file named `.env`.
2.  Copy the content below into the `.env` file and fill in your details.

    ```env
    # --- Google Gemini API Configuration ---
    # Get your key from Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

    # --- API Settings ---
    # The embedding model to use. 'text-embedding-004' is a strong default.
    EMBEDDING_MODEL=text-embedding-004

    # --- PostgreSQL Database Configuration ---
    # The full database connection URL.
    # Format: postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
    # IMPORTANT: If your password contains special characters (like @, #, $),
    # it must be URL-encoded.
    POSTGRES_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_database
    ```

**How to create your `POSTGRES_URL`:**
* Replace `YOUR_PASSWORD` with your actual PostgreSQL password.
* If your password has special characters, use a URL encoder to encode it. For example, a password like `p@ss#word` would become `p%40ss%23word`.
* The other parts (`postgres`, `localhost`, `5432`, `rag_database`) should match your local setup.

### 5. Install Dependencies
Install all required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

The application is run from the command line.

### Running the Application
* **Normal Mode (for users):**
    This mode provides a clean interface without internal logs.
    ```bash
    python document_search_cli.py
    ```

* **Debug Mode (for developers):**
    This mode prints detailed logs to the console, which is useful for troubleshooting.
    ```bash
    python document_search_cli.py --debug
    ```

### Example Workflow
1.  **Launch the application:** `python document_search_cli.py`
2.  **Add Documents:**
    * Select option `1. 📄 Add New Document`.
    * Choose your preferred method: browse folders, select a file from a specific path, or process an entire folder at once.
3.  **Search:**
    * Select option `2. 🔍 Search Documents`.
    * Enter your query, for example: `What are the main challenges of large language models?`
    * The system will return the most relevant text chunks from your indexed documents.
4.  **Manage Database:**
    * Select option `3. 🗄️ Manage Database`.
    * From here, you can list all indexed files, delete a specific one, or clear the entire database.

### Example Session
Below are screenshots demonstrating a typical user session, from adding a document to searching and managing the database.

<!--
INSTRUCTION FOR YOU, NETANEL:
This is the perfect place to add screenshots of your CLI in action.
For example:

**Main Menu:**
![Main Menu Screenshot](path/to/your/main_menu_screenshot.png)

**Search Results:**
![Search Results Screenshot](path/to/your/search_results_screenshot.png)
-->

---

## 📊 Results & Evaluation

This section evaluates the system's effectiveness and performance based on its architectural design and provides a framework for quantitative analysis.

### Performance by Design

The system was engineered for efficiency and scalability through several key architectural decisions:

* **In-Database Vector Search:** Instead of pulling all vectors into memory for comparison, the cosine similarity calculation is performed directly within PostgreSQL using custom SQL functions (`dot_product`, `vector_norm`). This dramatically reduces data transfer and leverages the database's optimized execution engine, resulting in significantly faster search times.
* **Pre-calculated Norms:** The vector norm for each document chunk is calculated once during indexing and stored in the `embedding_norm` column. This avoids redundant calculations during search operations, further optimizing query speed.
* **Efficient Batch Processing:** Both the embedding generation (using the Gemini API) and database insertion (using `psycopg2.extras.execute_values`) are performed in batches. This minimizes the number of network round-trips, leading to a much faster and more robust indexing process.
* **Proactive Memory Management:** The system explicitly frees large data objects (like full document text and embedding lists) from memory as soon as they are no longer needed, ensuring it can handle large documents without crashing.

---

## 🔮 Future Work
* **Support More File Formats:** Extend the system to handle `.txt`, `.html`, and other common file types.
* **Web Interface:** Build a simple web UI (using Streamlit or Flask) to make the system more accessible to non-technical users.
* **Advanced RAG Strategies:** Implement more sophisticated techniques like re-ranking, query expansion, or using a generator model to synthesize final answers.
* **Async Processing:** For very large files or folders, move the indexing process to a background task to avoid blocking the UI.

---

## 📬 Contact
* **Author:** Netanel Itzhak
* **LinkedIn:** [www.linkedin.com/in/netanelitzhak](https://www.linkedin.com/in/netanelitzhak)
* **Email:** ntitz19@gmail.com
* **GitHub:** [https://github.com/Net-AI-Git](https://github.com/Net-AI-Git)
