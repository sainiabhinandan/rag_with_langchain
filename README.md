# rag_with_langchain
Learning to use rag with Langchain and creating data pipeline for embedding and data parsing :)

# 📚 RAG with LangChain, Faiss, and Groq

This repository contains a simple, local implementation of a **Retrieval-Augmented Generation (RAG)** system built using the **LangChain** ecosystem (specifically its document loaders and text splitters), **Sentence Transformers** for embeddings, and **Faiss** for efficient vector storage and retrieval. The system uses a Large Language Model (LLM) to summarize retrieved context and answer user queries.

---

## 🚀 Key Features

* **Document Loading:** Loads PDF files using `PyMuPDFLoader`.(other data files can be used as well)
* **Text Processing:** Uses **Recursive Character Text Splitter** for optimal document chunking.
* **Vector Embedding:** Generates embeddings using a **Sentence Transformer** model (`all-MiniLM-L6-v2`).
* **Vector Storage:** Employs **Faiss** for high-performance indexing and similarity search, ensuring persistence by saving the index and metadata.
* **Contextual Search:** Retrieves the most relevant document chunks based on a user query.
* **LLM Integration:** Uses the **ChatGroq_API** for LLM (defaulting to `llama-3.1-8b-instant`) to generate a final summary grounded in the retrieved context.

---

## ⚙️ Prerequisites

To run this project, you need:

* Python 3.13.5
* A **GROQ API Key** (set as an environment variable or manually within `search.py`).

---

## 🛠️ Project Structure and Components

The RAG pipeline is organized into five main files:

### 1. `data_loader.py`

Handles the loading of raw documents from the specified data directory (`../data`).

| Function | Description |
| :--- | :--- |
| `load_all_documents(data_dir)` | Scans the directory for documents (currently only **PDFs** using `PyMuPDFLoader`) and loads them into a list of documents. |

### 2. `embedding.py`

Manages the document chunking and embedding generation process.

| Class/Method | Description |
| :--- | :--- |
| `EmbeddingPipeline` | Initializes the embedding model and configuration (e.g., `chunk_size` and `chunk_overlap`). |
| `doc_chunker(documents)` | Splits raw documents into smaller, overlapping chunks using **`RecursiveCharacterTextSplitter`**. |
| `chunk_embedder(chunks)` | Encodes the text chunks into numerical vectors (**embeddings**) using the **Sentence Transformer** model. |

### 3. `vectorstore.py`

Implements the **Faiss-based vector store** for persistent, fast storage and search.

| Class/Method | Description |
| :--- | :--- |
| `FaissVectorStore` | Initializes the index and manages the persistent directory. |
| `build_from_documents(documents)` | Orchestrates the chunking and embedding process, then builds the **Faiss index**. |
| `query(query_text)` | Converts the user's query into an embedding and performs a similarity search (`faiss.IndexFlatL2`) to retrieve the top $k$ relevant document chunks. |
| `save()` / `load()` | Handles the serialization and deserialization of the Faiss index and chunk metadata using `pickle`. |

### 4. `search.py`

The main logic for the RAG chain, from retrieval to final answer generation.

| Class/Method | Description |
| :--- | :--- |
| `RAGSearch` | Initializes the `FaissVectorStore` and the LLM. It automatically loads existing stores or builds them if the index files are missing. |
| `search_and_summarize(query, top_k)` | 1. Queries the vector store to get the top $k$ context chunks. 2. Constructs a **prompt** with the user query and the retrieved context. 3. Sends the prompt to the Groq LLM for a summarized response. |

### 5. `app.py`

The entry point for the application, demonstrating the full workflow and providing a basic command-line interface.

| Section | Description |
| :--- | :--- |
| `if __name__ == "__main__":` | Runs the main loop. It loads documents, loads/builds the vector store, and continuously accepts user queries, performs RAG search, and prints the resulting summary until the user types 'exit'. |

---

## ▶️ Getting Started

1.  **Clone the repository and install dependencies** (e.g., `langchain`, `faiss-cpu`, `sentence-transformers`, `groq`, etc.).
2.  **Create a `data` directory** in the root of the project and place your **PDF files** inside it.
3.  **Ensure your GROQ API key** is configured.
4.  **Run `app.py`** to start the interactive query session:

```bash
python app.py        
