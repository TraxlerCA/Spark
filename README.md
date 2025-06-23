
# Local Personal AI Assistant with RAG

A fully local, privacy-friendly AI assistant that runs on your machine and can use your own documents to answer questions, thanks to Retrieval-Augmented Generation (RAG). Powered by [Ollama](https://ollama.com/) for LLMs and ChromaDB for document search.

## Features

* **Local, private, and secure:** No cloud, no external data leaks.
* **Personal knowledge base:** Ingest your own documents and query them using natural language.
* **Flexible LLM backend:** Use any model available in Ollama (default: Gemma 3:4b).
* **RAG pipeline:** Relevant context from your docs is retrieved and injected into the AI’s answer.
* **Streamlit UI and CLI:** Choose between an interactive web interface or command-line chat.

---

## Quickstart

### 1. Prerequisites

* Python 3.8+
* [Ollama](https://ollama.com/) (install and run locally)
* Git

### 2. Setup

Clone the repo and set up your environment:

```bash
git clone <your-repo-url>
cd <project-folder>
python -m venv venv
# Activate (Windows):
venv\Scripts\activate
# Activate (macOS/Linux):
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Ollama and Download a Model

Start Ollama and pull a model (e.g., Gemma 3.4b or DeepSeek):

```bash
ollama serve
ollama pull gemma:3.4b
```

You can change models in the config or via the UI.

### 4. Add Your Documents

Put any documents you want to query into the `source_documents/` directory (default; changeable via config).

### 5. Ingest Documents

Build your personal vector database:

```bash
python ingest.py
```

Add new docs any time and re-run `ingest.py` to update your knowledge base.

### 6. Chat with Your Assistant

**Web UI (recommended):**

```bash
streamlit run app.py
```

* Visit `http://localhost:8501` in your browser.
* Toggle RAG mode to query your own docs or just use the model.

**Command Line:**

```bash
python main.py
```

* Type your question, hit enter, and get answers (with sources if using RAG).

---

## Configuration

All settings are managed in `config.py` and can be overridden by environment variables or a `.env` file.

**Key config options:**

* `source_dir`: Folder for your docs (`source_documents/`)
* `persist_dir`: Where the vector database lives
* `collection`: Name of your ChromaDB collection
* `embed_model_id`: Which HuggingFace embedding model to use
* `ollama_host`/`ollama_port`: Ollama server details
* `ollama_model`: Default LLM (e.g. `gemma3:4b`)
* `similarity_top_k`: How many document chunks to retrieve

See `config.py` for full details.

---

## How it works

* **Ingestion (`ingest.py`):** Chunks and embeds your documents, stores them locally in ChromaDB.
* **Retrieval (`retriever.py`):** At question time, finds relevant pieces of your docs using vector search.
* **Prompt Building:** Merges conversation, system prompt, and retrieved context for the LLM.
* **Generation:** Sends the prompt to Ollama, which runs your chosen LLM locally and streams back a response.
* **UI:** Choose between Streamlit web app (`app.py`) or CLI (`main.py`).

---

## Requirements

All dependencies are pinned in `requirements.txt`. Main dependencies:

* `streamlit` (web UI)
* `ollama` (LLM backend, install separately)
* `chromadb` (vector DB)
* `llama-index` (RAG orchestration)
* `huggingface` (embeddings)
* `typer` (CLI)
* `pydantic` (configuration)

---

## Troubleshooting

* **Vector store not found:** Run `python ingest.py` before chatting.
* **Ollama connection errors:** Ensure Ollama is running and you’ve downloaded a model.
* **No context found:** Make sure your docs are in the correct folder and properly ingested.

---

## Extending

* Add support for other embedding models (via `config.py`)
* Change the default LLM model easily (Streamlit sidebar or config)
* Extend prompt formatting or system behavior via `history_utils.py`
* Use your own folder structure by editing the config

---

## FAQ

**Q: Are my documents or questions ever sent to the cloud?**
A: No, everything stays on your machine.

**Q: Can I use models other than Gemma?**
A: Yes, any Ollama-supported model (e.g., Llama3, DeepSeek, Phi3).

**Q: How do I reset my vector store?**
A: Run `python -m ingest.py --reset` to wipe and rebuild.

---

## Credits

* [Ollama](https://ollama.com/) for fast, local LLMs
* [ChromaDB](https://www.trychroma.com/) for vector storage
* [LlamaIndex](https://www.llamaindex.ai/) for RAG utilities
* [Streamlit](https://streamlit.io/) for the UI

---

Feel free to fork, extend, or customize for your own private AI workflow.

---