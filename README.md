# OmniDoc – Multimodal RAG & AI Document Intelligence

An AI-powered document question-answering system that explores two approaches to Retrieval-Augmented Generation (RAG): **Agentic RAG** and **RAG with Guardrails & Evaluation**.

OmniDoc allows users to interact with PDF documents using natural language, retrieve relevant information, and generate context-aware responses using Large Language Models (LLMs).

The repository contains two independent Streamlit applications, each demonstrating a different approach to building more useful and reliable RAG systems.

## 🚀 Live Applications

| Application | Description | Live Demo |
|---|---|---|
| **01. Agentic Multimodal RAG** | Agent-based document retrieval and question answering | [Launch App](https://omnidoc-multimodal-rag-ksnmfuvbddhx636nfhicze.streamlit.app/) |
| **02. RAG with Guardrails & Evaluation** | RAG with response safeguards and evaluation capabilities | [Launch App](https://omnidoc-multimodal-rag-kkrddg7vex9e4vpnmcuicj.streamlit.app/) |

## 📌 Project Overview

### 1. Agentic Multimodal RAG

An agent-based RAG application designed to retrieve relevant information from uploaded PDF documents and answer user queries.

**Key Features**
- Upload and process PDF documents.
- Extract text and embedded images from PDFs.
- Split document content into smaller chunks.
- Generate embeddings using OpenAI or Google Gemini embedding models.
- Store and retrieve document information using ChromaDB.
- Use LangGraph to build an agent workflow.
- Integrate tools for document retrieval.
- Generate responses based on retrieved document context.

**Technologies:** Python, Streamlit, LangChain, LangGraph, ChromaDB, PyMuPDF, Pillow, OpenAI and Google Gemini.

### 2. RAG with Guardrails & Evaluation

A RAG application focused on improving response reliability through guardrails and evaluation techniques.

**Key Features**
- Upload and process PDF documents.
- Extract and split document text into chunks.
- Generate embeddings using OpenAI or Google Gemini.
- Perform semantic similarity search using FAISS.
- Generate context-aware answers using LLMs.
- Apply guardrail mechanisms to help identify potentially unsafe or inappropriate inputs and responses.
- Evaluate generated responses using LLM-based evaluation techniques.
- Explore retrieval and response evaluation metrics.

**Technologies:** Python, Streamlit, LangChain, FAISS, OpenAI, Google Gemini and NLP techniques.

## 🏗️ Project Architecture

The repository is organized into two independent applications:

```text
omnidoc-multimodal-rag/
│
├── 01-agentic-multimodal-rag/
│   ├── app.py
│   ├── agents.py
│   └── tools.py
│
├── 02-rag-guardrails-evaluation/
│   ├── app.py
│   ├── guardrails.py
│   └── evals.py
│
├── requirements.txt
└── README.md
```

## ⚙️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM Framework | LangChain |
| Agent Framework | LangGraph |
| LLM Providers | OpenAI, Google Gemini |
| Embedding Models | OpenAI Embeddings, Google Generative AI Embeddings |
| Vector Databases | ChromaDB, FAISS |
| PDF Processing | PyMuPDF, PyPDF |
| Image Processing | Pillow |
| Evaluation | LLM-based evaluation techniques |

## 🔄 How RAG Works

Both applications follow the fundamental RAG workflow, with different retrieval and response-generation approaches.

1. **Document Upload:** Users upload PDF documents.
2. **Text Extraction:** Document content is extracted.
3. **Chunking:** Extracted text is divided into smaller, manageable chunks.
4. **Embedding Generation:** Text chunks are converted into vector representations.
5. **Vector Storage:** Embeddings are stored in a vector database.
6. **Similarity Search:** Relevant chunks are retrieved based on the user's query.
7. **Response Generation:** An LLM uses the retrieved context to generate an answer.

The Agentic RAG application additionally uses an agent workflow to coordinate tool usage, while the second application incorporates guardrails and evaluation mechanisms.

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/omnidoc-multimodal-rag.git

cd omnidoc-multimodal-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application 1 – Agentic Multimodal RAG

```bash
cd 01-agentic-multimodal-rag

streamlit run app.py
```

### 5. Run Application 2 – RAG with Guardrails & Evaluation

Open another terminal, activate the virtual environment, and run:

```bash
cd 02-rag-guardrails-evaluation

streamlit run app.py
```

### 🔑 API Configuration

The applications use LLM and embedding providers such as OpenAI and Google Gemini.

Provide the required API credentials through the application's configured input mechanism. Ensure that the appropriate provider credentials are available for the selected models and features.

Never commit API keys, passwords, or other secrets to GitHub.

## 🎯 Learning Outcomes

Through this project, I explored and implemented concepts related to:

- Retrieval-Augmented Generation (RAG).
- Document processing and text chunking.
- Vector embeddings and semantic search.
- Vector databases and similarity retrieval.
- Agentic workflows using LangGraph.
- LLM integration using OpenAI and Google Gemini.
- Guardrails for more controlled LLM interactions.
- LLM-based response evaluation.
- Building and deploying interactive AI applications using Streamlit.

## 🔮 Future Enhancements

- Extend document support beyond PDFs.
- Improve visual understanding by passing extracted images directly to vision-capable LLMs.
- Introduce more robust prompt-injection detection.
- Improve evaluation using reliable, independently verified ground-truth datasets.
- Add conversational memory and multi-document retrieval.
- Implement more advanced agentic workflows.

## 👩‍💻 Author

**Shivani Singh**

Data Analytics & Data Science Enthusiast | Python | SQL | Power BI | Generative AI | Machine Learning

---

⭐ If you find this project interesting, feel free to explore the applications and the repository!
