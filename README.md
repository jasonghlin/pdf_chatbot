# RAG PDF Chatbot with Llama3

This repository provides a Streamlit application that demonstrates a Retrieval-Augmented Generation (RAG) system for chatting with PDF documents. It supports **multi-modal embedding** (text and images) using CLIP, stores embeddings in a Qdrant vector database, and uses an Ollama LLM (Llama3) for generating answers.

---

## Table of Contents
- [RAG PDF Chatbot with Llama3](#rag-pdf-chatbot-with-llama3)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
    - [Step 2: Build docker image](#step-2-build-docker-image)
    - [Step 3: Install and configure Qdrant](#step-3-install-and-configure-qdrant)
    - [Step 4: Setup Ollama LLM](#step-4-setup-ollama-llm)
  - [Usage](#usage)
  - [Key Components](#key-components)
  - [How it Works](#how-it-works)

---

## Overview

The **RAG PDF Chatbot** allows you to:
1. Upload one or more PDF files.
2. Parse and embed their text content, tables, images, and vector graphics.
3. Store these embeddings in a [Qdrant](https://qdrant.tech/) vector store.
4. Use an LLM (Ollama's Llama3) to perform retrieval-augmented question answering over the uploaded PDFs.

By leveraging CLIP embeddings for images and HuggingFace embeddings for text, this example demonstrates a **multi-modal** retrieval approach. It can handle text-based content, as well as table data and images that exist within PDF documents.

---

## Features

- **Multi-modal Embedding**: Uses CLIP for image/vector graphics embedding and HuggingFace model for text/table embedding.
- **PDF Parsing**: Extracts textual content, tables, and images from PDFs using [pdfplumber](https://github.com/jsvine/pdfplumber) and [PyMuPDF (fitz)](https://pypi.org/project/PyMuPDF/).
- **Vector Store**: Manages embeddings in Qdrant, supporting both text and image collections.
- **Retrieval-Augmented QA**: Retrieves relevant documents from Qdrant and uses Ollama’s Llama3 model to generate an answer.
- **Streamlit UI**: Provides an interactive interface for uploading PDFs and asking questions.

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/pdf_chatbot.git
cd pdf_chatbot
```

### Step 2: Build docker image
```bash
docker image build -t chatpdf .
```

### Step 3: Install and configure Qdrant
- Follow the [Qdrant installation guide](https://qdrant.tech/documentation/guides/installation/) to get your local Qdrant instance running on localhost:6333.
- Ensure it is up and running before starting the chatbot.

### Step 4: Setup Ollama LLM
- Make sure Ollama is installed and running on http://localhost:11434 (default).
- Confirm the Llama3 model is installed in Ollama.

## Usage

1.  Run container

    On Linux/Mac: 
    ```bash
    docker container run -d -v ~/.cache:/root/.cache -p 8501:8501 chatpdf
    ```
    Access on http://0.0.0.0:8501
    (Be patient when start application first time, it takes time to download embedding model blevlabs/stella_en_v5)

2. Upload PDFs in the sidebar to process. Once processed, the text chunks, tables, and images will be embedded into Qdrant.
3. Ask a question in the main UI text box. The system will:
    * Retrieve relevant chunks and images from Qdrant.
    * Pass them to the LLM (Ollama Llama3) for context-augmented answer generation.
    * Display both the answer and the context documents (sources)
4. You may need to restart Qdrant when embedding different PDFs

## Key Components

`RAGPDFChatbot`

This is the main class that orchestrates:

1. Initialization of all major components (embeddings, Qdrant client, Ollama LLM).
2. create_qdrant_collection: Sets up separate collections for text and images in Qdrant.
3. process_pdf:
    * Parses PDFs using pdfplumber and PyMuPDF.
    * Extracts text, tables, images, and vector graphics.
    * Splits text into chunks and embeds them (using HuggingFace).
    * Embeds images/vector graphics (using CLIP).
    * Upserts embeddings into two Qdrant collections (_text and _images).
4. retrieval_qa:
    * Retrieves relevant text + image chunks from Qdrant.
    * Passes these to Ollama’s Llama3 model for final answer generation.
    * Returns both the generated answer and the retrieval context.

`CLIPTextEmbeddings`

A simple class to generate text embeddings using CLIP. This is used specifically to handle image-based queries, allowing us to perform cross-modal retrieval (e.g., searching for images by text queries).

## How it Works

1. User uploads PDF(s) from the Streamlit sidebar.
2. RAGPDFChatbot:
    * Calls _extract_pdf_content to read text, tables, and images from each PDF.
    * Splits the text content into manageable chunks using TokenTextSplitter.
    * Embeds the text with a HuggingFace model.
    * Embeds images/vector graphics with a CLIP model.
    * Upserts all embeddings into Qdrant.
3. User enters a query in the text box:
   * The chatbot performs retrieval on both text and image collections in Qdrant.
   * The relevant chunks are combined and sent to Ollama’s Llama3.
   * The LLM returns an answer leveraging the retrieved context.
4. Answer and Source Documents are displayed in the Streamlit UI.