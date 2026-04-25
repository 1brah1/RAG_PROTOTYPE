# Project Plan: Edge AI RAG Prototype for NVIDIA Internship

## Objective
Develop a localized Retrieval-Augmented Generation (RAG) system optimized for an 8GB VRAM GPU (RTX 4060). The architecture will be designed to mimic the NVIDIA NIM (Inference Microservices) workflow, using quantized models and a high-performance vector database.

## Phase 1: Environment & Dependency Setup 
- [ ] **Install Core Dependencies:**
  - Python 3.10+
  - `langchain`, `langchain-community`, `chromadb`, `pypdf`, `sentence-transformers`
- [ ] **GPU Toolkit:** Ensure NVIDIA Drivers and `nvidia-smi` are accessible.
- [ ] **Model Server:** Install **Ollama** as the local inference engine (best for 8GB VRAM).
- [ ] **Pull Models:**
  - `ollama pull llama3.1:8b-instruct-q4_K_M` (Fits in ~4.7GB)
  - `ollama pull mxbai-embed-large` (Efficient embedding model)

## Phase 2: Data Ingestion & Vector Storage 
- [ ] **Dataset:** Download a technical PDF (e.g., NVIDIA Jetson Orin NX Datasheet).
- [ ] **Document Loader:** Implement `PyPDFLoader` to parse the text.
- [ ] **Recursive Text Splitting:** Split text into 1000-character chunks with 200-character overlap for context retention.
- [ ] **Vector Database:** Initialize **ChromaDB** with the embedding model. Store chunks as persistent vectors.

## Phase 3: RAG Chain Development
- [ ] **Inference Class:** Create a wrapper to connect LangChain to the Ollama local API.
- [ ] **Prompt Engineering:** Design a "System Prompt" that forces the model to use only provided context (NVIDIA style).
- [ ] **Retrieval Logic:** Implement a `similarity_search` that pulls the top 3 relevant chunks.
- [ ] **The Chain:** Connect Retrieval -> Prompt -> LLM.

## Phase 4: Optimization & Benchmarking 
- [ ] **Memory Management:** Implement a script to monitor VRAM usage during inference using `nvidia-smi`.
- [ ] **Performance Metrics:** Measure "Time to First Token" and "Total Generation Time".
- [ ] **Quantization Impact:** Document the speed of the 4-bit model vs. 8-bit (if space allows).

## Phase 5: Documentation for CV 
- [ ] Create a public repo with the name "RAG_PROTOTYPE", remember to not share any secrets there
- [ ] **GitHub README:** - Add a "Hardware Specs" section (RTX 4060 8GB).
  - Include a "Performance" table showing latency results.
  - Diagram the architecture: `Data -> Embedding -> Vector DB -> LLM -> Response`.
  - Use the phrase: *"Architected for NIM-compatibility."*
  - Create a public repo with the name "RAG_PROTOTYPE", remember to not share any secrets there

## Technical Constraints for Agent
- **VRAM Limit:** Total VRAM usage must stay under 7.5GB to avoid system lag.
- **Backend:** Use OpenAI-compatible API endpoints where possible to ensure the code is "NIM-ready."
- **Precision:** Default to 4-bit quantization for the LLM to ensure stability.
