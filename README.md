# 📊 PSX AI Assistance — AI-Powered Financial Report & Market Analysis

**PSX AI Assistance** is an intelligent financial analytics app that integrates **Pakistan Stock Exchange (PSX)** data, company fundamentals, and financial reports with **LLM-powered summarization**, **embedding-based retrieval (RAG)**, and **interactive visualizations**.

Built with **Streamlit**, **Ollama**, and **ChromaDB**, it enables users to:
- Fetch real-time market and company data from PSX APIs  
- Download and process company financial PDFs  
- Extract tables and text for semantic understanding  
- Summarize complex financial data into human-readable insights  
- Query embedded knowledge for interactive AI-assisted analysis  

---

## 🚀 Features

✅ **Real-Time Market Data**  
Fetches and displays live data for indices (KSE100, KSE30, PSXDIV20, ALLSHR)

✅ **Company Financial Details**  
Retrieves fundamentals, dividends, and K-Line data from PSX APIs

✅ **PDF Financial Reports Parsing**  
Automatically downloads recent company PDFs from PSX  
Extracts text and tables using `pdfplumber`

✅ **AI Summarization**  
Summarizes structured and unstructured financial data using Ollama models  
Supports multiple LLM backends (Qwen, DeepSeek, Kimi, GPT-OSS)

✅ **RAG-Enabled Conversational Assistant**  
Uses **ChromaDB** for local vector storage of summaries  
Users can chat with embedded financial knowledge

✅ **Data Visualization**  
Interactive candlestick charts and metrics using Plotly and Streamlit

---

## 🧠 System Architecture

📊 **PSX API + 📄 PDF Reports**  
  │  
  ▼  
**Data Collection Layer**  
*(Requests, BeautifulSoup)*  
  │  
  ▼  
**PDF Parsing & Structuring**  
*(pdfplumber, table extraction)*  
  │  
  ▼  
**LLM Summarization (Ollama)**  
Functions used:  
- `summarize_table()`  
- `summarize_ticker_detail()`  
- `summary_ai_assistance()`  
  │  
  ▼  
**Embedding (ChromaDB + embeddinggemma)**  
  │  
  ▼  
**RAG Query & AI Response**  
Functions used:  
- `ai_assistance()`  
- `embedd_prompt()`  
  │  
  ▼  
**Streamlit UI**  
Modules:  
- charts, metrics, chat interface




---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|-------------|
| **Frontend** | Streamlit, Plotly |
| **Backend** | Python 3.10+, Requests, BeautifulSoup, asyncio |
| **AI Models** | Ollama (Qwen, DeepSeek, Kimi, GPT-OSS), embeddinggemma |
| **Vector DB** | ChromaDB |
| **PDF Parsing** | pdfplumber, PyPDF |
| **Data Sources** | [PSX Terminal API](https://psxterminal.com), [DPS PSX](https://dps.psx.com.pk) |

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/psx-ai-assistance.git
cd psx-ai-assistance
```

## Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # (Linux/macOS)
venv\Scripts\activate       # (Windows)


## Run the Streamlit app
streamlit run main.py


## 🧩 Configuration
You may need to configure Ollama or adjust models:
response = ollama.chat(
    model="qwen3-coder:latest",  # or "deepseek-v3.1:671b-cloud"
    messages=[...]
)

## ChromaDB persistent collection is initialized automatically:
client = chromadb.PersistentClient()
collection = client.get_or_create_collection("my_db")

## 🧾 Core Functions Overview

| Function                            | Purpose                                                   |
| ----------------------------------- | --------------------------------------------------------- |
| `makeMarketRequest()`               | Fetch index data (KSE100, KSE30, etc.)                    |
| `makeApiRequest(ticker)`            | Fetch company fundamentals, dividends, and K-Line data    |
| `getPdfFiles(ticker, session_data)` | Download and parse company PDFs                           |
| `summarize_ticker_detail()`         | Summarize financial reports using LLMs                    |
| `embedd_text()`                     | Store summaries in ChromaDB                               |
| `embedd_prompt()`                   | Retrieve context embeddings and answer user queries       |
| `ai_assistance()`                   | LLM reasoning based on embedded context                   |
| `root()`                            | Streamlit entry point that loads both data columns and UI |


## 🧠 Example Workflow

1. Select a symbol (e.g., OGDC) in the sidebar
2. App fetches fundamentals, K-Line, and dividends
3. Downloads the company’s latest financial PDFs
4. Extracts tables and text, summarizes, and embeds them
5. Ask natural language questions like:
 “What is OGDC’s profit growth over the last year?”
 “Summarize the company’s recent dividend trends.”
6. The app answers using embedded summaries and AI reasoning


## 🖼️ UI Preview

<img width="1440" height="900" alt="Screenshot 2025-10-20 at 10 09 28 AM" src="https://github.com/user-attachments/assets/2213d0db-9033-4a3b-bb77-3329ae0302df" />
<img width="1440" height="900" alt="Screenshot 2025-10-20 at 10 09 38 AM" src="https://github.com/user-attachments/assets/e04e075c-7635-4279-b50d-1a38f86af7e2" />



https://github.com/user-attachments/assets/9279a6db-db31-4f86-bb93-d08adc48dfa6





