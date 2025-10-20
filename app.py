from sys import exception
from numpy import partition
import ollama
from pandas.core import base
import requests
import streamlit as st
from models import Symbols,Companies,Fundamentals,Dividend,KLine,TickerDetail
import pandas as pd
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from pypdf import PdfReader
import asyncio
import threading
import chromadb.api
import pdfplumber
from multiprocessing import Pool,freeze_support
import math

baseUrl = "https://psxterminal.com"
pdfBaseUrl = "https://dps.psx.com.pk"
pdf_links : list[str] = []
tickers_symbol =  ["KSE100","ALLSHR","PSXDIV20","KSE30"]
tickerDetail = []

CHROMA_COLLECTION_NAME = "my_db"


def makeMarketRequest():
    for symbol in tickers_symbol:
        ticker_request = requests.get(f"{baseUrl}/api/ticks/{'IDX'}/{symbol}")
        ticker_pydantic = TickerDetail.model_validate_json(ticker_request.text)
        tickerDetail.append(ticker_pydantic)

def makeApiRequest(ticker):
    r_1 = requests.get(f"{baseUrl}/api/companies/{ticker}")
    r_2 = requests.get(f"{baseUrl}/api/fundamentals/{ticker}")
    r_3 = requests.get(f"{baseUrl}/api/dividends/{ticker}")
    r_4 = requests.get(f"{baseUrl}/api/klines/{ticker}/{"1d"}")

    st.session_state.companies = Companies.model_validate_json(r_1.text)
    st.session_state.fundamentals = Fundamentals.model_validate_json(r_2.text)
    st.session_state.dividend = Dividend.model_validate_json(r_3.text)
    st.session_state.kline = KLine.model_validate_json(r_4.text)



def summary_ai_assistance(data):
    system_prompt = (

        """
                    You are a financial data parser and summarizer AI.

            Your task is to:
            1. Read structured financial API or JSON-like responses.
            2. Extract and organize relevant data fields clearly.
            3. Produce a clean, human-readable summary including:
            - **Company Overview**: name, incorporation details, parent company, main business activities, and key executives.
            - **Financial & Market Data**: current price, market cap, volume, P/E ratio, dividend yield, free float, year change, compliance status, and listing information.
            - **Recent Stock Performance**: summarize 1D KLine or OHLCV data (open, high, low, close, volume), highlighting trends, volatility, and notable movements.
            - **Timestamps**: latest data timestamp and range of trading data.
            - **Summary Insight**: a concise interpretation of performance, trends, and financial health.

            Formatting requirements:
            - Use clear section headers (Company Overview, Financial Data, etc.)
            - Use bullet points and percentages/numbers with symbols (₨, %, M, B).
            - If data is missing or zero, note it as “Not available” or “Likely not reported”.
            - Keep the summary concise but comprehensive (around 250–350 words).
            - Always mention the date of the latest update.

            Output example:

            ---
            ### 🏢 Company Overview
            - Name: ...
            - Business: ...
            - Key People: ...

            ### 📊 Financial Data
            - Market Cap: ...
            - Price: ...
            - Yearly Change: ...
            - etc.

            ### 📈 Recent Stock Performance
            - Recent Close: ...
            - Range: ...
            - Trend Summary: ...

            ### 🕒 Data Timestamps
            - Last Updated: ...
            - Data Range: ...

            ### 💡 Summary Insight
            (1–2 paragraph interpretation)
            ---

            You should be able to process arrays like:
            ["success=True data=FundamentalData(...)", "success=True data=CompanyData(...)", "success=True data=[KLineData(...)]"]
            and extract all fields accurately.

            If multiple entries are found, use the most recent `timestamp` values.
            Always end with an interpretation of company and stock performance.



        """
    
    )
    full_prompt = f"""
    Context:
    {data}

    Question:
    {"concise summary of all the provided data"}
    """
    
    response = ollama.chat(
       # model="gpt-oss:20b",
        model = "qwen3-coder:480b-cloud",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": full_prompt.strip()}
        ],
        think='low'
    )
    print(response['message']['content'])
    return response['message']['content']

def ai_assistance(embedResult,prompt):

    system_prompt = (
       """ 
       You are a financial analyst AI specializing in interpreting and answering questions using only the provided context.
Follow these rules carefully:
1. Use only the information contained in the context — do not rely on outside knowledge or assumptions.
2. Provide answers that are clear, concise, and directly address the user’s question.
3. When appropriate, include brief quantitative or qualitative reasoning drawn from the context.
4. Maintain a professional, objective, and analytical tone.
5. Do not speculate, infer, or fabricate information beyond what is given.
       """
    )

    full_prompt = f"""
    Context:
    {embedResult}

    Question:
    {prompt}
    """
    
    response = ollama.chat(
       # model="gpt-oss:20b",
        model = "gpt-oss:20b-cloud",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": full_prompt.strip()}
        ],
        think='low'
    )
    print(response['message']['content'])
    return response['message']['content']

def embedd_prompt(prompt,collection):
    prompt_embedd = ollama.embed(
        model='embeddinggemma:latest',
        input= prompt
    )
    result = collection.query(
        query_embeddings=prompt_embedd['embeddings'],
        n_results = 4
    )
    # Safely access nested data
    if result is not None \
    and 'documents' in result \
    and isinstance(result['documents'], list) \
    and len(result['documents']) > 0 \
    and isinstance(result['documents'][0], list) \
    and len(result['documents'][0]) > 0:
        data = result['documents']
        return  ai_assistance(data,prompt)
        
    else:
        data = None  # or raise an error/handle missing data
        return ''

def embedd_text(text,collection):
    
    for i, d in enumerate(text):
        embedd_response = ollama.embed(
            model="embeddinggemma:latest",
            input=d
        )
        embedding = embedd_response["embeddings"]
        collection.add(
            ids=[f"{str(i)}"],
            embeddings=embedding,
            documents=[d]
            )
        print(f"{i} embedding done")

    
def chunks_text(text,chunk_size = 1800):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    
def summarize_table(table):
    system_prompt = (
        """
        You are an expert data analyst.\n
        Given the following table, generate a dense, information-rich summary that accurately represents all data points and relationships contained in the table.\n
        Write in natural language suitable for text embedding and retrieval — include all relevant facts, figures, and entities, but exclude formatting or table structure.\n
        Do not repeat the table. Output only the summary text.

        """
    )

    full_prompt = f"""
    Context:
    {table}

    Question:
    {"Summarize the table"}
    """
    response = ollama.chat(
       # model="gpt-oss:20b",
        model = "deepseek-v3.1:671b-cloud",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    print(response['message']['content'])
    return response['message']['content']

def summarize_ticker_detail(ticker_detail,model_name,ticker_symbol):
    system_prompt = (
        """
        You are a professional financial analyst specialized in transforming structured or unstructured financial reports into dense, semantically rich summaries suitable for embedding and retrieval systems.

Given a portion (chunk) of a financial report, produce a factually accurate and contextually complete natural-language summary. Your summary should integrate all quantitative and qualitative information present in the text, including but not limited to:

- Revenue, profit, and cash flow performance
- Key financial ratios and year-over-year changes
- Segment or regional performance highlights
- Management commentary, strategic outlook, and guidance
- Notable risks, uncertainties, or auditor comments

Write in continuous prose (no bullet points, headings, or formatting). The output should read as a cohesive paragraph that preserves both numeric and contextual details for downstream semantic search and synthesis.

Output only the summary text, nothing else.

        """
    )

    full_prompt = f"""
    Context:
    {ticker_symbol} {ticker_detail}

    Question:
    {"Summarize this financial report chunk as described in the system prompt."}
    """
    response = ollama.chat(
       # model="gpt-oss:20b",
        model = model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    print(response['message']['content'])
    return response['message']['content']

def worker (model_name,sub_chunk,ticker_symbol):
        results = []
        for text in sub_chunk:
            result = summarize_ticker_detail(text,model_name,ticker_symbol)
            print(result)
            results.append(result)
        return results

def summarize_chunk_text(chunks_of_text,model_name,tickers_symbol):
    num_parts = len(model_name)
    print(len(model_name))
    chunk_size = math.ceil(len(chunks_of_text)/num_parts)
    print(chunk_size)

    divide_chunks = [
        chunks_of_text[i:i + chunk_size] for i in range(0, len(chunks_of_text), chunk_size)
    ]
    print(len(divide_chunks))
    with Pool(processes=num_parts) as pool:
        async_result = [
            pool.apply_async(worker,(model_name[i],divide_chunks[i],tickers_symbol))
            for i in range(len(divide_chunks))
        ]
        results = [r.get() for r in async_result]

    all_summaries = [item for sublist in results for item in sublist]
    return all_summaries

def summarize_chunk_text_01(chunks_of_text):
    list_of_summarize = []
    for index , chunks in enumerate(chunks_of_text):
        result = summary_ai_assistance(chunks)
        list_of_summarize.append(result)
    return list_of_summarize


async def getPdfFiles(ticker,session_data,collection):
    r_5 = requests.get(f"{pdfBaseUrl}/company/{ticker}")
    text = ""
    all_text = ""
    all_tables = []

    soup = BeautifulSoup(r_5.text,"html.parser")
    for i,link in enumerate(soup.find_all('a')):
        if str(link.get('href')).endswith('.pdf'):
            pdfDownloadUrl = pdfBaseUrl + str(link.get('href'))
            pdf_links.append(pdfDownloadUrl)

    limited_pdf_links : list[str] = pdf_links[:2]

    for i , pdfLinks in enumerate(limited_pdf_links):
        innes_text = ''
        inner_table = []
        with requests.get(str(pdfLinks),stream=True) as r:
            fileName = f"{i}.pdf"
            with open(fileName,"wb") as f:
                for chunks in r.iter_content(chunk_size=8140):
                    if chunks:
                        f.write(chunks)

            with pdfplumber.open(fileName) as pdf:
                for page_number,page in enumerate(pdf.pages,start=1):
                    text = page.extract_text()
                    if text:
                        innes_text  += f"\n--- Page {page_number} ---\n{text}"
                        all_text += f"\n--- Page {page_number} ---\n{text}"
                    
                    tables = page.extract_tables()
                    if tables:
                        for table_index,table in enumerate(tables,start=1):
                            inner_table.append(
                                {
                                "page": page_number,
                                "table_index": table_index,
                                "data": table
                                }
                            )
                            all_tables.append(
                               {
                                "page": page_number,
                                "table_index": table_index,
                                "data": table
                               }
                            )
                



       

    structured_parts = []
    for key, data_obj in session_data.items():
        if data_obj:
            structured_parts.append(str(data_obj))
        else:
            print("no data")

    tables_as_text = ["\n".join(["\t".join(map(str, row)) for row in t["data"]]) for t in all_tables]
    combined_text = all_text  + "\n\n".join(tables_as_text)
    
    chunks_of_text = chunks_text(text=combined_text)
    print(f"chunks of text length {len(chunks_of_text)}")
    chunks_of_text_01 = chunks_text(text="\n\n".join(structured_parts))
    print(f"Chunks of length {len(chunks_of_text_01)}")
    result = summarize_chunk_text(chunks_of_text=chunks_of_text,model_name=
    ['qwen3-coder:480b-cloud','gpt-oss:20b-cloud','deepseek-v3.1:671b-cloud','kimi-k2:1t-cloud'],tickers_symbol=ticker)
    result_1 = summarize_chunk_text_01(chunks_of_text=chunks_of_text_01)
    print(f"summarize of text length {len(result)}")
    embedd_text(text=result + result_1,collection=collection)
    print("embedding")
    #embedd_prompt()
    

#st.session_state.fundamentals + st.session_state.companies + st.session_state.dividend + st.session_state.kline

def colums_1(collection):
    r = requests.get(f"{baseUrl}/api/symbols")
    symbols = Symbols.model_validate_json(r.text)

    options = st.selectbox("Select the symbol",
    (symbols.data),
    index=None,
    placeholder="Select the symbol for AI assistance")

    st.session_state.options = options

    if options:

        makeApiRequest(options)
        session_copy = {
            "fundamentals": st.session_state.get("fundamentals"),
            "companies": st.session_state.get("companies"),
            "dividend": st.session_state.get("dividend"),
            "kline": st.session_state.get("kline")
        }
        if "pdf_thread" not in st.session_state or not st.session_state.pdf_thread.is_alive():
            thread = threading.Thread(target=lambda:asyncio.run(getPdfFiles(options,session_copy,collection)))
            thread.start()
            st.session_state.pdf_thread = thread
            st.info("Downaloading pdf files")
        else:
            st.info("Files downloaded")

        input_outout_ui(collection)
        
        
@st.fragment
def input_outout_ui(collection):
    if prompt := st.chat_input("Enter your question?"):
        st.chat_message("user").markdown(prompt)
        

        response = embedd_prompt(prompt,collection)

        
        with st.chat_message("assistant"):
            st.markdown(response)

def colums_2():

    if tickerDetail:
        tickers = [ticker.data.model_dump() for ticker in tickerDetail]
        lows = [ticker.data.low for ticker in tickerDetail]
        row = st.container(horizontal=True)
        with row:
            col = st.columns(len(tickers))
            for i , ticker_metric in enumerate(tickers):
                with col[i]:
                    st.metric(
                        label= ticker_metric['symbol'],
                        value=f"{ticker_metric['price']:,.2f}",
                        delta=f"{ticker_metric['changePercent']:,.4f}%",
                        chart_data=lows,
                        chart_type='area',
                    )
    
    st.divider()

    if "fundamentals" in st.session_state and "companies" in st.session_state:
        st.metric(
            label = st.session_state.fundamentals.data.symbol,
            value = st.session_state.fundamentals.data.price,
            delta = f"{st.session_state.fundamentals.data.changePercent:,.4f}%"
            )
        st.text(st.session_state.companies.data.businessDescription)
        col1, col2 ,col3 = st.columns(3)
        col1.metric(
            label="Shares",
            value= f"{st.session_state.companies.data.financialStats.shares.numeric:,.2f}"
        )
        col2.metric(
            label="Market Capital",
            value= f"{st.session_state.companies.data.financialStats.marketCap.numeric:,.2f}"
        )
        col3.metric(
            label="Free float",
            value= f"{st.session_state.companies.data.financialStats.freeFloat.numeric:,.2f}",
            delta= f"{st.session_state.companies.data.financialStats.freeFloatPercent.numeric:,.2f}",
            delta_color='off'
        )
        key_persons = [companies.model_dump() for companies in st.session_state.companies.data.keyPeople]
        if key_persons:
                df = pd.DataFrame(key_persons)
                st.table(df)
            
    if "kline" in st.session_state:
        records = [kline.model_dump() for kline in st.session_state.kline.data]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')
        fig = go.Figure(data=[go.Candlestick(
            x= df['timestamp'],
            close=df['close'],
            open=df['open'],
            high=df['high'],
            low=df['low']
        )])
        fig.update_xaxes(
            rangeslider_visible=True,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # weekends
                dict(bounds=[4, 9.5], pattern="hour")
            ]
        )

        


        st.plotly_chart(fig)
        st.bar_chart(df,x='timestamp',y="volume")

        if "dividend" in st.session_state:
            dividends = [dividend.model_dump() for dividend in st.session_state.dividend.data]
            
            if dividends:
                df = pd.DataFrame(dividends)
                st.table(df)

def root(collection):
    st.set_page_config(page_title="PSX AI Assistance",layout="wide")
    makeMarketRequest()
    col1 , col2 = st.columns([0.4,0.6])
    with col1:
        colums_1(collection)
    with col2:
        colums_2()

@st.cache_resource
def get_chromddb():
    # Initialize persistent client (you can specify a path if needed)
    client = chromadb.PersistentClient(path="chroma_db")

    # Optional: check if the collection already exists
    existing_collections = [c.name for c in client.list_collections()]

    if CHROMA_COLLECTION_NAME in existing_collections:
        print(f"Deleting existing collection: {CHROMA_COLLECTION_NAME}")
        client.delete_collection(CHROMA_COLLECTION_NAME)

    # Create a fresh collection
    collection = client.create_collection(CHROMA_COLLECTION_NAME)

    print(f"✅ Collection '{CHROMA_COLLECTION_NAME}' created successfully.")
    return collection

if __name__ == '__main__':

    collection = get_chromddb()

    freeze_support()
    root(collection)