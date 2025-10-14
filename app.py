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

baseUrl = "https://psxterminal.com"
pdfBaseUrl = "https://dps.psx.com.pk"
pdf_links : list[str] = []
tickers_symbol =  ["KSE100","ALLSHR","PSXDIV20","KSE30"]
tickerDetail = []

client = chromadb.PersistentClient()
client.clear_system_cache()
client.delete_collection(name='my_db')
collection = client.get_or_create_collection("my_db")


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
        model = "qwen3-coder:latest",
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

def embedd_prompt(prompt):
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

def embedd_text(text):
    
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

def summarize_ticker_detail(ticker_detail):
    system_prompt = (
        """
        You are a professional financial analyst trained to convert structured or textual financial data into dense, semantically rich summaries for retrieval systems.
        Given the following financial report, write a factually complete, context-rich summary in natural language that captures all quantitative and qualitative insights — including financial performance, ratios, year-over-year changes, management outlook, and risk factors.
        The output must be embedding-friendly: avoid bullet points, headings, or formatting.
        Output only the summary text.

        """
    )

    full_prompt = f"""
    Context:
    {ticker_detail}

    Question:
    {"Summarize the financial report"}
    """
    response = ollama.chat(
       # model="gpt-oss:20b",
        model = "kimi-k2:1t-cloud",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    print(response['message']['content'])
    return response['message']['content']


def summarize_chunk_text(chunks_of_text):
    list_of_summarize = []
    for index , chunks in enumerate(chunks_of_text):
        result = summarize_ticker_detail(chunks)
        list_of_summarize.append(result)
    return list_of_summarize

def summarize_chunk_text_01(chunks_of_text):
    list_of_summarize = []
    for index , chunks in enumerate(chunks_of_text):
        result = summary_ai_assistance(chunks)
        list_of_summarize.append(result)
    return list_of_summarize

#https://dps.psx.com.pk/download/document/258577.pdf
#https://dps.psx.com.pk/download/document/257240.pdf

async def getPdfFiles(ticker,session_data):
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
    result = summarize_chunk_text(chunks_of_text=chunks_of_text)
    result_1 = summarize_chunk_text_01(chunks_of_text=chunks_of_text_01)
    print(f"summarize of text length {len(result)}")
    embedd_text(text=result + result_1)
    print("embedding")
    #embedd_prompt()
    

#st.session_state.fundamentals + st.session_state.companies + st.session_state.dividend + st.session_state.kline

def colums_1():
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
            thread = threading.Thread(target=lambda:asyncio.run(getPdfFiles(options,session_copy)))
            thread.start()
            st.session_state.pdf_thread = thread
            st.info("Downaloading pdf files")
        else:
            st.info("Files downloaded")

        input_outout_ui()
        
        
@st.fragment
def input_outout_ui():
    if prompt := st.chat_input("Enter your question?"):
        st.chat_message("user").markdown(prompt)
        

        response = embedd_prompt(prompt)

        
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

def root():
    st.set_page_config(page_title="PSX AI Assistance",layout="wide")
    makeMarketRequest()
    col1 , col2 = st.columns([0.4,0.6])
    with col1:
        colums_1()
    with col2:
        colums_2()

if __name__ == '__main__':

    root()