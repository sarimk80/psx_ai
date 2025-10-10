from pydantic import BaseModel
from typing import List,Any

## Symbols
class Symbols(BaseModel):
    success:bool
    data: List[str]
    timestamp:int

## Companies 

class FinancialStatsData(BaseModel):
    raw: str
    numeric: float

class FinancialStats(BaseModel):
    marketCap: FinancialStatsData
    shares: FinancialStatsData
    freeFloat: FinancialStatsData
    freeFloatPercent: FinancialStatsData

class KeyPerson(BaseModel):
    name: str
    position: str

class CompanyData(BaseModel):
    symbol: str
    scrapedAt: str
    financialStats: FinancialStats
    businessDescription: str
    keyPeople: List[KeyPerson]

class Companies(BaseModel):
    success:bool
    data:CompanyData
    timestamp:int


#Fundamentals

class FundamentalData(BaseModel):
    symbol: str
    sector: str
    listedIn: str
    marketCap: str
    price: float
    changePercent: float
    yearChange: float
    peRatio: float
    dividendYield: float
    freeFloat: str
    volume30Avg: float
    isNonCompliant: bool
    timestamp: str

class Fundamentals(BaseModel):
    success: bool
    data: FundamentalData
    timestamp: int


## Dividend

class DividendData(BaseModel):
    symbol: str
    ex_date: str
    payment_date: str
    record_date: str
    amount: float
    year: int

class Dividend(BaseModel):
    success: bool
    data: List[DividendData]
    count: int
    symbol: str
    timestamp: int
    cacheUpdated: str


## Kline

class KLineData(BaseModel):
    symbol: str
    timeframe: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int

class KLine(BaseModel):
    success: bool
    data: List[KLineData]
    count: int
    symbol: str
    timeframe: str
    startTimestamp: Any
    endTimestamp: Any
    timestamp: int

## tickers detail

class TickerDetailModel(BaseModel):
    market: str
    st: str
    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int
    trades: int
    value: float
    high: float
    low: float
    bid: int
    ask: int
    bidVol: int
    askVol: int
    timestamp: int

class TickerDetail(BaseModel):
    success: bool
    data: TickerDetailModel
    timestamp: int