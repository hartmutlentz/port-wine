"""
https://pythoninvest.com/long-read/exploring-finance-apis#popup:subscribe

"""
import yfinance as yf
from pprint import pprint

pfe = yf.Ticker('PFE')
pprint(pfe.info)

