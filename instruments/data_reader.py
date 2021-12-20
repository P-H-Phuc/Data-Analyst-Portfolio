"""This module read history stock price for stocks
   Note: please import this module, not itself function"""

from urllib.request import Request, urlopen
from lxml import html 
import datetime
import pandas as pd

#Scrape data from website
def crawl(url: str=None, css: str=None) -> html.HtmlElement:
    """Analysis of the HTML of the page uses the lxml.html module
    Return the node to extract

    Args:
        url (str): the address of a given unique resource on the Web.
        css (str): the CSS expression.
    Returns:
        list: a list of the results.
    """    
    #Not Forbidden
    ret = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    doc = html.fromstring(urlopen(ret).read().decode('utf-8'))
    return doc.cssselect(css)

#Check symbols
def check_symbols(symbols: str=None) -> bool:
    """Check the symbols if exists on Vietnam's stock market.
    To avoid being penalized, sometimes, this function can not working .

    Args:
        symbols (str | list): Single stock symbol (ticker), list object of symbols, ex: ['ABC', 'XYZ']

    Returns:
        list bool: True if the symbols exist
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    results = []
    for symbol in symbols:
        if isinstance(symbols, str):
            symbol = symbols 
        #URL
        url = f"https://www.cophieu68.vn/profilesymbol.php?id={symbol}"
        response_content = html.fromstring(urlopen(url).read().decode('utf-8'))
        #Select element
        elements = response_content.cssselect('table:nth-child(1) tr:nth-child(1) td+ td')
        #Check if exists
        if not elements:
            results.append(False)
        else:
            results.append(True)
    return results

#Get closing price
def get_close_price(symbol: str=None, page: int=None) -> pd.DataFrame:
    """Return the price of symbol by page of website, contain 100 values for a page

    Args:
        symbol (str): A symbol of stock in Vietnam's stock market
        page (int): a number of page website
    
    Returns:
        pandas.DataFrame: A DataFrame of symbol's closing price
    """    

    url = f'https://www.cophieu68.vn/historyprice.php?currentPage={page}&id={symbol}'
    #Trading date
    date_trade = crawl(url, css='.td_bg1:nth-child(2)')
    #Closing price
    close_price = crawl(url, css='.td_bg2 strong')
    #Append to array
    results = [[], []]
    for date in date_trade:
        d = date.text_content() #Select contents
        d = datetime.datetime.strptime(d, "%d-%m-%Y").strftime("%Y-%m-%d") #Format datetime
        results[0].append(d)
    for price in close_price:
        results[1].append(price.text_content()) 
    return pd.DataFrame(data={"date_trade": results[0], 
                              f"{symbol.upper()}": results[1]})

#Get data
def get_data(symbols: str, start: str = '2000-01-01', 
             end: str = datetime.datetime.now().strftime('%Y-%m-%d')) -> pd.DataFrame:
    """Returns DataFrame of with historical over date range, start to end.\n
    To avoid being penalized, sometimes, this function can not working.

    Args:
        symbols (str | list): Single stock symbol (ticker), list object of symbols.\n
        start (str of datetime): Starting date, format: %Y-%m-%d, default is 2000-01-01.\n
        end (str of datetime): Ending date, format: %Y-%m-%d, default is today.

    Return
        pandas.Dataframe: A DataFrame with requested options data.
    """

    #Check symbols
    check = check_symbols(symbols)
    not_exist = []
    for i, j in enumerate(check):
        if j is False:
            not_exist.append(symbols[i])
    #Return
    if not_exist != []:
        return f'Symbols: {" ".join(not_exist)} not found'

    #Date range
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    range_date = (end - start).days
    num_pages = int(range_date/100) + 1
    #Convert a symbol to list
    if isinstance(symbols, str):
        symbols = [symbols]
    #Unique symbols
    symbols = list(dict.fromkeys(symbols))
    #Get price
    df_a_symbol = get_close_price(symbols[0], 1)
    df_all = df_a_symbol
    for symbol in symbols:
        symbol = symbol.upper()
        df_a_symbol = [get_close_price(symbol, page) for page in range(1, num_pages)]
        df_a_symbol = pd.concat(df_a_symbol, axis=0, ignore_index=True)
        df_all = pd.concat([df_all, df_a_symbol], axis=1)
    df = df_all.dropna(axis=1,how='any')
    df = df.loc[:, ~ df.columns.duplicated()]
    # Select data with date range
    df['date_trade'] = pd.to_datetime(df['date_trade'], format='%Y-%m-%d')
    df = df[df['date_trade'] >= start]
    df = df[df['date_trade'] <= end]
    # Set index
    df = df.set_index(pd.DatetimeIndex(df['date_trade'].values))
    # Drop and sort index asc
    df = df.drop('date_trade', axis=1).sort_index()
    # As freq Day
    df = df.asfreq('D').dropna()
    return df