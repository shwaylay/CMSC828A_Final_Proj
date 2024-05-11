from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import os

from random import randrange
from datetime import timedelta

def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)



layoffs = pd.read_csv(r"tech_layoffs_CLEANED_NoNA.csv", parse_dates=["Date_layoffs"])


yf.pdr_override()
target_length = 120
earliest_layoff = layoffs["Date_layoffs"].min()
latest_layoff = layoffs["Date_layoffs"].max()

for symbol in layoffs['Symbol'].unique():
    temp = layoffs.loc[layoffs['Symbol']==symbol]
    layoff_dates = [earliest_layoff]
    
    # pull stocks with layoffs
    for i, row in temp.reset_index().iterrows():
        industry = row["Industry"]
        layoff_dates.append(row['Date_layoffs'])

        directory = "stocks_layoffs\\" + industry

        if not os.path.exists(directory):
            os.makedirs(directory)

        start_date = row['Date_layoffs'] - pd.Timedelta(days=90)
        end_date = row['Date_layoffs'] + pd.Timedelta(days=90)

        df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
        df['open_percent_change'] = ((df["Open"].shift(-1) - df['Open']) / df['Open'].shift(-1))
        df['open_inproportion_to_average'] = ((df['Open']- df["Open"].mean()) / df['Open'].mean())
        df['open_normalized'] = ((df['Open']- df["Open"].mean()) / df['Open'].std())

        if len(df) >= target_length:
            df.reset_index().iloc[:target_length].to_csv(directory + f"\\{symbol}{i}.csv")
    
    layoff_dates.append(latest_layoff)
    # pull stocks during 180 day periods without layoffs
    max_non_layoff_periods = len(temp) # want same number periods without layoffs as periods with
    num_periods = 0
    directory = "stocks_no_layoffs\\" + industry

    if not os.path.exists(directory):
        os.makedirs(directory)

    other_count = 0
    while num_periods < max_non_layoff_periods:
        if other_count > 10000000:
            print('took too long')
            break
        other_count += 1
        rand_start_date = random_date(earliest_layoff, latest_layoff)
        rand_end_date = rand_start_date + pd.Timedelta(days=180)

        if any([rand_start_date < x < rand_end_date for x in layoff_dates[1:-1]]):
            continue
        else:
            df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
            df['open_percent_change'] = ((df["Open"].shift(-1) - df['Open']) / df['Open'].shift(-1))
            df['open_inproportion_to_average'] = ((df['Open']- df["Open"].mean()) / df['Open'].mean())
            df['open_normalized'] = ((df['Open']- df["Open"].mean()) / df['Open'].std())
        
            if len(df) >= target_length:
                df.reset_index().iloc[:target_length].to_csv(directory + f"\\{symbol}{num_periods}.csv")
                num_periods += 1