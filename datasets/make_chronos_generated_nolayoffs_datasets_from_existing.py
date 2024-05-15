from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import os
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("..")
from chronos_forecasting.src.chronos import ChronosPipeline

from random import randrange
from datetime import timedelta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

layoffs = pd.read_csv(r"tech_layoffs_CLEANED_NoNA.csv", parse_dates=["Date_layoffs"])


yf.pdr_override()
target_length = 90

from glob import glob
folder_nolayoff = "stocks_no_layoffs"
EXT = "*.csv"  # Define the variable EXT
for path, subdir, files in os.walk(folder_nolayoff):
    
    for file in glob(os.path.join(path, EXT)):
        # print(file)
        basesymbol = file.split("\\")[-1].split(".")[0]
        # Remove numbers from symbol

        symbol = ''.join([i for i in basesymbol if not i.isdigit()])
        code = ''.join([i for i in basesymbol if i.isdigit()])
        print(symbol)
        df_base = pd.read_csv(file, index_col=0).sort_index()
        # print(df)
        # Get start date of chronos prediction (1 year before first date in data)
        
        start_date = pd.to_datetime(df_base["index"][0]) - pd.Timedelta(days=366)
        
        end_date = pd.to_datetime(df_base["index"][0]) - pd.Timedelta(days=1)


        df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
        if df.empty:
            break

        idx = pd.date_range(start_date, end_date, freq='D', inclusive='both') # fill in missing dates
        df = df.reindex(idx)
        df["Open"] = df['Open'].interpolate(limit_direction='both')

        context = torch.tensor(df['Open'])
        prediction_length = 90
        forecast = pipeline.predict(
            context,
            prediction_length,
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            limit_prediction_length=False
        )
        idx2 = pd.date_range(end_date+pd.Timedelta(days=1), end_date+pd.Timedelta(days=90), freq='D', inclusive='both')
        generated = pd.DataFrame(index=idx2)
        generated['low'], generated['median'], generated['high'] = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        generated['low_percent_change'] = ((generated["low"].shift(-1) - generated['low']) / generated['low'].shift(-1))
        generated['low_inproportion_to_average'] = ((generated['low']- generated["low"].mean()) / generated['low'].mean())
        generated['low_normalized'] = ((generated['low']- generated["low"].mean()) / generated['low'].std())

        generated['median_percent_change'] = ((generated["median"].shift(-1) - generated['median']) / generated['median'].shift(-1))
        generated['median_inproportion_to_average'] = ((generated['median']- generated["median"].mean()) / generated['median'].mean())
        generated['median_normalized'] = ((generated['median']- generated["median"].mean()) / generated['median'].std())

        generated['high_percent_change'] = ((generated["high"].shift(-1) - generated['high']) / generated['high'].shift(-1))
        generated['high_inproportion_to_average'] = ((generated['high']- generated["high"].mean()) / generated['high'].mean())
        generated['high_normalized'] = ((generated['high']- generated["high"].mean()) / generated['high'].std())

        generated['Open'] = generated['median']
        rolling_min = list(generated['Open'].rolling(window=90).min())[-1]
        rolling_max = list(generated['Open'].rolling(window=90).max())[-1]
        generated['Scaled_Price_MinMax'] = (generated["Open"] - rolling_min) / (rolling_max - rolling_min)
        directory = "chronos_stocks_nolayoffs\\chronos\\"
        directory_real = "chronos_stocks_nolayoffs\\real\\"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(directory_real):
            os.makedirs(directory_real)
        if len(generated) >= target_length:
            generated.reset_index().iloc[:target_length].to_csv(directory + f"\\{symbol}{code}.csv")
            df_base.reset_index().iloc[:target_length].to_csv(directory_real + f"\\{symbol}{code}.csv")
        

        # X_train.append(df[src_col].values)
        # Y_train.append(float(1))




# for symbol in layoffs['Symbol'].unique():
#     print(symbol)
#     temp = layoffs.loc[layoffs['Symbol']==symbol]
#     layoff_dates = [earliest_layoff]
    
#     # pull stocks with layoffs
#     for i, row in temp.reset_index().iterrows():
#         industry = row["Industry"]
#         layoff_dates.append(row['Date_layoffs'])

#         directory = "chronos_stocks_layoffs\\" + industry

#         if not os.path.exists(directory):
#             os.makedirs(directory)

#         start_date = row['Date_layoffs'] - pd.Timedelta(days=365)
#         end_date = row['Date_layoffs'] - pd.Timedelta(days=45)

#         df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
#         if df.empty:
#             break

#         idx = pd.date_range(start_date, end_date, freq='D', inclusive='both') # fill in missing dates
#         df = df.reindex(idx)
#         df["Open"] = df['Open'].interpolate(limit_direction='both')

#         context = torch.tensor(df['Open'])
#         prediction_length = 91
#         forecast = pipeline.predict(
#             context,
#             prediction_length,
#             num_samples=20,
#             temperature=1.0,
#             top_k=50,
#             top_p=1.0,
#             limit_prediction_length=False
#         )
#         idx2 = pd.date_range(row['Date_layoffs'] - pd.Timedelta(days=45), row['Date_layoffs'] + pd.Timedelta(days=45), freq='D', inclusive='both')
#         generated = pd.DataFrame(index=idx2)
#         generated['low'], generated['median'], generated['high'] = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

#         generated['low_percent_change'] = ((generated["low"].shift(-1) - generated['low']) / generated['low'].shift(-1))
#         generated['low_inproportion_to_average'] = ((generated['low']- generated["low"].mean()) / generated['low'].mean())
#         generated['low_normalized'] = ((generated['low']- generated["low"].mean()) / generated['low'].std())

#         generated['median_percent_change'] = ((generated["median"].shift(-1) - generated['median']) / generated['median'].shift(-1))
#         generated['median_inproportion_to_average'] = ((generated['median']- generated["median"].mean()) / generated['median'].mean())
#         generated['median_normalized'] = ((generated['median']- generated["median"].mean()) / generated['median'].std())

#         generated['high_percent_change'] = ((generated["high"].shift(-1) - generated['high']) / generated['high'].shift(-1))
#         generated['high_inproportion_to_average'] = ((generated['high']- generated["high"].mean()) / generated['high'].mean())
#         generated['high_normalized'] = ((generated['high']- generated["high"].mean()) / generated['high'].std())

#         generated['Open'] = generated['median']

#         if len(generated) >= target_length:
#             generated.reset_index().iloc[:target_length].to_csv(directory + f"\\{symbol}{i}.csv")
    
#     layoff_dates.append(latest_layoff)
#     # pull stocks during 180 day periods without layoffs
#     max_non_layoff_periods = len(temp) # want same number periods without layoffs as periods with
#     num_periods = 0
    # directory = "chronos_stocks_no_layoffs\\" + industry

    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # other_count = 0
    # while num_periods < max_non_layoff_periods:
    #     if other_count > 1000:
    #         print('took too long')
    #         break
    #     other_count += 1
    #     rand_start_date = random_date(earliest_layoff, latest_layoff)
    #     rand_end_date = rand_start_date + pd.Timedelta(days=90)

    #     if any([rand_start_date < x < rand_end_date for x in layoff_dates[1:-1]]):
    #         continue
    #     else:
    #         start_date = rand_start_date - pd.Timedelta(days=365)
    #         end_date = rand_start_date - pd.Timedelta(days=45)
    #         df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
    #         if df.empty:
    #             break

    #         idx = pd.date_range(start_date, end_date, freq='D', inclusive='both') # fill in missing dates
    #         df = df.reindex(idx)
    #         df["Open"] = df['Open'].interpolate(limit_direction='both')

    #         context = torch.tensor(df['Open'])
    #         prediction_length = 91
    #         forecast = pipeline.predict(
    #             context,
    #             prediction_length,
    #             num_samples=20,
    #             temperature=1.0,
    #             top_k=50,
    #             top_p=1.0,
    #             limit_prediction_length=False
    #         )
    #         idx2 = pd.date_range(rand_start_date, rand_start_date + pd.Timedelta(days=90), freq='D', inclusive='both')
    #         generated = pd.DataFrame(index=idx2)
    #         generated['low'], generated['median'], generated['high'] = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    #         generated['low_percent_change'] = ((generated["low"].shift(-1) - generated['low']) / generated['low'].shift(-1))
    #         generated['low_inproportion_to_average'] = ((generated['low']- generated["low"].mean()) / generated['low'].mean())
    #         generated['low_normalized'] = ((generated['low']- generated["low"].mean()) / generated['low'].std())

    #         generated['median_percent_change'] = ((generated["median"].shift(-1) - generated['median']) / generated['median'].shift(-1))
    #         generated['median_inproportion_to_average'] = ((generated['median']- generated["median"].mean()) / generated['median'].mean())
    #         generated['median_normalized'] = ((generated['median']- generated["median"].mean()) / generated['median'].std())

    #         generated['high_percent_change'] = ((generated["high"].shift(-1) - generated['high']) / generated['high'].shift(-1))
    #         generated['high_inproportion_to_average'] = ((generated['high']- generated["high"].mean()) / generated['high'].mean())
    #         generated['high_normalized'] = ((generated['high']- generated["high"].mean()) / generated['high'].std())

    #         if len(generated) >= target_length:
    #             generated.reset_index().iloc[:target_length].to_csv(directory + f"\\{symbol}{num_periods}.csv")
    #             num_periods += 1