import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("chronos_forecasting")
from src.chronos import ChronosPipeline
import time

class ChronosRunner:
    def __init__(self, model="amazon/chronos-t5-small", device_map="cuda"):
        self.pipeline = ChronosPipeline.from_pretrained(
            model,
            device_map = device_map,
            torch_dtype = torch.bfloat16,
        )

    def evaluate(self, df, input_col, label_col=None, prediction_length=30, num_samples=20, 
                 temperature=1.0, top_k=50, top_p=1.0, make_plot=True):
        if label_col == None:
            label_col = input_col
        context = torch.tensor(df[input_col].iloc[:-prediction_length])
        forcast = self.pipeline.predict(
            context,
            prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        low, median, high = np.quantile(forcast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        # loss is MSE
        lloss = np.sum(np.square(np.array(df[label_col].iloc[-prediction_length:]) - np.array(low))) / prediction_length
        mloss = np.sum(np.square(np.array(df[label_col].iloc[-prediction_length:]) - np.array(median))) / prediction_length
        hloss = np.sum(np.square(np.array(df[label_col].iloc[-prediction_length:]) - np.array(high))) / prediction_length

        if make_plot:
            curr_time = str(
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(round(time.time())))
            )
            forecast_index = range(len(df.iloc[:-prediction_length]), len(df.iloc[:-prediction_length]) + prediction_length)
            plt.figure(figsize=(8, 4))
            plt.plot(df[label_col], color="royalblue", label="historical data")
            plt.plot(forecast_index, median, color="tomato", label="median forecast")
            plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
            plt.legend()
            plt.grid()
            plt.savefig(f"{label_col}_chronos_plot_{curr_time}.png")

        return dict(
            lloss=lloss, 
            mloss=mloss, 
            hloss=hloss,
            labels=np.array(df[label_col].iloc[-prediction_length:]),
            low=np.array(low),
            median=np.array(median),
            high=np.array(high))

        
