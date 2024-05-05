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

    # TODO: make a version of evaluate that uses a dataloader
    def evaluate(self, data_loader):
        raise NotImplementedError

    def evaluate_step(self, input_vals, label_vals=None, prediction_length=30, num_samples=20, 
                 temperature=1.0, top_k=50, top_p=1.0, make_plot=True):
        
        if label_vals == None:
            label_vals = input_vals
        
        context = input_vals[:-prediction_length]
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
        lloss = np.sum(np.square(np.array(label_vals[-prediction_length:]) - np.array(low))) / prediction_length
        mloss = np.sum(np.square(np.array(label_vals[-prediction_length:]) - np.array(median))) / prediction_length
        hloss = np.sum(np.square(np.array(label_vals[-prediction_length:]) - np.array(high))) / prediction_length

        if make_plot:
            curr_time = str(
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(round(time.time())))
            )
            forecast_index = range(len(input_vals[:-prediction_length]), len(input_vals[:-prediction_length]) + prediction_length)
            plt.figure(figsize=(8, 4))
            plt.plot(label_vals, color="royalblue", label="historical data")
            plt.plot(forecast_index, median, color="tomato", label="median forecast")
            plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
            plt.legend()
            plt.grid()
            plt.savefig(f"chronos_plot_{curr_time}.png")

        return dict(
            lloss=lloss, 
            mloss=mloss, 
            hloss=hloss,
            labels=np.array(label_vals[-prediction_length:]),
            low=np.array(low),
            median=np.array(median),
            high=np.array(high))

        
