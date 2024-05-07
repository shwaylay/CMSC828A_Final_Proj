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

    def evaluate(self, src_input, target_input, make_plot=False):
        batches = zip(src_input, target_input)
        llosses = []
        mlosses = []
        hlosses = []
        labels = []
        medians = []
        for src_sample, target_sample in batches:
            results = self.evaluate_step(src_sample[0], target_sample[0], make_plot=make_plot)
            llosses.append(results['lloss'])
            mlosses.append(results['mloss'])
            hlosses.append(results['hloss'])
            labels.append(results['labels'])
            medians.append(results['median'])

        return dict(
            lloss = np.array(llosses).mean(),
            mloss = np.array(mlosses).mean(),
            hloss = np.array(hlosses).mean(),
            labels = torch.tensor(np.array(labels), requires_grad=True),
            median = torch.tensor(np.array(medians), requires_grad=True)
        )

    def evaluate_step(self, input_vals, label_vals=None, prediction_length=30, num_samples=20, 
                 temperature=1.0, top_k=50, top_p=1.0, make_plot=True):
        
        if label_vals == None:
            label_vals = input_vals

        input_vals.to('cpu')
        label_vals.to('cpu')
        
        context = input_vals[:-prediction_length]
        forecast = self.pipeline.predict(
            context.to("cpu"),
            prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        low, median, high = np.quantile(forecast[0].detach().numpy(), [0.1, 0.5, 0.9], axis=0)
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
            plt.plot(input_vals, color="purple", label="source data")
            plt.plot(label_vals, color="royalblue", label="target data")
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

        
