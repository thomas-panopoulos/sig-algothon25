#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the stock price data from the text file
file_path = "./prices.txt"
data = pd.read_csv(file_path, delim_whitespace=True, header=None)

def plotAll():
    # Plotting each column (representing a different stock)
    plt.figure(figsize=(12, 6))
    for column in data.columns:
        plt.plot(data.index, data[column], label=f'Stock {column+1}')

    plt.title("Stock Prices Over Time")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend(loc='upper right', fontsize='small', ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotSum(listt):
    plt.figure(figsize=(12, 6))
    for num in listt:
        plt.plot(data.index, data[num], label=f'Stock {num+1}')

    plt.title("Stock Prices Over Time")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend(loc='upper right', fontsize='small', ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plotSum([23,22])


kendall = data.corr(method="kendall")
pearson = data.corr(method="pearson")
spearman = data.corr(method="spearman")

maskKendall = kendall.mask(kendall <= 0.7);
maskPearson = pearson.mask(pearson <= 0.7);
maskSpearman = spearman.mask(pearson <= 0.7);

def plotHeat(corr_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

plotHeat(maskKendall,"Kendall")
plotHeat(maskPearson,"pearson")
plotHeat(maskSpearman,"spearman")
