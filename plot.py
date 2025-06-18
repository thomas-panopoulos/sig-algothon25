#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the stock price data from the text file
file_path = "./prices.txt"
data = pd.read_csv(file_path, delim_whitespace=True, header=None)

# Plotting each column (representing a different stock)
plt.figure(figsize=(12, 6))
# for column in data.columns:


    plt.plot(data.index, data[column], label=f'Stock {column+1}')

plt.title("Stock Prices Over Time")
plt.xlabel("Day")
plt.ylabel("Price")
plt.legend(loc='upper right', fontsize='small', ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()

