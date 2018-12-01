import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


Patch_data = pd.read_excel(
    '/Users/felipeantoniomendezsalcido/Desktop/Patch Analysis.xlsx', sheet_name=None)

Patch_data.keys()
Patch_data['sp_Peaks']


def ecdf(raw_data):
    '''[np.array -> tuple]
    Equivalent to R's ecdf(). Credit to Kripanshu Bhargava from Codementor'''
    cdfx = np.sort(data.unique())
    x_values = np.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))
    size_data = raw_data.size
    y_values = []
        for i in x_value:
            # all the values in raw data less than the ith value in x_values
        temp = raw_data[raw_data <= i]
    # fraction of that value with respect to the size of the x_values
        value = temp.size / size_data
    # pushing the value in the y_values
        y_values.append(value)
    # return both x and y values
    return (x_values, y_values)
