#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student Name: Dmytro 
Student ID: R00274472
Cohort/Group/Course: SDH3A 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
def load_data():
    # get current script directory
    folder = os.path.dirname(__file__)
    # create full file path
    data_path = os.path.join(folder,"..", "data", "shopping.csv")
    # convert to absolute path
    data_path = os.path.abspath(data_path)

    # read csv file
    df = pd.read_csv(data_path)

    # clean column names, remove extra spaces
    df.columns = df.columns.str.strip()

    # # show column names
    # print("\n--- Columns in shopping.csv ---")
    # print(df.columns)

    return df

# Task 1: Analyze relationship between products and shipping types 
def product_shipping(df):
    # remove empty rows
    df = df.dropna(how='all')

    # remove spaces from text columns
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip()

    # select important columns
    product_col = "Item Purchased"
    ship_col = "Shipping Type"

    # remove rows with missing values
    df = df.dropna(subset=[product_col, ship_col])

    # count shipping for each product
    counts = df.groupby([product_col, ship_col]).size().reset_index(name='Count')

    # calculate persentages
    counts['Percent'] = (counts['Count'] / counts.groupby(product_col)['Count'].transform('sum') * 100).round(2)

    # sort and print
    counts = counts.sort_values([product_col, 'Percent'], ascending=[True, False])

    # print each product results
    for product in counts[product_col].unique():
        print(f"\nProduct: {product}")
        sub = counts[counts[product_col] == product]
        for _, row in sub.iterrows():
            print(f" {row[ship_col]} - {row['Percent']}% (count: {row['Count']})")


# main function
def main():
   product_shipping(load_data())


main()
