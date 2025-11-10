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

    # calculate persentages (divide each shipping count by the total number of orders of that product row by row)
    counts['Percent'] = (counts['Count'] / counts.groupby(product_col)['Count'].transform('sum') * 100).round()

    # sort and print
    counts = counts.sort_values([product_col, 'Percent'], ascending=[True, False])

    # print each product results
    for product in counts[product_col].unique():
        print(f"\nProduct: {product}")
        sub = counts[counts[product_col] == product]
        for _, row in sub.iterrows():
            print(f" {row[ship_col]} - {row['Percent']}% (amount: {row['Count']})")

# -----------------------------------------------------------------------------------
# Task2: customer segments separated by gender
def customer_segments(df):
    # removing extra space
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip()

    # convert columns to numeriic values
    df["Purchase Amount (USD)"] = pd.to_numeric(df["Purchase Amount (USD)"], errors='coerce')
    df["Previous Purchases"] = pd.to_numeric(df["Previous Purchases"], errors='coerce')

    # drop rows with missing or wrong values
    df = df.dropna(subset=["Purchase Amount (USD)", "Previous Purchases", "Gender"])

    # create new column
    df["Total Purchased USD"] = df["Purchase Amount (USD)"] * df["Previous Purchases"]

    # definr segments
    bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    labels = ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "Segment 5", "Segment 6", 
              "Segment 7", "Segment 8", "Segment 9", "Segment 10", "Segment 11", "Segment 12" ]
    
    df["Segment"] = pd.cut(df["Total Purchased USD"], bins=bins, labels=labels, include_lowest=True)

    # group by gender and segment
    group_counts = df.groupby(["Segment", "Gender"]).size()

    print("----- Population of each gender ------")
    print(group_counts)

    # visualization
    vis = group_counts.plot(kind='bar', figsize=(10,8), color=['green', 'blue'])
    vis.set_title("Customer Segments by gender")
    vis.set_xlabel("Segments")
    vis.set_ylabel("Number of customers")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# main function
def main():
    # Task 1
    # product_shipping(load_data())
    # Task 2
    customer_segments(load_data())


if __name__ == "__main__":
    main()
