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
        df[col] = df[col].astype(str).str.strip().str.capitalize()

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
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # convert columns to numeriic values
    df["Purchase Amount (USD)"] = pd.to_numeric(df["Purchase Amount (USD)"], errors='coerce')
    df["Previous Purchases"] = pd.to_numeric(df["Previous Purchases"], errors='coerce')

    # drop rows with missing or wrong values
    df = df.dropna(subset=["Purchase Amount (USD)", "Previous Purchases", "Gender"])

    # create new column
    df["Total Purchased USD"] = df["Purchase Amount (USD)"] * df["Previous Purchases"]

    # definr 12 segments (0–500, 500–1000, ...)
    valid = df["Total Purchased USD"] >= 0
    df = df[valid].copy()

    idx = ((df["Total Purchased USD"] - 1) // 500).astype("int64")
    idx = idx.clip(lower=0, upper=11)

    # make segment names
    label_map = {i: f"Segment{i+1}" for i in range(12)}
    df["Segment"] = idx.map(label_map)

    # group by gender and segment
    group_counts = df.groupby(["Segment", "Gender"]).size().unstack(fill_value=0)

    # dictionary to show numeric ranges
    segment_labels = {
        "Segment1": "0-500",
        "Segment2": "500-1000",
        "Segment3": "1000-1500",
        "Segment4": "1500-2000",
        "Segment5": "2000-2500",
        "Segment6": "2500-3000",
        "Segment7": "3000-3500",
        "Segment8": "3500-4000",
        "Segment9": "4000-4500",
        "Segment10": "4500-5000",
        "Segment11": "5000-5500",
        "Segment12": "5500-6000"
    }

    # sort index by numeric part
    group_counts = group_counts.reindex(sorted(group_counts.index, key=lambda x: int(x.replace("Segment", ""))))

    # print with both segment name and range
    print("\n--- Segment names with ranges ---")
    for seg, row in group_counts.iterrows():
        range_label = segment_labels.get(seg, "Unknown")
        female = row.get("Female", 0)
        male = row.get("Male", 0)
        print(f"{seg} ({range_label}): Female={female}, Male={male}")

    # update index labels
    group_counts.index = group_counts.index.map(segment_labels)
    
    # visualization
    vis = group_counts.plot(kind='bar', figsize=(8,6)) 
    vis.set_title("Customer Segments by gender")
    vis.set_xlabel("Total Purchased USD Range")
    vis.set_ylabel("Number of customers")
    vis.legend(title="Gender")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


# -----------------------------------------------------------------------------------
# Task 3: Product Analysis
def product_analysis(df):
    # removing extra space
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # convert numeric columns
    df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
    df["Previous Purchases"] = pd.to_numeric(df["Previous Purchases"], errors='coerce')

    # drop rows with missing important data
    df = df.dropna(subset=["Age", "Previous Purchases", "Item Purchased"])

    # calculate A and B for every product
    products = df["Item Purchased"].unique()
    results = []

    for product in products:
        sub = df[df["Item Purchased"] == product]

        # average across age groups
        age_avg = sub.groupby("Age")["Previous Purchases"].mean().round(2)
        a = age_avg.mean().round(2)

        # overall average 
        b = sub["Previous Purchases"].mean().round(2)

        results.append((product, a, b ))

    # create data frame of results
    results_df = pd.DataFrame(results, columns=["Product", "A", "B"])
    print("----- Averages for product ------")
    print(results_df)

    # print products where a < b
    print()
    print("Products where A < B:")
    for _, row in results_df.iterrows():
        if row["A"] < row["B"]:
            print(f"{row['Product']} (A={row['A']}, B={row['B']})")

    # visualizations
    x = results_df["Product"]
    plt.figure(figsize=(10,8))

    plt.bar(x, results_df["A"],width=0.5,label="Average of Age-Group Averages (A)", align='center')
    plt.bar(x, results_df["B"],width=0.5, label="Overal Average (B)", align='edge')

    plt.title("Comparison of A and B for every product")
    plt.xlabel("Product")
    plt.ylabel("Average Previous Purchases")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.show() 


# -----------------------------------------------------------------------------------
# Task 4: DAte Analysis
def date_analysis(df):
    # clean date columns
    df["Dates"] = pd.to_datetime(df["Dates"],errors='coerce')
    df = df.dropna(subset=["Dates"])
    
    # extract date
    df["Month"] = df["Dates"].dt.month
    df["Year"] = df["Dates"].dt.year

    # define seasons (4 seasons, start Mar 21 and lasts three months) 
    def get_season(date):
        month = date.month
        day = date.day
        if(month == 3 and day>=21) or month in [4,5] or (month == 6 and day <= 20):
            return "Spring" 
        elif(month == 6 and day>=21) or month in [7,8] or (month == 9 and day <= 20):
            return "Summer"
        elif (month == 9 and day>=21) or month in [10,11] or (month == 12 and day <=20):
            return "Autumn"
        else:
            return "Winter"
        
    df["Season"] = df["Dates"].apply(get_season)

    # ------ popularity of season ------
    season_counts = df["Season"].value_counts()
    print("Sales in season")
    print(season_counts)
    
    # visualizations
    sectionToExplode = (0.05, 0, 0,0)
    plt.figure(figsize=(6,6))
    season_counts.plot(kind="pie", autopct='%1.1f%%', shadow = True, startangle = 90, explode=sectionToExplode)
    plt.title("Seasonal Popularity (Sales Share)")
    plt.ylabel("")  
    plt.show()

    # ------ popularity of months ------
    month_counts = df["Month"].value_counts().sort_index()

    # print first 3 months
    top_months = month_counts.sort_values(ascending=False).head(3)
    
    print("\nMost 3 popular months:")

    month_names = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
    }

    for month, count in top_months.items():
        name = month_names[month]
        print(f"{name}: {count} sales")

    month_counts.index = month_counts.index.map(month_names)
    
    # visualization
    plt.figure(figsize=(10,6))
    plt.plot(
        month_counts.index,          # x-axis (month names)
        month_counts.values,         # y-axis (sales counts)
        marker='o'                            
    )
    plt.title("Monthly Popularity (Number of Sales)")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ----- yearly product sales ------
    def product_yearly_sale(prod_name):
        sub = df[df["Item Purchased"].str.lower() == prod_name.lower()]
        if sub.empty:
            print("No sales information for this product.")
            return
       
        print(f" \nYearly sales for {prod_name}")
        print(sub["Year"].value_counts().sort_index())

        plt.figure(figsize=(10,6))
        sns.histplot(
            data=sub,
            x="Year",             # which column to plot
            bins=len(sub["Year"].unique()),  # one bin per year
            kde=True
        )
        plt.title(f"Yearly Sales Distribution for {prod_name}")
        plt.xlabel("Year")
        plt.ylabel("Number of Sales")
        plt.tight_layout()
        plt.show()

    # ask user to type product name
    # prod_name = input("\nEnter the product name to see its yearly sales: ")

    # call the function
    # product_yearly_sale(prod_name)

    # example output
    sample = df["Item Purchased"].iloc[0]
    print(f"Example for {sample}")
    product_yearly_sale(sample)



# main function
def main():
    df = load_data()
    # Task 1
    # product_shipping(df)
    # Task 2
    # customer_segments(df)
    # Task 3
    # product_analysis(df)
    # Task 4
    date_analysis(df)


if __name__ == "__main__":
    main()
