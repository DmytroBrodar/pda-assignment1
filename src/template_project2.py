#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student Name: Dmytro Brodar
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
    data_path = os.path.join(folder,"..", "data", "shopping.csv")
    data_path = os.path.abspath(data_path)

    # read csv file
    df = pd.read_csv(data_path)

    # clean column names, remove extra spaces
    df.columns = df.columns.str.strip()

    # remove spaces from text columns
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip()

    return df

df = load_data()

# -----------------------------------------------------------------------------------
# Task 1: Analyze relationship between products and shipping types 
def product_shipping(df):
 
    # remove rows with missing values
    df = df.dropna(subset=["Item Purchased", "Shipping Type"])

    # count shipping for each product
    counts = df.groupby(["Item Purchased", "Shipping Type"]).size().reset_index(name='Count')

    counts["Percent"] = 0.0
    for product in counts["Item Purchased"].unique():
        sub_idx = counts["Item Purchased"] == product
        total = counts.loc[sub_idx, "Count"].sum()
        counts.loc[sub_idx, "Percent"] = (counts.loc[sub_idx, "Count"] / total * 100).round()
    
    counts = counts.sort_values(["Item Purchased", 'Percent'], ascending=[True, False])

    # print each product results
    for product in counts["Item Purchased"].unique():
        print(f"\nProduct: {product}")
        sub = counts[counts["Item Purchased"] == product]
        for _, row in sub.iterrows():
            print(f" {row["Shipping Type"]} - {row['Percent']}% (amount: {row['Count']})")

product_shipping(df)

# -----------------------------------------------------------------------------------
# Task2: customer segments separated by gender
def customer_segments(df):

    # convert numeric columns
    df["Purchase Amount (USD)"] = pd.to_numeric(df["Purchase Amount (USD)"],  errors='coerce')
    df["Previous Purchases"] = pd.to_numeric(df["Previous Purchases"], errors='coerce')

    # drop rows with invalid values
    df= df.dropna(subset=["Purchase Amount (USD)", "Previous Purchases", "Gender"])

    # new column
    df["Total Purchased USD"] = df["Purchase Amount (USD)"] * df["Previous Purchases"]

    # define segment
    def get_segment(total):
        if 0 <= total < 500:
            return "0-500"
        elif 500 <= total < 1000:
            return "500-1000"
        elif 1000 <= total < 1500:
            return "1000-1500"
        elif 1500 <= total < 2000:
            return "1500-2000"
        elif 2000 <= total < 2500:
            return "2000-2500"
        elif 2500 <= total < 3000:
            return "2500-3000"
        elif 3000 <= total < 3500:
            return "3000-3500"
        elif 3500 <= total < 4000:
            return "3500-4000"
        elif 4000 <= total < 4500:
            return "4000-4500"
        elif 4500 <= total < 5000:
            return "4500-5000"
        elif 5000 <= total < 5500:
            return "5000-5500"
        elif 5500 <= total <= 6000:
            return "5500-6000"
        else:
            return None
        
    df["Segment"] = df["Total Purchased USD"].apply(get_segment)

    # group by segment and gender
    group = df.groupby(["Segment", "Gender"]).size().unstack(fill_value = 0)

    # order in correct way
    order = [
        "0-500","500-1000","1000-1500","1500-2000","2000-2500","2500-3000",
        "3000-3500","3500-4000","4000-4500","4500-5000","5000-5500","5500-6000"
    ]

    group = group.reindex(order)

    # print population
    print("\nPopulation of each gender per segment")
    for seg, row in group.iterrows():
        female = row.get("Female", 0)
        male = row.get("Male", 0)
        print(f"{seg}: Female={female}, Male={male}")
    
    # visualization
    group.plot(kind='bar') 
    plt.title("Customer Segments by gender")
    plt.xlabel("Total Purchased USD Range")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# customer_segments(df)

# -----------------------------------------------------------------------------------
# Task 3: Product Analysis
def product_analysis(df):

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

# product_analysis(df)

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

    # product_yearly_sale(prod_name)

    # example output
    sample = df["Item Purchased"].iloc[0]
    print(f"Example for {sample}")
    product_yearly_sale(sample)

# date_analysis(df)

# -----------------------------------------------------------------------------------
# Task 5: My analytical task
# Find which customers age groups are the most loyal based on the previous purchases.
# I think this can help understand which customers return more often and plan some promotions for them. 
def customer_loyality(df):
    # convert columns to numeric
    df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
    df["Previous Purchases"] = pd.to_numeric(df["Previous Purchases"], errors='coerce')

    # drop rows with missing values
    df = df.dropna(subset=["Age", "Previous Purchases"])

    # make age groups
    def age_group(age):
        start = int(age // 10 * 10)
        end = start + 10
        return f"{start}-{end}"

    df["Age Groups"] = df["Age"].apply(age_group)

    #  find average prev purchases
    loyal = df.groupby("Age Groups")["Previous Purchases"].mean().sort_index()

    # print top5 loayal age groups
    print("Top 5 most loyal age groups:")
    top5 = loyal.sort_values(ascending=False).head(5)
    for age, avg in top5.items():
        print(f"{age}: {avg:.1f} average previous purchase") 

    # visualization
    plt.figure()
    loyal.plot(kind="bar")
    plt.title("Average Previous Purchases by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Average Previous Purchases")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# customer_loyality(df)

