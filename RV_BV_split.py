# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:42:17 2018

@author: salman
"""
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Quandl API key. Mandatory for high usage
quandl.ApiConfig.api_key = "9RSjfiE4HjFV_xv14Ve-"


if __name__ == "__main__":
    csvfile = "../data/bbg_data.csv"
    snp_df = pd.read_csv(csvfile)

    # Basic DF manipulation
    snp_df.dropna(subset=["PX_LAST"], inplace=True)
    snp_df = snp_df[["Dates", "PX_LAST"]]
    snp_df["Dates"] = pd.to_datetime(snp_df["Dates"])
    snp_df['weekday'] = snp_df['Dates'].dt.dayofweek
    snp_df = snp_df.set_index("Dates")

    # RV, BV and k calculations
    snp_df = snp_df.rename(columns={"PX_LAST": "Price"})
    snp_df["Return"] = np.log(snp_df["Price"]) -\
        np.log(snp_df["Price"].shift(1))
    snp_df["Return(-1)"] = snp_df["Return"].shift(1)
    snp_df["Return(-2)"] = snp_df["Return"].shift(2)
    snp_df["Return(-3)"] = snp_df["Return"].shift(3)
    snp_df["Return(-4)"] = snp_df["Return"].shift(4)
    snp_df["mu"] = np.sqrt((abs(snp_df["Return"]) +
                            abs(snp_df["Return(-1)"]) +
                            abs(snp_df["Return(-2)"]) +
                            abs(snp_df["Return(-3)"]) +
                            abs(snp_df["Return(-4)"]))/5)
    snp_df["RV"] = snp_df["Return"]**2 + snp_df["Return(-1)"]**2 +\
        snp_df["Return(-2)"]**2 + snp_df["Return(-3)"]**2 +\
        snp_df["Return(-4)"]**2
    snp_df["BV"] = abs(snp_df["Return"]*snp_df["Return(-1)"]) +\
        abs(snp_df["Return(-1)"]*snp_df["Return(-2)"]) +\
        abs(snp_df["Return(-2)"]*snp_df["Return(-3)"]) +\
        abs(snp_df["Return(-3)"]*snp_df["Return(-4)"])
    snp_df["k"] = np.sqrt(snp_df["RV"] - snp_df["BV"])
    snp_df = snp_df[snp_df["weekday"] == 4]
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        # print(len(snp_df[snp_df["k"] > 0.05])/len(snp_df))
        print(snp_df["k"].quantile(0.99))
        # print(snp_df.head(20))
    snp_df.to_csv("../data/snp_with_rv_bv.csv")
    # Plot
    plt.figure(1)
    plt.plot(snp_df.loc[snp_df.index > '2012-01-01']["RV"], label="RV")
    plt.plot(snp_df.loc[snp_df.index > '2012-01-01']["BV"], label="BV")
    plt.legend(loc="best")
    plt.title("Realized variance: Total vs. smooth")
    plt.show()
    plt.figure(2)
    plt.plot(snp_df.loc[snp_df.index > '2012-01-01']["k"], label="k")
    plt.legend(loc="best")
    plt.title("Jump volatility")
    plt.show()
