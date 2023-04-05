# -*- coding: utf-8 -*-
"""
@author: Durga
"""

# importing requiredpackages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_file(file_name):
    """
    This function will read data from all required data sets
    """

    address = file_name
    df = pd.read_csv(address)
    df_transpose = pd.DataFrame.transpose(df)
    # Header setting
    header = df_transpose.iloc[0].values.tolist()
    df_transpose.columns = header
    return (df, df_transpose)


def clean_df(df):
    """
    Function to be cleaned and index convertor..
    """

    # Cleaning the dataframe
    df = df.iloc[1:]
    df = df.iloc[11:55]

    # Converting index ot int
    df.index = df.index.astype(int)
    df = df[df.index > 1989]

    # cleaning empty cells
    df = df.dropna(axis="columns")
    return df


# Creates a list of countries and years to use in the plots
countries = [
    "Bangladesh",
    "China",
    "France",
    "United Kingdom",
    "India",
]
years = [1990, 1994, 1998, 2002, 2006, 2010, 2014]


# Reads the files
df_co2_total, df_co2_countries = read_file("Co2 Emission.csv")
df_electric_total, df_electric_countries = read_file("Electricity_consumption.csv")
df_urban_total, df_urban_countries = read_file("Access to Electricity Urban.csv")
df_energy_total, df_energy_countries = read_file("Energy use.csv")
df_emtvsele_total, df_emtvsele_countries = read_file("Co2 Emission Compare.csv")


"""
CO2 emissions (kt) bar graph
Creating bar graph of CO2 emissions (kt) of five countries from 1990-2014
"""

# Cleaning the dataframe
df_co2_countries = clean_df(df_co2_countries)

# selecting only required data
df_co2_time = pd.DataFrame.transpose(df_co2_countries)
df_co2_subset_time = df_co2_time[years].copy()
df_co2_subset_time = df_co2_subset_time.loc[df_co2_subset_time.index.isin(countries)]

# plotting the data
n = len(countries)
r = np.arange(n)
width = 0.1

plt.bar(
    r - 0.3,
    df_co2_subset_time[1990],
    color="lightblue",
    width=width,
    edgecolor="black",
    label="1990",
)
plt.bar(
    r - 0.2,
    df_co2_subset_time[1994],
    color="khaki",
    width=width,
    edgecolor="black",
    label="1995",
)
plt.bar(
    r - 0.1,
    df_co2_subset_time[1998],
    color="plum",
    width=width,
    edgecolor="black",
    label="2000",
)
plt.bar(
    r,
    df_co2_subset_time[2002],
    color="turquoise",
    width=width,
    edgecolor="black",
    label="2005",
)
plt.bar(
    r + 0.1,
    df_co2_subset_time[2006],
    color="chartreuse",
    width=width,
    edgecolor="black",
    label="2010",
)
plt.bar(
    r + 0.2,
    df_co2_subset_time[2010],
    color="aqua",
    width=width,
    edgecolor="black",
    label="2014",
)
plt.bar(
    r + 0.3,
    df_co2_subset_time[2014],
    color="darkgrey",
    width=width,
    edgecolor="black",
    label="2014",
)
plt.xlabel("Countries")
plt.ylabel("CO2 emissions (kt)")
plt.xticks(width + r, countries, rotation=90)
plt.legend()
plt.title("CO2 emissions (kt)")
plt.savefig("CO2 emissions (kt).png", bbox_inches="tight", dpi=500)
plt.show()


"""
Electricty use (kWh) bar graph
Creating bar graph of Electricty use (kWh) by five countries from 1990-2014
"""

# Cleaning the dataframe
df_electric_countries = clean_df(df_electric_countries)

# selecting only required data
df_electric_time = pd.DataFrame.transpose(df_electric_countries)
df_electric_subset_time = df_electric_time[years].copy()
df_electric_subset_time = df_electric_subset_time.loc[
    df_electric_subset_time.index.isin(countries)
]

# plotting the data
n = len(countries)
r = np.arange(n)
width = 0.1
plt.bar(
    r - 0.3,
    df_electric_subset_time[1990],
    color="aqua",
    width=width,
    edgecolor="black",
    label="1990",
)
plt.bar(
    r - 0.2,
    df_electric_subset_time[1994],
    color="aquamarine",
    width=width,
    edgecolor="black",
    label="1994",
)
plt.bar(
    r - 0.1,
    df_electric_subset_time[1998],
    color="azure",
    width=width,
    edgecolor="black",
    label="1998",
)
plt.bar(
    r,
    df_electric_subset_time[2002],
    color="beige",
    width=width,
    edgecolor="black",
    label="2002",
)
plt.bar(
    r + 0.1,
    df_electric_subset_time[2006],
    color="blue",
    width=width,
    edgecolor="black",
    label="2006",
)
plt.bar(
    r + 0.2,
    df_electric_subset_time[2010],
    color="darkgrey",
    width=width,
    edgecolor="black",
    label="2010",
)
plt.bar(
    r + 0.3,
    df_electric_subset_time[2014],
    color="gold",
    width=width,
    edgecolor="black",
    label="2014",
)
plt.xlabel("Countries")
plt.ylabel("Electricity Use (kWh)")
plt.xticks(width + r, countries, rotation=90)
plt.legend()
plt.title("Electric power consumption")
plt.savefig("Electricty.png", dpi=500, bbox_inches="tight")
plt.show()

"""
Access to electricity (% of population) plot graph
Creates a plot chart of Access to electricity (% of population)
of five countries during 1990-2014
"""
# Cleaning the dataframe
df_urban_countries = df_urban_countries.iloc[1:]
df_urban_countries = df_urban_countries.iloc[11:55]
df_urban_countries.index = df_urban_countries.index.astype(int)
df_urban_countries = df_urban_countries[df_urban_countries.index > 1990]

# plotting the data
plt.plot(df_urban_countries.index, df_urban_countries["Bangladesh"])
plt.plot(df_urban_countries.index, df_urban_countries["China"])
plt.plot(df_urban_countries.index, df_urban_countries["France"])
plt.plot(df_urban_countries.index, df_urban_countries["United Kingdom"])
plt.plot(df_urban_countries.index, df_urban_countries["India"])
plt.xlabel("Year")
plt.ylabel("Access to electricity Urban")
plt.legend(
    [
        "Bangladesh",
        "China",
        "France",
        "United Kingdom",
        "India",
    ],
    prop={"size": 8},
)
plt.title("Access to electricity, urban (% of urban population)")
plt.savefig("Access to electricity.png", dpi=500, bbox_inches="tight")
plt.show()


"""
Scatter plot Energy Use vs Co2 Emission
Creates a scatter plot of PEnergy Use vs Co2 Emission of five countries during 1990-2014.
"""
df_emtvsele_countries = df_emtvsele_countries.iloc[1:]
df_emtvsele_countries = df_emtvsele_countries.iloc[11:55]
df_emtvsele_countries.index = df_emtvsele_countries.index.astype(int)
df_emtvsele_countries = df_emtvsele_countries[df_emtvsele_countries.index > 1990]


# plotting the data
plt.figure()
plt.scatter(
    df_urban_countries["Bangladesh"], df_emtvsele_countries["Bangladesh"], alpha=0.5
)
plt.scatter(df_urban_countries["China"], df_emtvsele_countries["China"], alpha=0.5)
plt.scatter(df_urban_countries["France"], df_emtvsele_countries["France"], alpha=0.5)
plt.scatter(
    df_urban_countries["United Kingdom"],
    df_emtvsele_countries["United Kingdom"],
    alpha=0.5,
)
plt.scatter(df_urban_countries["India"], df_emtvsele_countries["India"], alpha=0.5)
plt.xlabel("Energy use (kg of oil equivalent per capita) - World")
plt.ylabel("co2 Emission")
plt.legend(
    ["Bangladesh", "China", "France", "United Kingdom", "India"],
    prop={"size": 7},
)
plt.title("Energy Use vs Co2 Emission")
plt.savefig("Energy Use vs Co2 Emission.png", dpi=500, bbox_inches="tight")
plt.show()


"""
Energy use  plot graph
Creates a plot chart of Energy use 
of five countries during 1990-2014
"""
# Cleaning the dataframe
df_energy_countries = df_energy_countries.iloc[1:]
df_energy_countries = df_energy_countries.iloc[11:55]
df_energy_countries.index = df_energy_countries.index.astype(int)
df_energy_countries = df_energy_countries[df_energy_countries.index > 1990]

# plotting the data
plt.plot(df_energy_countries.index, df_energy_countries["Bangladesh"])
plt.plot(df_energy_countries.index, df_energy_countries["China"])
plt.plot(df_energy_countries.index, df_energy_countries["France"])
plt.plot(df_energy_countries.index, df_energy_countries["United Kingdom"])
plt.plot(df_energy_countries.index, df_energy_countries["India"])
plt.xlabel("Year")
plt.ylabel("Energy use (kg of oil equivalent per capita) - World")
plt.legend(
    [
        "Bangladesh",
        "China",
        "France",
        "United Kingdom",
        "India",
    ],
    prop={"size": 8}
)
plt.title("Energy use (kg of oil equivalent per capita) - World")
plt.savefig("Energy Use.png", dpi=500, bbox_inches="tight")
plt.show()

# Sorting in terms of co2 emission

df_co2_countries_n = df_co2_total[
    df_co2_total["Country Name"].isin(countries)
]
print(df_co2_countries_n.describe().iloc[:, -5:])

dt_co2_sorted=df_co2_countries_n.sort_values(by='Country Name',ascending=False)
print(dt_co2_sorted["Country Name"])
