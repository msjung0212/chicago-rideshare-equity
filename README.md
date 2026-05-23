# Rideshare Pricing and Access Equity in Chicago
**Authors:** Minseo Jung, Daniela Chafloque
**Course:** DATA259 Data Ethics — University of Chicago, Spring 2026
**Repository:** https://github.com/msjung0212/chicago-rideshare-equity
## Overview
This repository contains the code, data processing pipeline, and outputs for our 
DATA259 final paper investigating equity in Chicago's rideshare market. Using City
of Chicago Transportation Network Provider (TNC) trip data from January 2024, we
test whether riders in transit-poor, lower-income, or majority-minority neighborhoods
pay disproportionately more per mile than riders in wealthier, transit-rich
neighborhoods, and compare rideshare pricing patterns to Chicago's regulated taxi
market as a counterfactual.
**Research question:**
Does rideshare pricing in Chicago produce systematic per-mile cost disparities across
neighborhoods defined by transit access, income, and racial composition — and does
any such disparity persist relative to regulated taxi pricing, isolating the role of
the algorithmic pricing mechanism itself?
**Main findings:**
1. No statistically significant per-mile price premium in transit-deprived
   neighborhoods across six regression specifications (β = -0.260, p = 0.126)
2. A nearly five-fold access gap: affluent neighborhoods generate 2,180 rideshare
   trips per 1,000 residents vs. 443 in high-poverty transit-desert neighborhoods
3. Income burden nearly three times higher in transit-desert neighborhoods (3.26%
   of median household income) vs. affluent neighborhoods (1.20%) for equivalent
   rideshare use
4. Estimated 3.88 million annual foregone trips in transit-desert neighborhoods
## Repository Structure
```
.
chicago-rideshare-equity/ 
│ ├── data/ │ 
├── raw/ ← not included (see Data Access below) 
│ └── processed/ ← not included (see Data Access below) 
│ ├── src/ 
│ ├── 01_clean_tnc.py ← clean rideshare trip data 
│ ├── 02_clean_taxi.py ← clean taxi trip data 
│ ├── 03_clean_acs.py ← clean ACS demographics 
│ ├── 04_merge.py ← merge trips with demographics 
│ ├── 05_eda.py ← exploratory data analysis figures 
│ ├── 06_cluster_neighborhoods.py ← k-means neighborhood clustering 
│ ├── 07_regressions.py ← regression models 1-6 
│ ├── 08_equity_burden.py ← equity burden analysis 
│ └── download_data.py ← download data from Chicago Data Portal 
│ ├── notebooks/ 
│ └── 05_eda_jan2024.ipynb ← exploratory analysis notebook 
| └── Transportation.ipynb ← Data cleaning for TNP and some exploratory analysis notebook 
| └── Data_Analysis_Models.ipynb ← Data Analysis of OLS Regression to see residuals + ACS exploratory data visuals
│ ├── outputs/ 
│ ├── figures/ ← all figures used in paper and poster 
│ └── tables/ ← all CSV output tables 
│ ├── requirements.txt 
└── README.md
```
## Data Access
Raw data files are not included in this repository due to file size constraints
### 1. Chicago TNC (Rideshare) Trips — January 2024
**Source:** City of Chicago Data Portal
**Dataset:** Transportation Network Providers - Trips (2024)
**Link:** https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2023-2024-/n26f-ihde/about_data
Filter for January 2024
### 2. Chicago Taxi Trips — January 2024
**Source:** City of Chicago Data Portal
**Dataset:** Taxi Trips
**Link:** https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew/about_data
### 3. ACS Community Area Demographics
**Source:** City of Chicago Data Portal
**Dataset:** Selected Socioeconomic Indicators by Community Area (2023 ACS
5-year estimates)
**Link:** https://catalog.data.gov/dataset/acs-5-year-data-by-community-area?
## Installation
### Requirements
- Python 3.9 or higher
- pip
### Setup
**1. Clone the repository:**
**2. Create and activate a virtual environment:**
**3. Install dependencies:**
pip install -r requirements.txt



