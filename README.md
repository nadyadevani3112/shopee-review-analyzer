## Contributors
Leow Wei Sheng, Nadya Devani, Lee Jinghan, Kok Yi Ling

# Shopee Review Analyzer
This repository contains a tool that users can use to extract key insights from large numbers of Shopee product reviews without having to go through the hassle of reading each review. This tool is meant to be easy to use, and aims to help users answer questions such as "Why are people giving a rating of only 1 star?", or "What proportion of reviews are 5 star reviews?" – questions which most online shoppers would consider before buying a product.

## Installation Guide
1. Clone or download this repository to your local machine
2. Navigate to your local directory and install the packages stated in *requirements.txt* by running `pip3 install -r requirements.txt` in the command line
3. Download and install ChromeDriver from [here](https://chromedriver.chromium.org/downloads). Our tool requires Google Chrome to work successfully

## Usage Guide
1. The config file *config.yaml* is where the user sets the configs before running the tool. There are two configs to be set:
    - `webdriver_path`: Path to your ChromeDriver (e.g. "C:\chromedriver")
    - `url`: The url of the Shopee product of interest
2. Users can run this tool either using the Jupyter Notebook file *shopee_review_analyzer.ipynb* or the Python file *shopee_review_analyzer.py*
3. After analysing the reviews of the product stated in Step 1, a folder called "Results" will be created to store the results of the analysis. Inside this folder, there will be a csv file listing the product reviews and a text file containing key insights from the reviews, such as key phrases mentioned by users who gave a particular rating

## Troubleshooting Guide
- If the product webpage does not load fully, scroll through the page manually once and the page should load as usual thereafter

