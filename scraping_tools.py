import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import *


def get_num_iterations(webdriver_path, url):
    num_reviews = 0
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('start-maximized')
    options.add_argument('disable-infobars')
    options.add_argument('--disable-extensions')

    browser = webdriver.Chrome(webdriver_path, options=options)
    browser.get(url)
    browser.implicitly_wait(10)

    source = browser.page_source
    soup = BeautifulSoup(source, 'lxml')
    for i in soup.find_all('div', class_='OitLRu'):
        num_reviews = i.text
    print('We found ' + str(num_reviews) + ' reviews to be scraped!')

    # Check if num_reviews is an integer such as '368' or a string such as '1.2k'
    # If string, get upper bound of number of reviews (e.g. 1.2k might be up to 1299, so round up to 1300)
    if type(num_reviews) == str and 'k' in num_reviews:
        num_reviews = num_reviews[:-1]
        num_reviews = float(num_reviews) + 0.1
        num_reviews = int(num_reviews * 1000)
    else:
        num_reviews = int(num_reviews) + 100 # Additional iterations to be safe
        
    # Set num_iterations to determine the end point of the webscraping process
    # Each page of reviews has 6 reviews
    num_iterations = int(num_reviews / 6) + 1
    return num_iterations


def process_shopee_df(df):
    # Remove empty reviews, inappropriate reviews, and duplicate reviews
    df = df[df['Reviews'] != '']
    df = df[df['Reviews'] != '******The review has been hidden due to inappropriate content.******']
    df = df.drop_duplicates()
    return df


def shopee_scraper(webdriver_path, url):
    options = webdriver.ChromeOptions()
    options.add_argument('start-maximized')
    options.add_argument('disable-infobars')
    options.add_argument('--disable-extensions')

    browser = webdriver.Chrome(webdriver_path, options=options)
    browser.get(url)

    comment_list = []
    star_rating = []
    username_list = []
    datetime_list = []
    
    # Get number of pages of reviews to scrape
    num_iterations = get_num_iterations(webdriver_path, url)
    
    try:
        browser.implicitly_wait(30)
        
        try:
            for i in range(num_iterations):
                # Scroll to the bottom of the page as Shopee uses lazy loading
                # Scroll partially first to load reviews
                browser.execute_script('window.scrollTo(0, 1620)')
                time.sleep(0.1) # To ensure that reviews load
                browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                
                # Reviews
                for e in browser.find_elements_by_class_name('shopee-product-rating__content'):
                    comment_list.append(e.text)

                # Star ratings
                for rating in browser.find_elements_by_css_selector('.shopee-product-rating'):
                    stars = rating.find_elements_by_css_selector('.icon-rating-solid')
                    star_rating.append(len(stars))
                    
                # Usernames
                for usernames in browser.find_elements_by_class_name('shopee-product-rating__author-name'):
                    username_list.append(usernames.text)
                    
                # Datetimes
                for datetimes in browser.find_elements_by_class_name('shopee-product-rating__time'):
                    datetime_list.append(datetimes.text)
                
                # Find the next page button to view more reviews
                for e in browser.find_elements_by_xpath('//button[@class="shopee-icon-button ' + 
                                                        'shopee-icon-button--right " and ' + 
                                                        'not(@aria-disabled)]'):
                    browser.implicitly_wait(10)
                    e.click()
            
            browser.quit()
            print('Webscrape successful!')
        
        except Exception as e:
            browser.quit()
            print(e)
    
    except NoSuchElementException as e: 
        browser.quit()
        print(e)
        
    df = pd.DataFrame(list(zip(comment_list, star_rating, username_list, datetime_list)), 
                      columns = ['Reviews', 'Ratings', 'Usernames', 'Datetimes'])
    df = process_shopee_df(df)
    os.makedirs('./Results/', exist_ok=True) 
    df.to_csv('./Results/scraped_reviews.csv', index=False, encoding='utf-8-sig')
    return df