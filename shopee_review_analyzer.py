# Import Modules
import utils
import nlp_tools
import scraping_tools


# Scraping Reviews
config = utils.read_yaml('config.yaml')
webdriver_path = config['webdriver_path']
url = config['url']
reviews = scraping_tools.shopee_scraper(webdriver_path, url)


# Ratings Proportion
proportion = nlp_tools.get_ratings_proportion(reviews)


# Get Top Reviews
sentences = nlp_tools.tokenize_reviews(reviews['Reviews'])
top_reviews = nlp_tools.get_top_reviews(sentences)


# Get Key Phrases
ratings_summary = nlp_tools.get_key_phrases(reviews)


# Get Analysis Report
utils.write_analysis_report(url, proportion, top_reviews, ratings_summary)