import yaml
import pandas as pd


def read_yaml(config):
    output = None
    
    with open(config, 'r') as stream:
        try:
            output = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    return output


def write_analysis_report(url, top_reviews, ratings_summary):
    # Writing to text file
    with open('./Results/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('------------------------------Analysis of Shopee Reviews------------------------------\n')
        f.write('\n')
        f.write('Product scraped: ' + url + '\n')
        f.write('\n')
        f.write('Top reviews are:\n')
        for i in top_reviews:
            f.write('- ' + i + '\n')
        f.write('\n')
        for i in range(1, 6):
            f.write('Key phrases for reviews with rating ' + str(i) + ':\n')
            for j in range(len(ratings_summary[i])):
                f.write('- ' + ratings_summary[i][j] + '\n')
            f.write('\n')