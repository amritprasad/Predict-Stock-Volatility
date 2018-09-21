# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:15:21 2018

@author: salman
"""
# http://theautomatic.net/2017/08/24/scraping-articles-about-stocks/
import requests
from bs4 import BeautifulSoup
from random import randint
import time
from datetime import datetime, timedelta
import json

start_date = "2016-12-31"
end_date = "2014-12-31"
filename = "Data from " + start_date + " to " + end_date + ".txt"
curr_date = datetime.strptime(start_date, "%Y-%m-%d")
data_dict = {}
num_days_done = 0

while curr_date > datetime.strptime(end_date, "%Y-%m-%d"):
    if not num_days_done % 10:
        print(num_days_done, " days done")

    date_str = datetime.strftime(curr_date, "%Y-%m-%d")
    site = "http://www.wsj.com/public/page/archive-" + date_str + ".html"
    '''scrape the html of the site'''
    html = requests.get(site).content

    '''convert html to BeautifulSoup object'''
    soup = BeautifulSoup(html, 'lxml')

    '''get list of all links on webpage'''
    links = soup.find_all('a')
    urls = [(link.get('href'), link.getText()) for link in links]
    urls = [url for url in urls if url[0] is not None]
    urls = [url for url in urls if '/articles/' in url[0]]
    news_hls = [url[1] for url in urls]
    data_dict[date_str] = {}

    if news_hls:
        data_dict[date_str] = news_hls

    curr_date = curr_date - timedelta(1)
    num_days_done += 1
    time.sleep(randint(5, 10))

with open(filename, 'w') as file:
    file.write(json.dumps(data_dict))
