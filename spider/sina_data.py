# -*- coding: utf-8 -*-
import requests
from requests import RequestException
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
import os

x=0 #每条新闻的编号
def get_page_news():
    #获取当前页面所有包含新闻的a标签
    news = browser.find_elements_by_xpath('//div[@class="d_list_txt"]/ul/li/span/a')
    for i in news:
        link = i.get_attribute('href') 
        html = get_response(link)
        soup = BeautifulSoup(html, 'lxml')
        path= "E:/Project/文本分类/ml/test_spider/社会/"
        global x
        s=".txt"
        with open(path+str(x)+s, 'w', encoding='utf8') as file1:
            file1.write(i.text+"。")
        print(i.text)
        get_news(link)

def get_response(url):
    try:
        #添加User-Agent，放在headers中，伪装成浏览器
        headers = {
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
        }
        response = requests.get(url,headers=headers)
        if response.status_code == 200:
            response.encoding = 'utf-8'
            return response.text
        return None
    except RequestException:
        return None

def get_news(link):
    html = get_response(link)
    soup = BeautifulSoup(html, 'lxml')
    path= "E:/Project/文本分类/ml/test_spider/社会/"
    global x
    s=".txt"
    with open(path+str(x)+s, 'a+', encoding='utf8') as file1:
        file1.write(soup.find(attrs={"name":"description"})['content'])
        x+=1
        


if __name__ == '__main__':
    # 打开浏览器
    for i in range(50):
        option = webdriver.ChromeOptions()
        option.add_argument("headless")
        browser = webdriver.Chrome(chrome_options=option)
        #设置间隔时间
        browser.implicitly_wait(1) 
        browser.get('https://news.sina.com.cn/roll/#pageid=153&lid=2516&k=&num=50&page=' + str(i + 1))
        get_page_news()
        browser.quit() 










