# -*- coding: utf-8 -*-
import requests
from requests import RequestException
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
import os

x = 131  # 每条新闻的编号


def get_page_news():
    # 获取当前页面所有包含新闻的a标签
    news = browser.find_elements_by_xpath('//div[@class="box_left"]/div[@class="hlgd-content"]/div[@class="content"]/h2/a')
    for i in news:
        
        if i.text == "":
            continue
        link = i.get_attribute('href')  # 得到新闻url
        global x
        html = get_response(link)
        soup = BeautifulSoup(html, 'lxml')
        
        
        path = "E:/Project/文本分类/ml/data_spyder/IT/"
        
        s = ".txt"
        with open(path + str(x) + s, 'w', encoding='utf8') as file1:
            print(x)
            file1.write(i.text + "。")
        get_news(link)


def get_response(url):
    try:
        # 添加User-Agent，放在headers中，伪装成浏览器
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            response.encoding = 'utf-8'
            return response.text
        return None
    except RequestException:
        return None



def get_news(link):
    # 获取新闻的详细信息
    html = get_response(link)
    # 使用beautifulsoup进行解析
    soup = BeautifulSoup(html, 'lxml')
    # <h1 class="main-title">证监会要求北京银行说明是否串通*ST康得管理层舞弊</h1>
    # print(soup.meta.content)

    path = "E:/Project/文本分类/ml/data_spider/IT/"
    global x
    s = ".txt"
    with open(path + str(x) + s, 'a+', encoding='utf8') as file1:
        file1.write(soup.find(attrs={"name":"description"})['content'])
        x += 1

if __name__ == '__main__':
    # 打开浏览器
    # 打开网址
    #每页30,共2900
    for i in range(8,100):
        option = webdriver.ChromeOptions()
        option.add_argument("headless")
        browser = webdriver.Chrome(chrome_options=option)
        browser.implicitly_wait(1)
        browser.get('http://www.tw0755.cn/itzx/list_25_'+str(i+1)+'.html')
        # 获取当前页面新闻的url
        get_page_news()
        browser.quit()
