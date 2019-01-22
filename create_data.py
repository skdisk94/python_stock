import requests
from bs4 import BeautifulSoup

code = '035720' # Kakao 주식 코드(035720)
src = 'https://finance.naver.com/item/sise.nhn?code='+code
response = requests.get(src)
bs = BeautifulSoup(response.content, 'html.parser')
title = bs.find('title')
name = title.text.replace(' : 네이버 금융','')

data =[]
stock=[]
page=1
overlap = True
while overlap:
    src='https://finance.naver.com/item/sise_day.nhn?code='+code+'&page='+str(page)
    response = requests.get(src)
    bs = BeautifulSoup(response.content, 'html.parser')
    table = bs.find('table',{'class':'type2'})
    tr = table.find_all('tr')
    for i in range(len(tr)-1):
        #(0, 1, 7, 8, 9, 15) 제외
         if i in (0, 1, 7, 8, 9, 15):
             continue

         td = tr[i].find_all('td')

         if td[0].text =='\xa0' or td[4].text == '0': # 빈 공백과 0이라는 이상치를 제거
             continue
         date = td[0].text
         closing_price = td[1].text.replace(',','') # 종가
         open_price = td[3].text.replace(',','')  # 시가
         high_price = td[4].text.replace(',','') # 고가
         low_price = td[5].text.replace(',','')  # 저가
         trading_volume = td[6].text.replace(',','')  # 거래량
         stock = [date, closing_price, open_price, high_price, low_price, trading_volume]
         if stock in data :
             overlap = False
         else:
             data.append(stock)
    page+=1

data.reverse()
with open('stock_data.csv', 'w', encoding='euc_kr') as file:
    # file.write('종목명,{0}\n'.format(name))
    file.write('날짜,종목명,종가,시가,고가,저가,거래량\n')
    for line in data:
        file.write('{0},{1},{2},{3},{4},{5},{6}\n'.format(line[0], name, line[1], line[2], line[3], line[4], line[5]))