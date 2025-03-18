from bs4 import BeautifulSoup
from urllib.request import urlopen

def Q1(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    buda_day_div = soup.find_all("div", class_="bud-day")
    day_map = {'จันทร์':0,
    'อังคาร':1,
    'พุธ':2,
    'พฤหัสบดี':3,
    'ศุกร์':4,
    'เสาร์':5,
    'อาทิตย์':6
    }
    day_count = [0]*7
    for buda_day in buda_day_div:
        col = buda_day.find("div", class_="bud-day-col")
        text = col.get_text(strip=True)
        text = text.split(" ")
        day = text[0][3:-3]
        day_count[day_map[day]]+=1
    
    return day_count
def Q2(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    date = soup.find("a", title="วันวิสาขบูชา").find_parent().find_previous_sibling().find_previous_sibling()
    text = date.get_text()
    
    return text

print(Q2('2566.html'))