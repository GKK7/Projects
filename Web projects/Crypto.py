# import necessary modules
from bs4 import BeautifulSoup
import requests
import datetime
import pandas as pd


# create the file
writer = pd.ExcelWriter('data.xlsx', engine='xlsxwriter')

# data source
url = "https://coinmarketcap.com/"

# get the data from the source
result = requests.get(url).text
doc = BeautifulSoup(result, "html.parser")
tbody = doc.tbody
trs = tbody.contents

names = []
current_price = []
mkap = []
supply = []
volume = []

# extract the data
for tr in trs[:10]:
    name, price = tr.contents[2:4]
    fixed_name = name.p.string
    fixed_prices = price.a.string
    mcap, sup = tr.contents[8:10]
    fixed_mcap = mcap.a.string
    fixed_sup = sup.p.string
    vol = tr.contents[7]
    fixed_vol = vol.span.string

    names.append(fixed_name)
    current_price.append(fixed_prices)
    mkap.append(fixed_mcap)
    supply.append(fixed_sup)
    volume.append(fixed_vol)

# zip the extracted data together
final = (list(zip(names, current_price, mkap, volume, supply)))

# get the time of document creation
tm = datetime.datetime.now()
date_time = tm.strftime("%m/%d/%Y, %H:%M:%S")
stamp = f"Data generated at: {date_time}"

df = pd.DataFrame(final, columns=["Name", "Price", "Market Cap", "Volume", "Supply"])

# write the data in the workbook
df.to_excel(writer, sheet_name="Crypto Data", index=False)
# get workbook
workbook = writer.book
# get Crypto Data sheet
worksheet = writer.sheets['Crypto Data']
worksheet.write(15, 0, stamp)
# adjust column parameters
worksheet.set_column(0, 4, 20)

writer.close()
