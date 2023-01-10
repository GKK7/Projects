# The program scrapes data from bbc.com and produces an output of the top 10 articles and their links

# Import the required modules
import requests
from bs4 import BeautifulSoup
from pprint import pprint

# Make a request to the BBC homepage
URL = "https://www.bbc.com/"
page = requests.get(URL)

# Parse the HTML
soup = BeautifulSoup(page.content, 'html.parser')

headlines = []
links = []

# Find the top 10 headlines and their links by looking for the "h3" and "a[href]" elements and format the data
for h3 in soup.find_all('h3')[:10]:
    headline = h3.get_text()
    headlines.append(headline.strip())
    for a in h3.find_all('a')[:10]:
        link = a['href']
        links.append(link.strip())

# Some output may not print out in the first part of the link, so it's added to each element
links = ["https://www.bbc.com" + site if not site.startswith("https://www.bbc.com") else site for site in links]

# Create a new dictionary by zipping the headlines and links list
final = dict(zip(headlines, links))

# Use pprint module to format the output
pprint(final)
