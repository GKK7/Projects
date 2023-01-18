import pandas as pd
from sklearn.ensemble import IsolationForest
import openpyxl
import xlsxwriter
from pprint import pprint

df = pd.read_excel("Sales data.xlsx")
df= df.drop([113])


data = df[["Общо вписвания"]]

model = IsolationForest(contamination="auto")

model.fit(data)

anomalies = model.predict(data)

pprint(df[anomalies == -1])

# dtt=pd.DataFrame(anomalies)
# dtt.to_excel("Sales anomalies results.xlsx")




