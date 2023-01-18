print("""Goal: analyze the real estate market in Bulgaria based on data from Q4 2022. 
Employ different methods and compare results.
The base data is from the The National Agency of Public Registry:
https://www.registryagency.bg/bg/registri/imoten-registar/statistika/""")

# Import the required modules
import tabula
import pandas as pd
from sklearn.ensemble import IsolationForest
import openpyxl
import xlsxwriter
import time

# Read pdf into DataFrame
df_list = tabula.read_pdf('Real estate sales.pdf', pages=1)

for df in (df_list):
    df.to_excel(f"Real estate.xlsx", index=False)

# Export data to excel without the headers
dft = pd.read_excel("Real estate.xlsx", header=None)

# Remove the first two lines as they are leftover text from the pdf file
dft = dft.drop([0, 1])

# Final formatting of the table
dft.columns = range(dft.shape[1])
dft.to_excel("Real estate.xlsx", index=False, header=False)

time.sleep(2)


def anomaly_model():
    anom = pd.read_excel("Real estate.xlsx")
    anom = anom.drop([113])

    # Use 2 variables - total registrations and foreclosures to identify the anomalies in the list of cities
    data = anom[["Общо вписвания", "Възбрани"]]

    # Create an instance of the IsolationForest anomaly detection model
    model = IsolationForest(contamination="auto")

    # Fit the data into the model
    model.fit(data)

    # predict() anomalies
    anomalies = model.predict(data)

    # display only the anomalies
    print(anom[anomalies == -1])

    print("The model displays the cities identified as anomalies according to the predefined variables")


anomaly_model()

