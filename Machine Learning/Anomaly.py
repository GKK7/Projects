import pandas as pd
from sklearn.ensemble import IsolationForest
import openpyxl
import xlsxwriter


df = pd.read_excel("Real Estate.xlsx")
df= df.drop([113])


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




