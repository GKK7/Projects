# Import the required modules
import tabula
import pandas as pd

# Read pdf into DataFrame
df_list = tabula.read_pdf('Real estate sales.pdf', pages=1)

for df in (df_list):
    df.to_excel(f"Real estate.xlsx", index=False)

# Export data to excel without the headers
dft = pd.read_excel("Real estate.xlsx", header=None)

# Remove the first two lines as they are leftover text from the pdf file, formatting table
dft = dft.drop([0, 1])
dft.columns = range(dft.shape[1])
dft.to_excel("Real estate.xlsx", index=False, header=False)

df = pd.read_excel('Real Estate.xlsx')

# Write the dataframe to a new excel file
df.to_excel('Real Estate.xlsx', index=False)


class RunML:
    def __init__(self):
        print("""Utilize different machine learning methods to analyze and make predictions on the Real Estate market. 
The base data is from the The National Agency of Public Registry Q4 2022:
https://www.registryagency.bg/bg/registri/imoten-registar/statistika/""")

        print("Select type of method:\n"
              "1: Linear Regression\n"
              "2: Anomaly\n"
              "3: KMeans cluster\n"
              "4: other\n"
              "5: Quit")

    choices = {1: "Linear Regression", 2: "Anomaly", 3: "KMeans cluster", 4: "other", 5: "Quit"}

    def run(self):
        while True:
            try:
                choice = int(input("Select the ML type or 5 to quit \n"))
                if choice not in self.choices.keys():
                    print("Unavailable option")
                    continue
            except ValueError:
                print("Integers 1-5 only")

            else:
                print(f"You've chosen {self.choices[choice]}")
                if choice == 1:
                    import Regression
                if choice == 2:
                    import Anomaly
                if choice == 3:
                    import KM_cluster
                if choice == 4:
                    pass
                if choice == 5:
                    quit()
                    break


run_ml = RunML()
run_ml.run()
