# import required modules
import pandas as pd
from sklearn.model_selection import train_test_split
# Read the data from an Excel file
data = pd.read_excel('Sales Data.xlsx')
from sklearn.linear_model import LinearRegression

# Define the features and target
X = data[['Възбрани', "Продажби"]]
y = data['Общо вписвания']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# predict test data the test data
y_pred = model.predict(X_test)

# Slope of the line for each independent variable
print("Coefficients: ", model.coef_)

# Intercept: using the intercept_ attribute. This represents the point at which the line intercepts the y-axis.

print("Intercept: ", model.intercept_)

# R-squared: This value ranges between 0 and 1, and represents the proportion of the variance in the dependent variable
# that is predictable from the independent variable.

print("R-squared: ", model.score(X, y))

print(f"""The coefficients of a linear regression model represent the change in the outcome variable for a one unit 
change in the predictor variable, holding all other predictor variables constant. 
In this case, the coefficient is {model.coef_}, meaning that for every one unit change in the predictor variable, 
the outcome variable is expected to change by {model.coef_} units.

The intercept is the value of the outcome variable when all predictor variables are equal to zero. 
In this case, the intercept is {model.intercept_}.

The R-squared value is a measure of how well the model fits the data, with a value of 1 indicating a perfect fit
and a value of 0 indicating no fit. In this case, the R-squared value is {model.score(X, y)}, which is close to 1,
indicating that the model fits the data very well.""")
