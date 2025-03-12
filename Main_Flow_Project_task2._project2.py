import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load Dataset
df = pd.read_csv("sales-data.csv")

print("\nThe shape of the dataset is ", df.shape)

print("\n The datatype of the dataset is\n", df.dtypes)

# Data Cleaning

#handling missing values
df = df.drop(columns=["OrderProfitable", "SalesaboveTarget"], errors='ignore')
df.fillna(df.select_dtypes(include = ['float64', 'int64']).mean(), inplace = True)
df.fillna(df.select_dtypes(include = 'object').fillna('Unknown'), inplace = True)

print("\n The dataset after filling the missing values \n", df)

threshold = 2.5
for col in df.select_dtypes(include = ['int64', 'float64']).columns:
    std = df[col].std()
    mean = df[col].mean()
    zscore = (df[col] - mean)/std
    median = df[col].median()
    df[col] = np.where(abs(zscore)>threshold, median, df[col])
print("\n Data after replacing outliers with median:\n", df)

print("\n\n The check of missing values : \n", df.isnull().sum())

df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce") 
print("\nThe OrderData after conversion is \n", df["OrderDate"])


df = df.drop_duplicates() 
print("\n The dataset after droping the duplicates is \n", df)


# Predictive Modeling - Linear Regression
X = df[["Profit", "Discount"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Model Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR² Score: {r2}")
print(f"\nMean Squared Error: {mse}")

# Insights & Recommendations
print("\nInsights:")
print("* Sales trend shows peak seasons, optimize stock accordingly.")
print("* Higher discounts lead to lower profits.")
print("* Top-performing regions/categories should be focused on for marketing.")

# Exploratory Data Analysis (EDA)
OrderDate_sales = df.groupby("OrderDate")["Sales"].sum()
plt.figure(figsize=(9,9))
plt.subplot(2,2,1)
plt.plot(OrderDate_sales.index, OrderDate_sales)
plt.title("Sales Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)


plt.subplot(2,2,2)
sns.scatterplot(x=df["Discount"], y=df["Profit"])
plt.title("Profit vs. Discount")
plt.xlabel("Discount")
plt.ylabel("Profit")

Region_sales = df.groupby("Region")["Sales"].sum()
plt.subplot(2,2,3)
plt.bar(Region_sales.index, Region_sales, color="skyblue")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)


category_sales = df.groupby("Category")["Sales"].sum()

plt.subplot(2,2,4)
plt.pie(category_sales.values, labels=category_sales.index, autopct="%1.1f%%", startangle=180, colors=["lightblue", "orange", "lightgreen"])
plt.title("Sales Distribution by Category")


plt.tight_layout()
plt.show()


