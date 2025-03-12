import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv('Global_Superstore.csv', encoding = 'Latin1')

print("\n The head of the dataset is \n", data.head())

#handling missing values
data.fillna(data.select_dtypes(include = ['float64', 'int64']).mean(), inplace = True)
data.fillna(data.select_dtypes(include = 'object').fillna('Unknown'), inplace = True)

print("\n The dataset after handling missing values \n", data)

#removing duplicate to maintain data integrity

data.drop_duplicates(inplace = True)

print("\n The dataset after removing duplicates \n", data)


#handling the outliers using z score

threshold = 2.5

for col in data.select_dtypes(include = ['int64', 'float64']).columns:
    std = data[col].std()
    mean = data[col].mean()
    zscore = (data[col] - mean)/std
    median = data[col].median()
    data[col] = np.where(abs(zscore)>threshold, median, data[col])

print("\n Data after replacing outliers with median:\n", data)

#mean

print("\n The average profit, sales and quantity is \n", data[['Sales', 'Profit', 'Quantity']].mean())

#median

print("\n The median profit, sales and quantity is \n", data[['Sales', 'Profit', 'Quantity']].median())

#standard deviation

print("\n The standard deviation of profit, sales and quantity is \n", data[['Sales', 'Profit', 'Quantity']].std())

#variance

print("\n The variance of profit, sales and quantity is \n", data[['Sales', 'Profit', 'Quantity']].var())

#correlation between the variables

print("\n The correlation between sales and profit is ", data['Sales'].corr(data['Profit']))

print("\n Summary of the data \n", data.describe())

data.to_csv("cleaned_data.csv", index=False)

#histograms

plt.figure(figsize = (9,9))
plt.subplot(2,2,1)
plt.hist(data['Sales'], bins = 20, color = 'green')
plt.xlabel('Sales')
plt.ylabel('Count')
plt.title('Histogram of Sales')

#Use boxplots to identify outliers in continuous variables.
plt.subplot(2,2,2)
plt.boxplot([data['Sales'], data['Profit'], data['Quantity']])
plt.xticks([1, 2, 3], ['Sales', 'Profit', 'Quantity']) 
plt.xlabel('Features')
plt.ylabel('Count')
plt.title('Boxplot of Sales')

#Use heatmaps to visualize correlations and relationships between features.
plt.subplot(2,2,3)
sns.heatmap( data[['Sales', 'Profit', 'Quantity']].corr() , annot = True, cmap = 'coolwarm', linewidths = 0.5)
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()

