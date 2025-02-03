Zomato Data Analysis Using Python

Overview

This project focuses on analyzing Zomato restaurant data using Python and its libraries. The analysis helps answer key questions regarding restaurant preferences, pricing, and online versus offline ordering trends.

Technologies Used

The following Python libraries were used for the analysis:

Numpy – Enables fast and efficient numerical computations.

Matplotlib – Used for generating high-quality plots, charts, and histograms.

Pandas – Facilitates data loading and manipulation with DataFrames.

Seaborn – Provides an intuitive way to generate visually appealing statistical graphics.

Questions Addressed

Do more restaurants provide online delivery compared to offline services?

What types of restaurants are the most preferred by customers?

What is the preferred price range for couples dining at restaurants?

Data Preparation Steps

Step 1: Import Necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Step 2: Load the Dataset

dataframe = pd.read_csv("Zomato data.csv")
print(dataframe.head())

Step 3: Data Cleaning and Processing

Convert the "rate" column to float by removing the denominator.

def handleRate(value):
    value = str(value).split('/')
    return float(value[0])

dataframe['rate'] = dataframe['rate'].apply(handleRate)
print(dataframe.head())

Step 4: Data Summary and Null Value Check

dataframe.info()

Output shows no null values.

Data Analysis

1. Restaurant Types Distribution

sns.countplot(x=dataframe['listed_in(type)'])
plt.xlabel("Type of restaurant")
plt.show()

Conclusion: Dining restaurants are the most common.

2. Votes Distribution by Restaurant Type

grouped_data = dataframe.groupby('listed_in(type)')['votes'].sum()
result = pd.DataFrame({'votes': grouped_data})
plt.plot(result, c='green', marker='o')
plt.xlabel('Type of restaurant', c='red', size=20)
plt.ylabel('Votes', c='red', size=20)
plt.show()

Conclusion: Dining restaurants receive the highest votes.

3. Restaurant with Maximum Votes

max_votes = dataframe['votes'].max()
restaurant_with_max_votes = dataframe.loc[dataframe['votes'] == max_votes, 'name']
print('Restaurant(s) with the maximum votes:')
print(restaurant_with_max_votes)

Output: Empire Restaurant has the highest votes.

4. Online vs. Offline Orders

sns.countplot(x=dataframe['online_order'])
plt.show()

Conclusion: Most restaurants do not accept online orders.

5. Ratings Distribution

plt.hist(dataframe['rate'], bins=5)
plt.title('Ratings Distribution')
plt.show()

Conclusion: Most ratings fall between 3.5 and 4.

6. Preferred Price Range for Couples

sns.countplot(x=dataframe['approx_cost(for two people)'])
plt.show()

Conclusion: The most preferred cost for couples is around 300 rupees.

7. Ratings Comparison: Online vs Offline Orders

plt.figure(figsize=(6,6))
sns.boxplot(x='online_order', y='rate', data=dataframe)
plt.show()

Conclusion: Online orders receive higher ratings compared to offline orders.

8. Heatmap of Online Orders vs. Restaurant Type

pivot_table = dataframe.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Heatmap')
plt.xlabel('Online Order')
plt.ylabel('Listed In (Type)')
plt.show()

Conclusion: Cafes primarily receive online orders, whereas dining restaurants mostly accept offline orders.

Conclusion

This analysis provides insights into customer preferences and trends in the restaurant industry. Dining restaurants are the most popular, online orders receive higher ratings, and most couples prefer affordable restaurants. This information can help restaurant owners make data-driven decisions to improve their services.

How to Run the Project

Install required libraries if not already installed:

pip install numpy pandas matplotlib seaborn

Clone the repository and navigate to the project directory.

git clone <repository-url>
cd zomato-data-analysis

Run the Python script or execute the Jupyter Notebook to visualize the results.

Dataset

You can download the dataset from the following link: [Dataset Link]

License

This project is open-source and available under the MIT License.
