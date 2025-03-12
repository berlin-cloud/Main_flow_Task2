# Exploratory Data Analysis & Sales Performance Analysis

## Project Overview
This project is part of my **Data Analysis and Data Science with Python internship** at **Main Flow Services and Technologies Pvt. Ltd.** The goal of this project is to perform **Exploratory Data Analysis (EDA)** and **Sales Performance Analysis** on a given dataset to extract meaningful insights, detect patterns, and build a predictive model for sales forecasting.

## Dataset Information
- **Dataset Name**: `sales_data.csv`
- **Columns**:
  - Product
  - Region
  - Sales
  - Profit
  - Discount
  - Category
  - Date

## Steps Performed
### 1. Data Cleaning
- Handled missing values by filling them using appropriate strategies (mean, median, placeholders).
- Removed duplicate records to maintain data integrity.
- Detected and handled outliers using **Z-score** method.
- Converted the `Date` column into a **datetime** format for trend analysis.

### 2. Exploratory Data Analysis (EDA)
- Visualized **sales trends over time** using a **time series plot**.
- Created a **scatter plot** to analyze the relationship between **Profit and Discount**.
- Generated **bar charts** and **pie charts** to visualize **Sales by Region and Category**.
- Used **histograms** and **boxplots** to explore numerical distributions and detect anomalies.
- Created a **correlation heatmap** to understand relationships between key features.

### 3. Predictive Modeling
- Built a **Linear Regression Model** to predict `Sales` based on `Profit` and `Discount`.
- Split the dataset into training (80%) and testing (20%) sets.
- Evaluated the model using:
  - **R² Score** (Goodness of fit)
  - **Mean Squared Error (MSE)** (Prediction accuracy)

## Key Insights & Recommendations
- **Sales trends** show peak seasons, allowing for better stock management.
- **Higher discounts** often lead to **lower profits**, indicating a need for optimized discount strategies.
- **Certain regions/categories outperform others**, which can help focus marketing and sales efforts.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas` - Data manipulation & analysis
  - `numpy` - Numerical computing
  - `matplotlib` & `seaborn` - Data visualization
  - `sklearn` - Machine learning (Linear Regression, model evaluation)

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the Python script:
   ```bash
   python analysis.py
   ```

## Conclusion
This project provided valuable experience in data cleaning, exploratory analysis, visualization, and predictive modeling. The insights derived from this analysis can support better decision-making in sales and marketing strategies.

---

**Author:** Berlin Samvel Pandian S  
**Internship:** Main Flow Services and Technologies Pvt. Ltd.  
**LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/your-profile/)

