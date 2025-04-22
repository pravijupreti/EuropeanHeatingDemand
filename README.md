# 📊 European Heating Demand Data Analysis

## 📑 Project Overview
This project performs exploratory data analysis and predictive modeling on the **When2Heat - European Heating Demand Dataset**, sourced from Kaggle. The dataset contains detailed records of hourly space and water heating demands across European countries, alongside various heat pump performance profiles.

---

## 📥 Dataset Source
The dataset used in this project can be downloaded from:

👉 [When2Heat - European Heating Demand Dataset](https://www.kaggle.com/datasets/matthewjansen/when2heat-european-heating-demand-dataset)

---

## 📂 How to Set Up the Data

1. Download the dataset from the Kaggle link above.
2. Unzip the downloaded archive into your project directory.
3. Update the CSV file paths in the notebook as required:
   ```python
   # Example: Change this to match your file location
   df = pd.read_csv("path/to/your/csvfile.csv")
📦 Libraries and Tools Used
The following Python libraries were used throughout this analysis and modeling project:

📊 Data Analysis & Manipulation
pandas

tabulate

IPython.display

numpy

📈 Data Visualization
matplotlib

seaborn

plotly.express

plotly.graph_objects

mpl_toolkits.mplot3d

matplotlib.animation

📊 Statistical Analysis & Tests
scipy

pymannkendall

statsmodels

🤖 Machine Learning & Modeling
scikit-learn

train_test_split

GridSearchCV

cross_val_score

LinearRegression

RandomForestRegressor

LabelEncoder

OneHotEncoder

StandardScaler

ColumnTransformer

Pipeline

SimpleImputer

Metrics (mean_absolute_error, mean_squared_error, r2_score, confusion_matrix)
