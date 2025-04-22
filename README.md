# 📊 European Heating Demand Data Analysis

## 📑 Project Overview
This project performs exploratory data analysis and predictive modeling on the **When2Heat - European Heating Demand Dataset**, sourced from Kaggle. The dataset contains detailed records of hourly space and water heating demands across European countries, alongside various heat pump performance profiles.

---

## 📥 Dataset Source
The dataset used in this project can be downloaded from:

👉 [**When2Heat - European Heating Demand Dataset**](https://www.kaggle.com/datasets/matthewjansen/when2heat-european-heating-demand-dataset)

---

## 📂 How to Set Up the Data

1. **Download the dataset** from the Kaggle link above.
2. **Unzip the downloaded archive** into your project directory.
3. Update the CSV file paths in the notebook as required.  
   Example:
   ```python
   # Change this to match your file location
   df = pd.read_csv("path/to/your/csvfile.csv")

## 📦 Libraries and Tools Used

The following Python libraries were used throughout this analysis and modeling project:

---

### 📊 Data Analysis & Manipulation
```python
import pandas as pd
from tabulate import tabulate
from IPython.display import display, HTML
import numpy as np
📈 Data Visualization
python
Copy
Edit
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
📊 Statistical Analysis & Tests
python
Copy
Edit
from scipy import stats
from scipy.stats import linregress, f_oneway
import pymannkendall as mk
import statsmodels.api as sm
from statsmodels.formula.api import ols
🤖 Machine Learning & Modeling
python
Copy
Edit
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
✅ Running the Notebook
Ensure all the above libraries are installed in your Python environment before running the notebook.

You can install all required libraries at once using:

bash
Copy
Edit
pip install -r requirements.txt
Or, manually install them one by one via:

bash
Copy
Edit
pip install pandas seaborn matplotlib scikit-learn statsmodels plotly pymannkendall
📊 Output
The notebook will generate:

📈 Exploratory Data Analysis (EDA) plots

📊 Correlation heatmaps

📉 Time series demand trends

🤖 Predictive modeling performance metrics (R², MSE, MAE)

📊 Interactive visualizations using Plotly


