import streamlit as st
# Importing the pandas library for data manipulation and analysis
import pandas as pd

# Importing the seaborn library for data visualization
import seaborn as sns
import numpy as np

# Importing the matplotlib library for creating static, animated, and interactive visualizations
import matplotlib.pyplot as plt

# Importing the train_test_split function from scikit-learn to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Importing the mean_squared_error function from scikit-learn to evaluate the performance of regression models
from sklearn.metrics import mean_squared_error

# Importing the LinearRegression class from scikit-learn to implement linear regression models
from sklearn.linear_model import LinearRegression

# Importing the DecisionTreeRegressor class from scikit-learn to implement decision tree regression models
from sklearn.tree import DecisionTreeRegressor

# Importing the RandomForestRegressor class from scikit-learn to implement random forest regression models
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor


with open("Malti-Algurithm-Project/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Malti Model opertion on Dataset')



