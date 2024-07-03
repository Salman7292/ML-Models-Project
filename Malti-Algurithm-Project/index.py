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

from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

with open("Malti-Algurithm-Project/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



selections = option_menu(
        menu_title=None,
        options=['Home', 'Display DataSet',"Data Visulization",'Inserting Data', 'Display Predication', 'How Model Perdict'],
        icons=['house-fill', 'bi-display-fill', "bi-bar-chart-line-fill",'bi-database-fill-up', 'bi-easel-fill', 'bi-gear'],
        menu_icon="cast",  # Optional: Change the menu icon
        default_index=0 ,  # Optional: Set the default selected option
        orientation='horizontal',
        styles={
        "container": {"padding": "5!important","background-color":"#0d6efd", "border-radius": ".0","font-color":"white","box-shadow":" 2px 2px 7px -2px rgba(36, 2, 2, 0.75)"},
        "icon": {"color": "white", "font-size": "23px"}, 
         "hr": {
         "color": "rgb(255, 255, 255)"
          },
       "nav-link": {"color":"white","font-size": "15px", "text-align": "left", "margin":"5px", "--hover-color": "blue","border":"None"},
        "nav-link-selected": {"background-color": "#81B622"},


# .menu .container-xxl[data-v-5af006b8] {
#     background-color: var(--secondary-background-color);
#     /* border-radius:.5rem; */
# }
}

    )
    
























hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style,unsafe_allow_html=True)
