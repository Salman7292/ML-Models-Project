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



with st.sidebar:
    # st.image("Logo3.png", caption="Beautiful Landscape", use_column_width=True)
    st.title("Hello")







# Define the option menu for navigation
selections = option_menu(
    menu_title=None,  # No title for the menu
    options=['Home', 'Upload DataSet'],  # Options for the menu
    icons=['house-fill', 'bi-display-fill'],  # Icons for the options
    menu_icon="cast",  # Optional: Change the menu icon
    default_index=0,  # Optional: Set the default selected option
    orientation='horizontal',  # Set the menu orientation to horizontal
    styles={  # Define custom styles for the menu
        "container": {
            "padding": "5!important",
            "background-color": "#0d6efd",  # Background color of the menu
            "border-radius": ".0",
            "font-color": "white",
            "box-shadow": "2px 2px 7px -2px rgba(36, 2, 2, 0.75)"
        },
        "icon": {"color": "white", "font-size": "23px"},  # Style for the icons
        "hr": {"color": "rgb(255, 255, 255)"},  # Style for the horizontal line
        "nav-link": {
            "color": "white",
            "font-size": "13px",
            "text-align": "left",
            "margin": "5px",
            "--hover-color": "blue",
            "border": "None"
        },
        "nav-link-selected": {"background-color": "#ffc107"},  # Style for the selected option
    }
)

# Check the selected option from the menu
if selections == 'Home':
    
    # If 'Home' is selected, include Bootstrap CSS and JS
    st.markdown("""
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.bundle.min.js"></script>
    """, unsafe_allow_html=True)

    # Define HTML and CSS for the hero section
    code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Streamlit App Description</title>
    <style>
        .hero-section {
        background-color: #f0f0f0; /* Set your background color */
        padding: 80px 20px; /* Adjust padding as needed */
        text-align: center;
        font-family: Arial, sans-serif;
        }
        .hero-heading {
        font-size: 2.5rem;
        margin-bottom: 20px;
        color: #333; /* Set your text color */
        font-family: 'Roboto', sans-serif; /* Use Roboto font specifically for heading */
        font-weight: 700; /* Use Roboto's bold variant */
        }
        .hero-text {
        font-size: 1.2rem;
        line-height: 1.6;
        color: #666; /* Set your text color */
        max-width: 900px;
        margin: 0 auto;
        }
    </style>
    </head>
    <body>

    <section class="hero-section">
    <div class="container">
        <h1 class="hero-heading">Empower Your Data Insights with Predictive Analytics</h1>
        <p class="hero-text">
        Unlock the power of data-driven decision-making with our intuitive Streamlit app. Upload your dataset effortlessly, and watch as our app automatically analyzes attributes, applies sophisticated train-test splitting, and enables you to select from a range of powerful regression models. Whether you're forecasting trends, optimizing strategies, or making informed decisions, our app simplifies the complex, making predictive analytics accessible and actionable. Transform your data into insights today with our user-friendly, interactive tool.
        </p>
    </div>
    </section>

    </body>
    </html>
    """
    # Display the hero section using markdown
    st.markdown(code, unsafe_allow_html=True)

elif selections == 'Upload DataSet':
    # If 'Upload DataSet' is selected, display the title
    
    submation_file=st.form("Uploding the file")
    DataSet_name=submation_file.text_input(placeholder="DataSet Name",label="Insert Dataset Name")

    input=submation_file.expander("Insert your Data Here")
    uploaded_file = input.file_uploader("Upload CSV file", type=["csv"])

    Load_DataSet_button = submation_file.form_submit_button("Load DataSet")


    if Load_DataSet_button:
        data=pd.read_csv(uploaded_file)
        st.table(data.head())



























hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style,unsafe_allow_html=True)
