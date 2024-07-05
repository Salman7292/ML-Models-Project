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


st.set_page_config(
        page_icon="Malti-Algurithm-Project/logo3.png",
        page_title="Data Insights Predictor | app",
        layout="wide"

    )

with open("Malti-Algurithm-Project/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



with st.sidebar:
    st.image("Malti-Algurithm-Project/logo3.png", use_column_width=True)


    # Adding a custom style with HTML and CSS
    st.markdown("""
        <style>
            .custom-text {
                font-size: 25px;
                font-weight: bold;
                text-align: center;
                color:#ffc107
            }
            .custom-text span {
                color: #04ECF0; /* Color for the word 'Insights' */
            }
        </style>
    """, unsafe_allow_html=True)

    # Displaying the subheader with the custom class
    st.markdown('<p class="custom-text">Data <span>Insights</span> Predictor</p>', unsafe_allow_html=True)



    # HTML and CSS for the button
    github_button_html = """
    <div style="text-align: center; margin-top: 50px;">
        <a class="button" href="https://github.com/Salman7292" target="_blank" rel="noopener noreferrer">Visit my GitHub</a>
    </div>

    <style>
        /* Button styles */
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #ffc107;
            color: black;
            text-decoration: none;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #000345;
            color: white;
            text-decoration: none; /* Remove underline on hover */
        }
    </style>
    """

    # Display the GitHub button in the app
    st.markdown(github_button_html, unsafe_allow_html=True)
    
    # Footer
    # Footer content
 # HTML and CSS for the centered footer
    footer_html = """
    <div style="background-color:#023047; padding:10px; text-align:center;margin-top: 10px;">
        <p style="font-size:20px; color:#ffffff;">Made with ❤️ by Salman Malik</p>
    </div>
    """

    # Display footer in the app
    st.markdown(footer_html, unsafe_allow_html=True)
    



# Function to plot model results
def plot_results(model_name, y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Values')
    plt.plot(predictions, label='Predicted Values', alpha=0.7)
    plt.title(f'{model_name} Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)



# Function to plot bar plots of model evaluation
def plot_bar(model_name, mse, rmse,color1):
    evaluation_df = pd.DataFrame({
        "Metric": ["Mean Squared Error", "Root Mean Squared Error"],
        "Value": [mse, rmse]
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Metric", y="Value", data=evaluation_df,color=color1)
    plt.title(f'{model_name} Model Evaluation')
    plt.xlabel('Metric')
    plt.ylabel('Value')

    st.pyplot(plt)





# Function to plot combined comparison of all models
def plot_combined_comparison(all_results_df):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    sns.barplot(x="Models", y="RMS/Error", data=all_results_df, palette="tab10", ax=ax1, label='RMS Error')
    ax1.set_ylabel('RMS Error')
    ax1.set_title('Comparison of All Models - RMS and MSE Error')
    
    ax2 = ax1.twinx()
    sns.barplot(x="Models", y="MSE/Error", data=all_results_df, palette="tab10", ax=ax2, label='MSE Error')
    ax2.set_ylabel('MSE Error')

    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    st.pyplot(fig)





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
        background-color: #fff;
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
        <h1 class="hero-heading">Empower Your Data <span>Insights with Predictive</span> Analytics</h1>
        <p class="hero-text">
        Unlock the power of data-driven decision-making with our intuitive Streamlit app. Upload your Regession Type dataset effortlessly, and watch as our app automatically analyzes attributes, applies sophisticated train-test splitting, and enables you to select from a range of powerful regression models. Whether you're forecasting trends, optimizing strategies, or making informed decisions, our app simplifies the complex, making predictive analytics accessible and actionable. Transform your data into insights today with our user-friendly, interactive tool.
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
    
    # submation_file=st.form("Uploding the file")
    DataSet_name=st.text_input(placeholder="DataSet Name",label="Insert Dataset Name")

    input=st.expander("Insert your Data Here")
    # File uploader widget
    uploaded_file = st.file_uploader(label="Upload CSV file", type=["csv"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        try:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            # Display the DataFrame

            st.title(f"{DataSet_name} DataSet Loaded")
            st.write(f"Shape {data.shape}")
            st.dataframe(data)
            columns_list=data.columns
            columns_list2=list()
            for column in columns_list:
                columns_list2.append(column)


            st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); box-shadow: 10px 4px 8px rgba(3, 225, 129, 0.2);" />
                """,
                unsafe_allow_html=True
            )               
                

            st.markdown(
            """
            <h1 style='text-align: center;'>Select The columns as Final Dataset</h1>
            """,
            unsafe_allow_html=True
            ) 

            option=st.multiselect(
                    label='Remove The unnecessary  Features',
                    options=columns_list2,
                    default=columns_list2
                )
                
            final_DataSet=data[option]
            st.subheader("Final DataSet")
            st.dataframe(final_DataSet)





            spliting_data=final_DataSet.columns

            columns_list3=list()

            for column in spliting_data:
                columns_list3.append(column)

            st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); box-shadow: 10px 4px 8px rgba(3, 225, 129, 0.2);" />
                """,
                unsafe_allow_html=True
            )


         
            st.markdown(
            """
            <h1 style='text-align: center;'>Defining input and Traget Features</h1>
            """,
            unsafe_allow_html=True
            ) 


            st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(to right, #ff7e5f, #feb47b);" />
                """,
                unsafe_allow_html=True
            )


            col1,col2=st.columns([1,1])

            with col1:
                st.subheader("Input Features")
                option2=st.multiselect(
                    label='select The Input Features',
                    options=columns_list3,
                    default=columns_list3
                )
                
                Input_Features=data[option2]
                
                st.dataframe(Input_Features)


            # Insert a vertical line between the two columns using HTML/CSS
                st.markdown(
                    """
                    <style>
                    .vertical-line {
                        border-left: 2px solid #3498db;
                        height: 100%;
                        position: absolute;
                        left: 50%;
                        transform: translateX(-50%);
                    }
                    </style>
                    <div class="vertical-line"></div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:
                st.subheader("Target features")
                option3=st.multiselect(
                    label='select The Target Features',
                    options=columns_list3,
                    default=columns_list3
                )
                
                Target_Features=data[option3]
                
                st.dataframe(Target_Features)
            # Horizontal line with gradient color
            st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); box-shadow: 10px 4px 8px rgba(3, 225, 129, 0.2);" />
                """,
                unsafe_allow_html=True
            )
          
            st.markdown(
            """
            <h1 style='text-align: center;'>Applying Train Test spliting</h1>
            """,
            unsafe_allow_html=True
            )  



            st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(to right, #ff7e5f, #feb47b);" />
                """,
                unsafe_allow_html=True
            )




            x_train,x_test,y_train,y_test=train_test_split(Input_Features,Target_Features,test_size=.20,random_state=101)
           

            st.markdown(
            """
            <h3 style='text-align: center;'> Data Which Were Used In a Training Of Model</h3>
            """,
            unsafe_allow_html=True
            )

            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )


            col3,col4=st.columns(2)
            with col3:
                st.subheader("X Train Data")
                st.write(f"Shape {x_train.shape}")
                st.dataframe(x_train)

            with col4:
                st.subheader("Y Train Data")
                st.write(f"Shape {y_train.shape}")
                st.dataframe(y_train)


            st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
            )
            st.markdown(
            """
            <h3 style='text-align: center;'> Data Which Were Used In a Testing Of Model</h3>
            """,
            unsafe_allow_html=True
            )

            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )


            col5,col6=st.columns(2)
            with col5:
                st.subheader("X Testing Data")
                st.write(f"Shape {x_test.shape}")
                st.dataframe(x_test)

            with col6:
                st.subheader("Y Testing Data")
                st.write(f"Shape {y_test.shape}")
                st.dataframe(y_test)

           
        

            st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
            )
            st.markdown(
            """
            <h1 style='text-align: center;'>Select a Models Which You Want To Train</h1>
            """,
            unsafe_allow_html=True
            )



            Model_list=["RandomForestRegressor","GradientBoostingRegressor","DecisionTreeRegressor","LinearRegression","All"]


            
            option4 = st.radio(
                label="Select a Model",
                options=["RandomForestRegressor Model","GradientBoostingRegressor Model","DecisionTreeRegressor Model","LinearRegression Model","All Models"],
                index=3,  # Setting the default selected index to 1 (Banana)
        
            )

            # with st.spinner("Traning A Model"):
            #     time.sleep(10)

            st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
            )
            
            
            st.markdown(
            """
            <h1 style='text-align: center;'>Model Detail</h1>
            """,
            unsafe_allow_html=True
            )

            if option4=='RandomForestRegressor Model':


                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>RandomForestRegressor Model Predictions</h3>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



                RandomForestRegressor_Model=RandomForestRegressor()

                RandomForestRegressor_Model.fit(x_train,y_train)

                Result1=RandomForestRegressor_Model.predict(x_test)
       
                
                RandomForestRegressor_Model_Predictions=pd.DataFrame(Result1)
                RandomForestRegressor_Model_Predictions.rename(columns={0:"Predicted Values"},inplace=True)
                RandomForestRegressor_Model_Predictions=RandomForestRegressor_Model_Predictions.reset_index()
                RandomForestRegressor_Model_Predictions.drop(columns='index',axis=1,inplace=True)



                actual_values=y_test
                actual_values=pd.DataFrame(actual_values)
                actual_values=actual_values.reset_index()
                actual_values.drop(columns='index',axis=1,inplace=True)

    
                combind_Result=pd.concat([actual_values,RandomForestRegressor_Model_Predictions],axis=1)
                st.table(combind_Result.head(20))

                plot_results("RandomForestRegressor",actual_values.head(50),RandomForestRegressor_Model_Predictions.head(50))

                

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>Model evaluation Of RandomForestRegressor</h3>
                """,
                unsafe_allow_html=True
                )

                MSE=mean_squared_error(y_test,Result1)
                RMSE=np.sqrt(MSE)

                
 

                RandomForestRegressor_Model_evaluation={
                    "Errors":["Mean Squred Error","Root Mean Squred Error"],
                    "Values":[MSE,RMSE]
                }
                RandomForestRegressor_Model_evaluation=pd.DataFrame(RandomForestRegressor_Model_evaluation)
                st.table(RandomForestRegressor_Model_evaluation)

                
                plot_bar("RandomForestRegressor",MSE,RMSE,"Blue")





                


            elif option4=='GradientBoostingRegressor Model':


                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>GradientBoostingRegressor Model Predictions</h3>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



                GradientBoostingRegressor_Model=GradientBoostingRegressor()

                GradientBoostingRegressor_Model.fit(x_train,y_train)

                Result2=GradientBoostingRegressor_Model.predict(x_test)
       
                
                GradientBoostingRegressor_Model=pd.DataFrame(Result2)
                GradientBoostingRegressor_Model.rename(columns={0:"Predicted Values"},inplace=True)
                GradientBoostingRegressor_Model_Predictions=GradientBoostingRegressor_Model.reset_index()
                GradientBoostingRegressor_Model_Predictions.drop(columns='index',axis=1,inplace=True)



                actual_values=y_test
                actual_values=pd.DataFrame(actual_values)
                actual_values=actual_values.reset_index()
                actual_values.drop(columns='index',axis=1,inplace=True)

    
                combind_Result=pd.concat([actual_values,GradientBoostingRegressor_Model_Predictions],axis=1)
                st.table(combind_Result.head(20))
                plot_results("GradientBoostingRegressor",actual_values.head(50),GradientBoostingRegressor_Model_Predictions.head(50))
                

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>Model evaluation Of GradientBoostingRegressor</h3>
                """,
                unsafe_allow_html=True
                )

                MSE=mean_squared_error(y_test,Result2)
                RMSE=np.sqrt(MSE)

                
 

                GradientBoostingRegressor_Model_evaluation={
                    "Errors":["Mean Squred Error","Root Mean Squred Error"],
                    "Values":[MSE,RMSE]
                }
                GradientBoostingRegressor_Model_evaluation=pd.DataFrame(GradientBoostingRegressor_Model_evaluation)
                st.table(GradientBoostingRegressor_Model_evaluation)
                
                plot_bar("GradientBoostingRegressor",MSE,RMSE,"Green")



            elif option4=='DecisionTreeRegressor Model':

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>DecisionTreeRegressor Model Predictions</h3>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



                DecisionTreeRegressor_Model=DecisionTreeRegressor()

                DecisionTreeRegressor_Model.fit(x_train,y_train)

                Result3=DecisionTreeRegressor_Model.predict(x_test)
       
                
                DecisionTreeRegressor_Model=pd.DataFrame(Result3)
                DecisionTreeRegressor_Model.rename(columns={0:"Predicted Values"},inplace=True)
                DecisionTreeRegressor_Model_Predictions=DecisionTreeRegressor_Model.reset_index()
                DecisionTreeRegressor_Model_Predictions.drop(columns='index',axis=1,inplace=True)



                actual_values=y_test
                actual_values=pd.DataFrame(actual_values)
                actual_values=actual_values.reset_index()
                actual_values.drop(columns='index',axis=1,inplace=True)

    
                combind_Result=pd.concat([actual_values,DecisionTreeRegressor_Model_Predictions],axis=1)
                st.table(combind_Result.head(20))

                plot_results("DecisionTreeRegressor",actual_values.head(50),DecisionTreeRegressor_Model_Predictions.head(50))
                            

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>Model evaluation Of DecisionTreeRegressor</h3>
                """,
                unsafe_allow_html=True
                )

                MSE=mean_squared_error(y_test,Result3)
                RMSE=np.sqrt(MSE)

                
 

                DecisionTreeRegressor_Model_evaluation={
                    "Errors":["Mean Squred Error","Root Mean Squred Error"],
                    "Values":[MSE,RMSE]
                }
                DecisionTreeRegressor_Model_evaluation=pd.DataFrame(DecisionTreeRegressor_Model_evaluation)
                st.table(DecisionTreeRegressor_Model_evaluation)
               
                plot_bar("DecisionTreeRegressor",MSE,RMSE,"Orange")  


            elif option4=='LinearRegression Model':
 
                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>LinearRegression Model Predictions</h3>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



                LinearRegression_Model=LinearRegression()

                LinearRegression_Model.fit(x_train,y_train)

                Result4=LinearRegression_Model.predict(x_test)
       
                
                LinearRegression_Model=pd.DataFrame(Result4)
                LinearRegression_Model.rename(columns={0:"Predicted Values"},inplace=True)
                LinearRegression_Model_Predictions=LinearRegression_Model.reset_index()
                LinearRegression_Model_Predictions.drop(columns='index',axis=1,inplace=True)



                actual_values=y_test
                actual_values=pd.DataFrame(actual_values)
                actual_values=actual_values.reset_index()
                actual_values.drop(columns='index',axis=1,inplace=True)

    
                combind_Result=pd.concat([actual_values,LinearRegression_Model_Predictions],axis=1)
                st.table(combind_Result.head(20))

                plot_results("LinearRegression",actual_values.head(50),LinearRegression_Model_Predictions.head(50))
                

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>Model evaluation Of LinearRegression</h3>
                """,
                unsafe_allow_html=True
                )

                MSE=mean_squared_error(y_test,Result4)
                RMSE=np.sqrt(MSE)

                
 

                LinearRegression_Model_evaluation={
                    "Errors":["Mean Squred Error","Root Mean Squred Error"],
                    "Values":[MSE,RMSE]
                }
                LinearRegression_Model_evaluation=pd.DataFrame(LinearRegression_Model_evaluation)
                st.table(LinearRegression_Model_evaluation)

                
                plot_bar("LinearRegression",MSE,RMSE,"gray")




 
            elif option4=='All Models':
                

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>RandomForestRegressor Model Predictions</h3>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



                RandomForestRegressor_Model=RandomForestRegressor()

                RandomForestRegressor_Model.fit(x_train,y_train)

                Result1=RandomForestRegressor_Model.predict(x_test)
       
                
                RandomForestRegressor_Model_Predictions=pd.DataFrame(Result1)
                RandomForestRegressor_Model_Predictions.rename(columns={0:"Predicted Values"},inplace=True)
                RandomForestRegressor_Model_Predictions=RandomForestRegressor_Model_Predictions.reset_index()
                RandomForestRegressor_Model_Predictions.drop(columns='index',axis=1,inplace=True)



                actual_values=y_test
                actual_values=pd.DataFrame(actual_values)
                actual_values=actual_values.reset_index()
                actual_values.drop(columns='index',axis=1,inplace=True)

    
                combind_Result=pd.concat([actual_values,RandomForestRegressor_Model_Predictions],axis=1)
                st.table(combind_Result.head(20))

                plot_results("RandomForestRegressor",actual_values.head(50),RandomForestRegressor_Model_Predictions.head(50))                

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>Model evaluation Of RandomForestRegressor</h3>
                """,
                unsafe_allow_html=True
                )

                MSE_RandomForestRegressor=mean_squared_error(y_test,Result1)
                RMSE_RandomForestRegressor=np.sqrt(MSE_RandomForestRegressor)

                
 

                RandomForestRegressor_Model_evaluation={
                    "Errors":["Mean Squred Error","Root Mean Squred Error"],
                    "Values":[MSE_RandomForestRegressor,RMSE_RandomForestRegressor]
                }
                RandomForestRegressor_Model_evaluation=pd.DataFrame(RandomForestRegressor_Model_evaluation)
                st.table(RandomForestRegressor_Model_evaluation)

                

                plot_bar("RandomForestRegressor",MSE_RandomForestRegressor,RMSE_RandomForestRegressor,"Blue")

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                






                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>GradientBoostingRegressor Model Predictions</h3>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



                GradientBoostingRegressor_Model=GradientBoostingRegressor()

                GradientBoostingRegressor_Model.fit(x_train,y_train)

                Result2=GradientBoostingRegressor_Model.predict(x_test)
       
                
                GradientBoostingRegressor_Model=pd.DataFrame(Result2)
                GradientBoostingRegressor_Model.rename(columns={0:"Predicted Values"},inplace=True)
                GradientBoostingRegressor_Model_Predictions=GradientBoostingRegressor_Model.reset_index()
                GradientBoostingRegressor_Model_Predictions.drop(columns='index',axis=1,inplace=True)



                actual_values=y_test
                actual_values=pd.DataFrame(actual_values)
                actual_values=actual_values.reset_index()
                actual_values.drop(columns='index',axis=1,inplace=True)

    
                combind_Result=pd.concat([actual_values,GradientBoostingRegressor_Model_Predictions],axis=1)
                st.table(combind_Result.head(20))

                
                plot_results("GradientBoostingRegressor",actual_values.head(50),GradientBoostingRegressor_Model_Predictions.head(50))               

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>Model evaluation Of GradientBoostingRegressor</h3>
                """,
                unsafe_allow_html=True
                )

                MSE_GradientBoostingRegressor=mean_squared_error(y_test,Result2)
                RMSE_GradientBoostingRegressor=np.sqrt(MSE_GradientBoostingRegressor)

                
 

                GradientBoostingRegressor_Model_evaluation={
                    "Errors":["Mean Squred Error","Root Mean Squred Error"],
                    "Values":[MSE_GradientBoostingRegressor,RMSE_GradientBoostingRegressor]
                }
                GradientBoostingRegressor_Model_evaluation=pd.DataFrame(GradientBoostingRegressor_Model_evaluation)
                st.table(GradientBoostingRegressor_Model_evaluation)

                plot_bar("GradientBoostingRegressor",MSE_GradientBoostingRegressor,RMSE_GradientBoostingRegressor,"Green")




                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )              


                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>DecisionTreeRegressor Model Predictions</h3>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



                DecisionTreeRegressor_Model=DecisionTreeRegressor()

                DecisionTreeRegressor_Model.fit(x_train,y_train)

                Result3=DecisionTreeRegressor_Model.predict(x_test)
       
                
                DecisionTreeRegressor_Model=pd.DataFrame(Result3)
                DecisionTreeRegressor_Model.rename(columns={0:"Predicted Values"},inplace=True)
                DecisionTreeRegressor_Model_Predictions=DecisionTreeRegressor_Model.reset_index()
                DecisionTreeRegressor_Model_Predictions.drop(columns='index',axis=1,inplace=True)



                actual_values=y_test
                actual_values=pd.DataFrame(actual_values)
                actual_values=actual_values.reset_index()
                actual_values.drop(columns='index',axis=1,inplace=True)

    
                combind_Result=pd.concat([actual_values,DecisionTreeRegressor_Model_Predictions],axis=1)
                st.table(combind_Result.head(20))
                plot_results("DecisionTreeRegressor",actual_values.head(50),DecisionTreeRegressor_Model_Predictions.head(50))               

                

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>Model evaluation Of DecisionTreeRegressor</h3>
                """,
                unsafe_allow_html=True
                )

                MSE_DecisionTreeRegressor=mean_squared_error(y_test,Result3)
                RMSE_DecisionTreeRegressor=np.sqrt(MSE_DecisionTreeRegressor)

                
 

                DecisionTreeRegressor_Model_evaluation={
                    "Errors":["Mean Squred Error","Root Mean Squred Error"],
                    "Values":[MSE_DecisionTreeRegressor,RMSE_DecisionTreeRegressor]
                }
                DecisionTreeRegressor_Model_evaluation=pd.DataFrame(DecisionTreeRegressor_Model_evaluation)
                st.table(DecisionTreeRegressor_Model_evaluation)

                plot_bar("DecisionTreeRegressor",MSE_DecisionTreeRegressor,RMSE_DecisionTreeRegressor,"Orange")  



                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>LinearRegression Model Predictions</h3>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



                LinearRegression_Model=LinearRegression()

                LinearRegression_Model.fit(x_train,y_train)

                Result4=LinearRegression_Model.predict(x_test)
       
                
                LinearRegression_Model=pd.DataFrame(Result4)
                LinearRegression_Model.rename(columns={0:"Predicted Values"},inplace=True)
                LinearRegression_Model_Predictions=LinearRegression_Model.reset_index()
                LinearRegression_Model_Predictions.drop(columns='index',axis=1,inplace=True)



                actual_values=y_test
                actual_values=pd.DataFrame(actual_values)
                actual_values=actual_values.reset_index()
                actual_values.drop(columns='index',axis=1,inplace=True)

    
                combind_Result=pd.concat([actual_values,LinearRegression_Model_Predictions],axis=1)
                st.table(combind_Result.head(20))

                plot_results("LinearRegression",actual_values.head(50),LinearRegression_Model_Predictions.head(50))                

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                    
                st.markdown(
                """
                <h3 style='text-align: center;'>Model evaluation Of LinearRegression</h3>
                """,
                unsafe_allow_html=True
                )

                MSE_LinearRegression=mean_squared_error(y_test,Result4)
                RMSE_LinearRegression=np.sqrt(MSE_LinearRegression)

                
 

                LinearRegression_Model_evaluation={
                    "Errors":["Mean Squred Error","Root Mean Squred Error"],
                    "Values":[MSE_LinearRegression,RMSE_LinearRegression]
                }
                LinearRegression_Model_evaluation=pd.DataFrame(LinearRegression_Model_evaluation)
                st.table(LinearRegression_Model_evaluation)

                plot_bar("LinearRegression",MSE_LinearRegression,RMSE_LinearRegression,"gray")               



                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                

                st.markdown(
                """
                <h1 style='text-align: center;'>All Models Detail</h1>
                """,
                unsafe_allow_html=True
                )

                all_Result={
                    "Models":["Linear Regression Model","Decision Tree Regressor Model","Random Forest Regressor Model", "Gradient Boosting Regressor Model"],
                    "RMS/Error":[RMSE_LinearRegression,RMSE_DecisionTreeRegressor,RMSE_RandomForestRegressor,RMSE_GradientBoostingRegressor],
                    "MSE/Error":[MSE_LinearRegression,MSE_DecisionTreeRegressor,MSE_RandomForestRegressor,MSE_GradientBoostingRegressor]
                }
                all_Result=pd.DataFrame(all_Result)
                all_Result=all_Result.sort_values(by="RMS/Error",ascending=True).reset_index()
                all_Result.drop(columns="index",axis=1,inplace=True)
                st.table(all_Result)

                # Plot the combined comparison of all models
                plot_combined_comparison(all_Result)



                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                # Sutibal_model=all_Result.loc[,"Models"]

                Sutibal_model = all_Result.loc[all_Result.index[0], 'Models'] 

                st.subheader(f"Most sutible Model For {DataSet_name} DataSet are {Sutibal_model}")

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )               

                








           
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
    else:
        st.error("No file uploaded yet!")

    # Load_DataSet_button = submation_file.form_submit_button("Load DataSet")



    

hide_streamlit_style = """
<style>
 MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style,unsafe_allow_html=True)

