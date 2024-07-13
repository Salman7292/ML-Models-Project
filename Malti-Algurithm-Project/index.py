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


from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,StandardScaler,MinMaxScaler





# classification models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.metrics import accuracy_score



from sklearn.metrics import confusion_matrix



import time 




st.set_page_config(
        page_icon="Malti-Algurithm-Project/logo3.png",
        page_title="Data Insights Predictor | app",
        layout="wide",
        
        
            )

with open("Malti-Algurithm-Project/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



flag1=0


with st.sidebar:
    st.image("Malti-Algurithm-Project/logo3.png", use_column_width=True)


    # Adding a custom style with HTML and CSS
    st.markdown("""
        <style>
            .custom-text {
                font-size: 28px;
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

# funcation to draw boxplot for outliers

def create_boxplots(dataframe, columns):
    melted_df = dataframe.melt(value_vars=columns, var_name='Variable', value_name='Value')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Variable', y='Value', data=melted_df, palette='Set1')
    # plt.title('Boxplots of Selected Columns', fontsize=14)  # Adjust title font size
    plt.xlabel('Features', fontsize=12)  # Adjust x-axis label font size
    plt.ylabel('Values', fontsize=12)  # Adjust y-axis label font size
    plt.xticks(fontsize=6)  # Adjust x-axis tick labels font size
    plt.yticks(fontsize=10)  # Adjust y-axis tick labels font size
    # plt.grid(True)
    st.pyplot(plt)



# outlier Removal from the dataset using percentile

def Outliers_Removal(data, columns):
    indexlist = set()
    for column in columns:
        q1 = data[column].quantile(0.25)  # 1st quartile (25th percentile)
        q2 = data[column].quantile(0.50)  # Median (50th percentile)
        q3 = data[column].quantile(0.75)  # 3rd quartile (75th percentile)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr  # Lower outlier bound
        upper_bound = q3 + 1.5 * iqr  # Upper outlier bound

        # Identify outliers
        Outliers_OF_column = data[column].loc[(data[column] < lower_bound) | (data[column] > upper_bound)]
        
        # Add the indices of outliers to the set
        indexlist.update(Outliers_OF_column.index)

    # Drop rows with outliers from the DataFrame
    final_DataSet = data.drop(indexlist)
    
    return final_DataSet


# Function to get user-defined order for unique values in each column
def get_ordered_unique_values(df,column_list):

    ordered_values = {}

    for column in column_list:
        unique_values = df[column].unique().tolist()
        options = st.multiselect(
            label=f'Select and order the categories for {column} Feature',
            options=unique_values,
            default=None
        )

        # Reverse the order for demonstration purposes (remove if not needed)
        options = list(reversed(options))

        ordered_values[column] = options

    return ordered_values



def creating_cunfusion_Matrix(cm,color,model):
    # Use matplotlib to plot the confusion matrix
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax)
    plt.xlabel('Predicted',fontsize=8)
    plt.ylabel('Actual',fontsize=8)
    plt.title(F'Confusion Matrix for {model}',fontsize=8)

    # Display the plot in the Streamlit app
    st.pyplot(fig)   




# Define the option menu for navigation
selections = option_menu(
    menu_title=None,  # No title for the menu
    options=['Home', "DataSet Preprocessing",'Regression Models',"Classification Models"],  # Options for the menu
    icons=['house-fill', "bi-magic",'bi-graph-up',"bi-calculator"],  # Icons for the options
    menu_icon="cast",  # Optional: Change the menu icon
    default_index=0,  # Optional: Set the default selected option
    orientation='horizontal',  # Set the menu orientation to horizontal
    styles={  # Define custom styles for the menu
        "container": {
            "padding": "5px 23px",
            "background-color": "#0d6efd",  # Background color (dark grey)
            "border-radius": "8px",
            "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.25)"
        },
        "icon": {"color": "#f9fafb", "font-size": "18px"},  # Style for the icons (light grey)
        "hr": {"color": "#0d6dfdbe"},  # Style for the horizontal line (very light grey)
        "nav-link": {
            "color": "#f9fafb",  # Light grey text color
            "font-size": "14px",
            "text-align": "center",
            "margin": "0 10px",  # Adds space between the buttons
            "--hover-color": "#0761e97e",  # Slightly lighter grey for hover
            "padding": "10px 10px",
            "border-radius": "16px"
        },
        "nav-link-selected": {"background-color": "#ffc107"},  # Green background for selected option
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
        background-color: #ffffff /* Set your background color */
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

elif selections == 'Regression Models':
    st.markdown(
                                    """
                                    <h1 style='text-align: center;color:orange'>Wellcome to Regression Models</h1>
                                    """,
                                    unsafe_allow_html=True
                                    ) 

    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )     


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
            data=data.dropna()
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
                index=None,  # Setting the default selected index to 1 (Banana)
        
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



    











# DataSet Preprocessing

elif selections == 'DataSet Preprocessing':
    st.markdown(
                                    """
                                    <h1 style='text-align: center;color:orange'>Here Preprocess Your DataSet</h1>
                                    """,
                                    unsafe_allow_html=True
                                    ) 

    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )     

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
            data=data.dropna()
            st.success("File uploaded successfully!")
            # Display the DataFrame

            st.title(f"{DataSet_name} DataSet Loaded")
            st.markdown("""<hr style="border: none; height: 2px;width: 60%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
            )
            st.write(f"Shape {data.shape}")
            st.dataframe(data)

            st.markdown(
                        """
                        <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                        """,
                        unsafe_allow_html=True
                            )
            
            st.markdown(
                        """
                        <h1 style='text-align: center;'>Preprocessing the DataSet</h1>
                        """,
                        unsafe_allow_html=True
                        )
            st.markdown(
                        """
                        <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                        """,
                        unsafe_allow_html=True
                            )
            


# Basic information About the DataSet

            st.markdown(
                        """
                        <h3 style='text-align: center;'>Basic information About the DataSet</h3>
                        """,
                        unsafe_allow_html=True
                        )
            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )




            tab1,tab2,tab3=st.tabs(["DataSet Shape","DataSet dataTypes","statistics Of DataSet"])


# dataSet shape

            with tab1:
                st.subheader("    ")
                st.subheader("DataSet Shape")
                st.write(f"Total Number of Rows : {data.shape[0]}")
                st.write(f"Total Number of Columns : {data.shape[1]}")



# DataSet dataTypes

            with tab2:
                st.subheader("    ")
                st.subheader("DataSet dataTypes")
                

                
                full_info=dict()
                for i in data.columns:
                    full_info[i]=data[i].dtypes
                
                datatypes_info=pd.DataFrame(list(full_info.items()), columns=['Column Name', 'Data Type'])
                st.table(datatypes_info)
                st.subheader("    ")


# statistics Of DataSet

            with tab3:

                st.subheader(" ")
                st.subheader("statistics Of DataSet")
                paragraph = """
                                <p>The statistics of a dataset are typically 
                                represent a summary of key metrics that provide insights
                                into the data's distribution and characteristics. 
                                These metrics include the count of non-null entries, mean (average), standard deviation (which measures the spread of the data), minimum and maximum values, and the 25th, 50th (median), and 75th percentiles (quartiles). These statistical summaries help understand the central tendency, variability, and overall distribution of the data, which are crucial for data analysis and interpretation.</p>
                                """

            # Print the paragraph in the Streamlit app
                st.markdown(paragraph, unsafe_allow_html=True)
                st.dataframe(data.describe())

















           

            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
            )


            # remove the columns
                   
            

            st.markdown(
                        """
                        <h1 style='text-align: center;'>Remove Irrelevant Columns</h1>
                        """,
                        unsafe_allow_html=True
                        )
            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )


            
            columns_list=data.columns
            columns_list2=list()
            for column in columns_list:
                columns_list2.append(column)
            

            option=st.multiselect(
                    label='Remove The unnecessary  Features',
                    options=columns_list2,
                    default=columns_list2
                )
                
            final_DataSet=data[option]
            st.markdown(
                        """
                        <h4 style='text-align: center;'>Now Your DataSet are Look Like This</h4>
                        """,
                        unsafe_allow_html=True
                        )   
            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )
        
         
            st.dataframe(final_DataSet.head(10))


            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )


# removeing the dupliacted rows

            st.markdown(
                        """
                        <h4 style='text-align: center;'>Finding Duplicated Rows</h4>
                        """,
                        unsafe_allow_html=True
                        )  
            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )

            duplicate_rows=final_DataSet[final_DataSet.duplicated()]
            if duplicate_rows.shape[0]==0:
                st.success("There is No Dupliacted Rows 🤷")






            else:
                st.dataframe(duplicate_rows)
                index_list=list()
                for i in duplicate_rows.index:
                    index_list.append(i)
                
                final_DataSet.drop(index_list,axis=0,inplace=True)

                with st.spinner("Removing The Duplicted Rows.."):
                    time.sleep(1)
                
                st.success("Duplicted rows are Remove From the DataSet")

                st.markdown(
                        """
                        <h4 style='text-align: center;'>After Removing Dupulicted rows Your DataSet Are look like this</h4>
                        """,
                        unsafe_allow_html=True
                        ) 
                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )
               
                st.dataframe(final_DataSet)


                st.markdown(
                """
                <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )
                
            




# Detecting The Outliers in DataSet
            st.markdown(
                        """
                        <h1 style='text-align: center;'>Detecting and Removing The Outliers in DataSet</h1>
                        """,
                        unsafe_allow_html=True
                        ) 
            
            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )
            



            # selecting columns for outliers detection 
            full_info=dict()
            for i in final_DataSet.columns:
                full_info[i]=data[i].dtypes
                
 
            

            columns_selction_for_outliers=list()

            for key in full_info:
                if full_info[key]!='object':
                    columns_selction_for_outliers.append(key)
            

            st.markdown(
                        """
                        <h3 style='text-align: center;'>Detecting Outliers </h3>
                        """,
                        unsafe_allow_html=True
                        ) 
            
            create_boxplots(final_DataSet,columns_selction_for_outliers)


            st.markdown(
                        """
                        <h3 style='text-align: center;'>See with Focuse </h3>
                        """,
                        unsafe_allow_html=True
                        )            
            # Column selection
            selected_columns = st.multiselect('Select columns for boxplot To See With a Focus:',
                                              options= columns_selction_for_outliers,
                                              default=None
                                              
                                              )

            # Display boxplot in tabs if columns are selected
            if selected_columns:
                tabs = st.tabs(selected_columns)
                for tab, column in zip(tabs, selected_columns):
                    with tab:
                        create_boxplots(final_DataSet, [column])
            else:
                st.write("Please select at least one column.")           


            # Removing the Ouliers
            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                        """
                        <h3 style='text-align: center;'>Removing Outliers </h3>
                        """,
                        unsafe_allow_html=True
                        )
            



            # # first try to remove outliers
            cleaned_data1 = Outliers_Removal(final_DataSet, columns_selction_for_outliers)


            # # Second try to remove outliers
            cleaned_data2 = Outliers_Removal(cleaned_data1, columns_selction_for_outliers)

            # # last try to remove outliers
            cleaned_data5 = Outliers_Removal(cleaned_data2, columns_selction_for_outliers)



# show final boxplot 
            create_boxplots(cleaned_data5,columns_selction_for_outliers)     

     
            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )

            # Column selection
            selected_columns = st.multiselect(label='Select columns for boxplot ',
                                              options= columns_selction_for_outliers,
                                              default=None
                                              
                                              )

            # Display boxplot in tabs if columns are selected
            if selected_columns:
                tabs = st.tabs(selected_columns)
                for tab, column in zip(tabs, selected_columns):
                    with tab:
                        create_boxplots(cleaned_data5, [column])





# Display your Clean DataSet


            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )



            st.markdown(
                        """
                        <h4 style='text-align: center;'>After Removing Outliers  Your DataSet Are look like this</h4>
                        """,
                        unsafe_allow_html=True
                        ) 
            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )
            st.subheader("DataSet Shape")
            st.write(f"Total Number of Rows : {cleaned_data5.shape[0]}")
            st.write(f"Total Number of Columns : {cleaned_data5.shape[1]}")
            st.dataframe(cleaned_data5)

# Here is the leatest dataSet on the name of "cleaned_data5"

            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )
















# Extrcating the Categorical Data
            st.markdown(
                        """
                        <h3 style='text-align: center;'>Extracting The Categorical Features in DataSet</h3>
                        """,
                        unsafe_allow_html=True
                        ) 

            st.markdown(
                """
                <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )


            full_info_Categorical_Features=dict()
            for i in cleaned_data5.columns:
                full_info_Categorical_Features[i]=cleaned_data5[i].dtypes
                
 


            columns_selction_for_Encoding_and_Decoding=list()

            for key in full_info_Categorical_Features:
                if full_info[key]=='object':
                    columns_selction_for_Encoding_and_Decoding.append(key)    


            Categorical_Features_dataSet=cleaned_data5[columns_selction_for_Encoding_and_Decoding]
            
            if Categorical_Features_dataSet.shape[1]==0:
                flag1=1
                st.success("There is No Categorical Features In This DataSet 🤷")
                

            else:
                st.dataframe(Categorical_Features_dataSet)     

                    
            st.markdown(
                    """
                    <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )  












            # flage if there is no Catgrical data than do not show the other code  
                            
            if flag1==0:
                pass
 
                # Key Concepts About The Nominal And ordinal Features
                st.markdown(
                            """
                            <h2 style='text-align: center;'>Key Concepts About The Nominal And ordinal Features</h2>
                            """,
                            unsafe_allow_html=True
                            ) 

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )

                Right_Move_Tab,Nominal_tab,ordinal_tab=st.tabs(['➡️','Nominal Features','ordinal Features'])


                with Right_Move_Tab:
                    st.title("Know About Key Concepts ➡️")




    # nominal data explination
                
                with Nominal_tab:
        

                    st.title("What are Nominal Features?")
                    st.markdown("""
                    **Explanation:** Nominal features represent categorical data that do not have any intrinsic order or ranking among the categories. Each category is distinct and mutually exclusive, and the order in which categories are presented is arbitrary. These features are qualitative in nature and are used to label different groups or types within the data.
                    """)
                    st.markdown("**Example:**")
                    st.markdown("""
                    - **Gender**: Categories include male, female, and other.
                    - **Blood Type**: Categories include A, B, AB, and O.
                    - **Color**: Categories include red, blue, green, and yellow.
                    - **Type of Pet**: Categories include dog, cat, bird, and fish.
                    """)
                    st.markdown("**Which Technique is Used for Encoding/Labeling?**")
                    st.markdown("""
                    - **One-Hot Encoding**: This technique creates binary columns for each category, where a value of 1 indicates the presence of a category and 0 indicates its absence. It is useful because it avoids implying any ordinal relationship between categories.
                        - **Example**: For the feature "Color" with categories red, blue, and green, one-hot encoding would create three new binary features: Color_red, Color_blue, and Color_green.
                    - **get_dummies**: The get_dummies function in Pandas is used to convert categorical variables into numerical (dummy/indicator) variables. It creates new binary columns for each category in a categorical variable, where each column represents one category and contains binary values (0 or 1) indicating the presence or absence of that category in the original data
                        - **Example**:Imagine you have a dataset with a column that represents different types of fruits: apples, oranges, and bananas. Using get_dummies on this column would create new columns for each fruit type. Each row in these new columns would have a value of 1 if that row corresponds to that fruit type, and 0 otherwise.
                                
                                """)
                    
                    # Displaying code ans their output
                    
                    get_dummies='''
    import pandas as pd

    # Sample data
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Fruit': ['Apple', 'Orange', 'Banana', 'Apple', 'Banana']
    }

    df = pd.DataFrame(data)

    # Applying get_dummies to encode the 'Fruit' column
    encoded_df = pd.get_dummies(df, columns=['Fruit'])

    # Display the original and encoded DataFrame
    print("Original DataFrame:")
    print(df)

    print("\nEncoded DataFrame using get_dummies:")
    print(encoded_df)



    '''

                    output_code='''
    Original DataFrame:
    ID   Fruit
    0   1   Apple
    1   2  Orange
    2   3  Banana
    3   4   Apple
    4   5  Banana

    Encoded DataFrame using get_dummies:
    ID  Fruit_Apple  Fruit_Banana  Fruit_Orange
    0   1            1             0             0
    1   2            0             0             1
    2   3            0             1             0
    3   4            1             0             0
    4   5            0             1             0


    '''
                    st.subheader("get_dummies Code")
                    st.code(get_dummies,language='python')

                    st.subheader("Output")
                    st.code(output_code,language='yaml')   




                    One_Hot_Encoding='''
    import streamlit as st
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    # Sample data
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Fruit': ['Apple', 'Orange', 'Banana', 'Apple', 'Banana']
    }

    df = pd.DataFrame(data)

    # Creating an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse=False)

    # Encoding 'Fruit' column
    encoded_data = encoder.fit_transform(df[['Fruit']])

    # Creating a new DataFrame with encoded columns
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(['Fruit']))


    print(encoded_df)


    '''

                    output_code='''
    Encoded DataFrame using OneHotEncoder:
    ID  Fruit_Apple  Fruit_Banana  Fruit_Orange
    0   1            1             0             0
    1   2            0             0             1
    2   3            0             1             0
    3   4            1             0             0
    4   5            0             1             0

    '''
                    st.subheader("OneHotEncoder code")
                    st.code(One_Hot_Encoding,language='python')

                    st.subheader("Output")
                    st.code(output_code,language='yaml')   
















    # ordibal data explination


                with ordinal_tab:
                    st.title("What are Ordinal Features?")
                    st.markdown("""
                    **Explanation:** Ordinal features represent categorical data that have a meaningful order or ranking among the categories. The order of the categories indicates relative positions, but the differences between the categories are not necessarily equal or meaningful. These features are also qualitative but have an added layer of order.
                    """)
                    st.markdown("**Example:**")
                    st.markdown("""
                    - **Education Level**: Categories include high school, bachelor’s degree, master’s degree, and PhD.
                    - **Customer Satisfaction**: Categories include unsatisfied, neutral, satisfied, and very satisfied.
                    - **Socioeconomic Status**: Categories include low, middle, and high.
                    - **Movie Rating**: Categories include 1 star, 2 stars, 3 stars, 4 stars, and 5 stars.
                    """)
                    st.markdown("**Which Technique is Used for Encoding/Labeling?**")
                    st.markdown("""
                    - **Lable Encoding**: This technique assigns a unique integer to each category, preserving the ordinal relationship. The integer values reflect the order of the categories.
                        - **Example**: For the feature "Education Level" with categories high school, bachelor’s degree, master’s degree, and PhD, integer encoding might assign 1 to high school, 2 to bachelor’s degree, 3 to master’s degree, and 4 to PhD.
                
                        - High school: 1
                        - Bachelor’s degree: 2
                        - Master’s degree: 3
                        - PhD: 4
                    - **Ordinal Encoding**: This is similar to integer encoding but specifically ensures the ordinal relationship is maintained and understood by the machine learning algorithm.
                        - **Example**: For the feature "Customer Satisfaction" with categories unsatisfied, neutral, satisfied, and very satisfied, ordinal encoding would assign 1 to unsatisfied, 2 to neutral, 3 to satisfied, and 4 to very satisfied.
                    """)



                    Nominal_code='''
                        from sklearn.preprocessing import LabelEncoder

                        # Sample data
                        education_levels = ["high school", "bachelor’s degree", "master’s degree", "PhD"]

                        # Create a label encoder instance
                        label_encoder = LabelEncoder()

                        # Fit the label encoder to the data
                        label_encoder.fit(education_levels)

                        # Transform the data into numerical labels
                        encoded_labels = label_encoder.transform(education_levels)

                        # Display the encoded labels
                        print(f"Encoded labels: {encoded_labels}")


    '''
                    st.subheader("Lable Encoding Code")
                    st.code(Nominal_code,language='python')

                    # Define custom CSS for code block
                    css = '''
                    <style>
                    .st-emotion-cache-1hskohh {
                        margin: 0px;
                        padding-right: 2.75rem;
                        color: rgb(49, 51, 63);
                        border-radius: 0.5rem;
                        background: #0000001a;
                    }
                                    </style>
                    '''

                    # Render code block with custom CSS
                    st.markdown(css, unsafe_allow_html=True)

                    st.subheader("Ordinal Encoding Code")

                    ordinal_code='''
                        from sklearn.preprocessing import OrdinalEncoder
                        import numpy as np

                        # Define the categories with explicit order
                        education_levels = np.array(["high school", "bachelor’s degree", "master’s degree", "PhD"]).reshape(-1, 1)
                        categories = [["high school", "bachelor’s degree", "master’s degree", "PhD"]]

                        # Create an ordinal encoder instance with specified categories
                        ordinal_encoder = OrdinalEncoder(categories=categories)

                        # Fit and transform the data
                        encoded_labels = ordinal_encoder.fit_transform(education_levels)

                        # Display the encoded labels
                        print(f"Encoded labels: {encoded_labels.ravel()}")


    '''
                    st.code(ordinal_code,language='python')


                



                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )


# if there is Categorical_Features in dataset than show below code other wise do not show

            

                st.markdown(
                            """
                            <h3 style='text-align: center;'>Selecting Ordinal Features</h3>
                            """,
                            unsafe_allow_html=True
                            )
                        


                selected_columns = st.multiselect(label='Select Ordinal Features ',
                                                options= Categorical_Features_dataSet.columns,
                                                default=None
                                                
                                                )
                
                # Display boxplot in tabs if columns are selected
                if selected_columns:
                    tabs = st.tabs(selected_columns)
                    for tab, column in zip(tabs, selected_columns):
                        with tab:
                            st.subheader(f"Unique Values in {column}")

                            st.dataframe(Categorical_Features_dataSet[column].unique())





                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                            """
                            <h3 style='text-align: center;'>Selecting the Encoder For Ordinal Features</h3>
                            """,
                            unsafe_allow_html=True
                            )




                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )
                # Options for the radio button
                options = ["LabelEncoder", "OrdinalEncoder","No Need"]

                # Creating the radio button
                selected_encoder = st.radio(
                    label="Select an Encoder:",
                    options=options,
                    index=2 ,# Preselecting "Option 2"
                    format_func=lambda x: f"⏩ {x}",  # Adding an arrow before each option
    
                )


                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )

# lable encoding

                if selected_encoder=='LabelEncoder':


                    
 
                    st.markdown(
                            """
                            <h3 style='text-align: center;'>LabelEncoder Are only Works on Target column</h3>
                            """,
                            unsafe_allow_html=True
                            )                   
                    
                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )

                    selected_Target = st.radio(
                        label="Select an Target Varible",
                        options=Categorical_Features_dataSet.columns,
                        index=None ,# Preselecting "Option 2"
                        format_func=lambda x: f"➡️ {x}",  # Adding an arrow before each option
        
                    )

                    # Create a label encoder instance
                    label_encoder_for_Target = LabelEncoder()

                    # Fit the label encoder to the data
                    Target_encoded=label_encoder_for_Target.fit_transform(Categorical_Features_dataSet[selected_Target])



                    Target_encoded=pd.DataFrame(Target_encoded)

                    Target_encoded=Target_encoded.rename(columns={0:selected_Target})


 



                    New_clean_data=cleaned_data5.drop(columns=selected_Target,axis=1)

                    New_clean_data = New_clean_data.reset_index(drop=True)

                    # st.subheader("New_clean_data")
                    # st.write(New_clean_data.shape)


                    # st.dataframe(New_clean_data)




                    New_clean_data=pd.concat([New_clean_data,Target_encoded],axis=1)
                    # Resetting index (not typically necessary)
                    New_clean_data = New_clean_data.reset_index(drop=True)



                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )

                    
                    st.markdown(
                            """
                            <h3 style='text-align: center;'>Lable DataSet</h3>
                            """,
                            unsafe_allow_html=True
                            )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )

                    st.dataframe(New_clean_data)


                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )



# ordinal encoding

                elif selected_encoder=='OrdinalEncoder':


                    st.markdown(
                            """
                            <h3 style='text-align: center;'>Select and order the categories for Each Feature</h3>
                            """,
                            unsafe_allow_html=True
                            )
                    
                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )
                    # st.write(selected_columns)

                    # Get ordered unique values for all categorical columns
                    ordered_values = get_ordered_unique_values(Categorical_Features_dataSet,selected_columns)



                    # Preparing the categories in the format required by OrdinalEncoder
                    categories = []
                    columns_to_encode = []

                    for col in Categorical_Features_dataSet.columns:
                        if col in ordered_values:
                            categories.append(ordered_values[col])
                            columns_to_encode.append(col)

                    # Applying OrdinalEncoder with custom categories
                    encoder = OrdinalEncoder(categories=categories)
                    encoded_data = encoder.fit_transform(Categorical_Features_dataSet[columns_to_encode])

                    # Convert the encoded data back to a DataFrame
                    encoded_data = pd.DataFrame(encoded_data, columns=columns_to_encode)



                    encoded_data = encoded_data.reset_index(drop=True)

                    # st.subheader("encoded")
                    # st.write(encoded_data.shape)


                    # st.dataframe(encoded_data)






                    



                    New_clean_data=cleaned_data5.drop(columns=selected_columns,axis=1)

                    New_clean_data = New_clean_data.reset_index(drop=True)

                    # st.subheader("New_clean_data")
                    # st.write(New_clean_data.shape)


                    # st.dataframe(New_clean_data)



                    New_clean_data=pd.concat([New_clean_data,encoded_data],axis=1)
                    # Resetting index (not typically necessary)
                    New_clean_data = New_clean_data.reset_index(drop=True)



                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )


   
                    st.markdown(
                            """
                            <h3 style='text-align: center;'>Encoded DataSet</h3>
                            """,
                            unsafe_allow_html=True
                            )
                    
                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )
   

                    st.dataframe(New_clean_data)









                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )



                elif selected_encoder=='No Need':
                    New_clean_data=cleaned_data5

                


   
                st.markdown(
                                """
                                <h3 style='text-align: center;'>Selecting Nominal Features</h3>
                                """,
                                unsafe_allow_html=True
                                )
                        
                st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )
                


                full_info_Categorical_Features=dict()
                for i in New_clean_data.columns:
                    full_info_Categorical_Features[i]=New_clean_data[i].dtypes
                    
    
                # st.write(full_info_Categorical_Features)

                columns_selction_for_Nominal_Encoding=list()

                for key in full_info_Categorical_Features:

                    if full_info_Categorical_Features[key]=='object':
                        columns_selction_for_Nominal_Encoding.append(key)    


                Categorical_Features_dataSet=New_clean_data[columns_selction_for_Nominal_Encoding]
                

                # st.dataframe(Categorical_Features_dataSet)  

    # selctecing  encoding for the nominal Features 

                selected_Nonimal_columns = st.multiselect(label='Select Nominal Features ',
                                                    options= Categorical_Features_dataSet.columns,
                                                    default=None
                                                    
                                                    )
                


                st.markdown(
                                """
                                <h3 style='text-align: center;'>Selecting the Encoder For Nominal Features</h3>
                                """,
                                unsafe_allow_html=True
                                )




                st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )
                

                    # Options for the radio button
                options = ["Get Dummies", "One Hot encoding","No Need"]

                    # Creating the radio button
                selected_encoder = st.radio(
                        label="Select an Encoder:",
                        options=options,
                        index=2 ,# Preselecting "Option 2"
                        format_func=lambda x: f"⏩ {x}",  # Adding an arrow before each option
        
                    )



                st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True)
                




                if selected_encoder=='Get Dummies':


                    st.markdown(
                                    """
                                    <h3 style='text-align: center;'>DataSet Encoded By Get Dummies</h3>
                                    """,
                                    unsafe_allow_html=True
                                    )




                    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )

                    encoded_data = pd.get_dummies(New_clean_data, columns=selected_Nonimal_columns, dtype=bool,drop_first=True)
        
                    encoded_data = encoded_data.replace({True: 1, False: 0})

                    st.subheader("DataSet Shape")
                    st.write(f"Total Number of Rows : {encoded_data.shape[0]}")
                    st.write(f"Total Number of Columns : {encoded_data.shape[1]}")
                    st.write("")                        
                    st.markdown(
                                            """
                                            <p style=',color: green;'>Now This DataSet are Ready To Fit to Model, Downolod it And and fit To The Model </p>
                                            """,
                                            unsafe_allow_html=True
                                            ) 
                    
                    st.write("")                         
                    st.dataframe(encoded_data)


                    

                elif selected_encoder=='One Hot encoding':

                    st.markdown(
                                    """
                                    <h3 style='text-align: center;'>DataSet Encoded By One Hot encoding</h3>
                                    """,
                                    unsafe_allow_html=True
                                    )




                    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )

                    encoded_data = pd.get_dummies(New_clean_data, columns=selected_Nonimal_columns, dtype=bool,drop_first=True)
        
                    encoded_data = encoded_data.replace({True: 1, False: 0})

                    st.subheader("DataSet Shape")
                    st.write(f"Total Number of Rows : {encoded_data.shape[0]}")
                    st.write(f"Total Number of Columns : {encoded_data.shape[1]}")
                    st.write("") 
                    st.markdown(
                                            """
                                            <p style='color: green;'>Now This DataSet are Ready To Fit to Model, Downolod it And and fit To The Model </p>
                                            """,
                                            unsafe_allow_html=True
                                            ) 
                    
                    st.write("")                     
                    st.dataframe(encoded_data)


                elif selected_encoder=='No Need':
  
                        
                    cleaned_data5=New_clean_data
                    st.markdown(
                                    """
                                    <h3 style='text-align: center;'>Your Encoded and Final DataSet</h3>
                                    """,
                                    unsafe_allow_html=True
                                    ) 

                    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )                                     
                    st.markdown(
                                            """
                                            <p style='text-align: center;color: green;'>Now This DataSet are Ready To Fit to Model, Downolod it And and fit To The Model </p>
                                            """,
                                            unsafe_allow_html=True
                                            ) 
                    st.write("")                   
                    st.dataframe(cleaned_data5)

                else:
                    st.write("")   
















            else:
             
                st.markdown(
                            """
                            <h3 style='text-align: center;'>Do you Want To Scale Your DataSet </h3>
                            """,
                            unsafe_allow_html=True
                            ) 

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                ) 

                # Options for the radio button
                options = ["Standardization", "Min-Max Scaling","No Need"]

                    # Creating the radio button
                selected_Scaling= st.radio(
                        label="Select an Scalar:",
                        options=options,
                        index=None ,# Preselecting "Option 2"
                        format_func=lambda x: f"⏩ {x}",  # Adding an arrow before each option
        
                    )   


                st.markdown(
                        """
                        <hr style="border: none; height: 2px;width: 100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )   


    # applay Stander Scaling
                if selected_Scaling =='Standardization' :

                    st.markdown(
                                    """
                                    <h3 style='text-align: center;'>Your DataSet are Now Scaled By Standardization</h3>
                                    """,
                                    unsafe_allow_html=True
                                    ) 

                    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )   
                
    

                                    # Initialize the scaler
                    scaler= StandardScaler()

                    # Fit and transform the data
                    scaled_data = scaler.fit_transform(cleaned_data5)

                    # Convert the scaled data back to a DataFrame
                    scaled_data = pd.DataFrame(scaled_data, columns=cleaned_data5.columns)

                    st.markdown(
                                        """
                                        <p style='text-align: center;color: green;'>Now This DataSet are Ready To Fit to Model, Downolod it And and fit To The Model </p>
                                        """,
                                        unsafe_allow_html=True
                                        ) 
                    st.write("")
                    
                    st.dataframe(scaled_data)
                    st.balloons()             










    # apply MinMaxScaling

                elif selected_Scaling =='Min-Max Scaling' :

                    st.markdown(
                                    """
                                    <h3 style='text-align: center;'>Your DataSet are Now Scaled By Min-Max Scaling</h3>
                                    """,
                                    unsafe_allow_html=True
                                    ) 

                    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )   
                
    

                                    # Initialize the scaler
                    Normaloized_Scale= MinMaxScaler()

                    # Fit and transform the data
                    scaled_data = Normaloized_Scale.fit_transform(cleaned_data5)

                    # Convert the scaled data back to a DataFrame
                    scaled_data = pd.DataFrame(scaled_data, columns=cleaned_data5.columns)

                    st.markdown(
                                    """
                                    <p style='text-align: center;color: green;'>Now This DataSet are Ready To Fit to Model, Downolod it And and fit To The Model </p>
                                    """,
                                    unsafe_allow_html=True
                                    ) 
                    st.write("")
                    # st.write("Now This DataSet are Ready To Fit to Model, Downolod it And and fit To The Model ")
                    st.dataframe(scaled_data)
                    st.balloons()




                elif selected_Scaling =='No Need' :
                    scaled_data=cleaned_data5

                    st.markdown(
                                    """
                                    <h3 style='text-align: center;'>Your Clean and Final DataSet</h3>
                                    """,
                                    unsafe_allow_html=True
                                    ) 

                    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )  

                    st.markdown(
                                        """
                                        <p style='text-align: center;color: green;'>Now This DataSet are Ready To Fit to Model, Downolod it And and fit To The Model </p>
                                        """,
                                        unsafe_allow_html=True
                                        ) 
                    st.write("")
                    st.dataframe(scaled_data)
                    st.balloons()



         





        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
    else:
        st.error("No file uploaded yet!")
        
        st.markdown(
                        """
                        <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                        """,
                        unsafe_allow_html=True
                            )





elif selections == 'Classification Models':
    st.markdown(
                                    """
                                    <h1 style='text-align: center;color:orange'>Wellcome to Classification Models</h1>
                                    """,
                                    unsafe_allow_html=True
                                    ) 

    st.markdown(
                            """
                            <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                            """,
                            unsafe_allow_html=True
                        )     


    

    # If 'Upload DataSet' is selected, display the title
    
    # submation_file=st.form("Uploding the file")
    DataSet_name1=st.text_input(placeholder="DataSet Name",label="Insert Dataset Name")

    input=st.expander("Insert your Data Here")
    # File uploader widget
    uploaded_file1 = st.file_uploader(label="Upload CSV file", type=["csv"])

    # Check if a file has been uploaded
    if uploaded_file1 is not None:
        try:
            # Read the CSV file into a DataFrame
            Classification_DataSet = pd.read_csv(uploaded_file1)
            Classification_DataSet=Classification_DataSet.dropna()
            st.success("File uploaded successfully!")
            # Display the DataFrame

            st.title(f"{DataSet_name1} DataSet Loaded")
            st.write(f"Shape {Classification_DataSet.shape}")
            st.dataframe(Classification_DataSet)
            columns_list=Classification_DataSet.columns
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
                
            final_DataSet_of_classification=Classification_DataSet[option]
            st.subheader("Final DataSet")
            st.dataframe(final_DataSet_of_classification)

            spliting_data=final_DataSet_of_classification.columns

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
                
                Input_Features=final_DataSet_of_classification[option2]
                
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
                
                Target_Features=final_DataSet_of_classification[option3]
                
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





            
            Model_selection = st.radio(
                label="Select a Model",
                options=["LogisticRegression Model","DecisionTreeClassifier Model","RandomForestClassifier Model","GradientBoostingClassifier Model","All Models"],
                index=None,  # Setting the default selected index to 1 (Banana)
        
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




# LogisticRegression Model Selection 


            if Model_selection=='LogisticRegression Model':
   


                st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                    """,
                    unsafe_allow_html=True
                )

            
                st.markdown(
                """
                <h1 style='text-align: center;color:#15B5B0;'>LogisticRegression Model</h1>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )       

                tablist=["Model accuracy ","Model Prediction","Confusion Matrix"]

                accuracy_Tab,Predication_tab,Confusion_matrix_tab=st.tabs(tablist)





                with Predication_tab :   
                

                    LogisticRegression_model=LogisticRegression()

                    # feed data to LogisticRegression_model

                    LogisticRegression_model.fit(x_train,y_train)

                    # Now Predicted from the LogisticRegression_model

                    Predication_by_LogisticRegression_model=LogisticRegression_model.predict(x_test)


                    




                    Predication_by_LogisticRegression_model=pd.DataFrame(Predication_by_LogisticRegression_model)
                    Predication_by_LogisticRegression_model.rename(columns={0:"Predicted Values"},inplace=True)
    
                    Predication_by_LogisticRegression_model=Predication_by_LogisticRegression_model.reset_index()
                    Predication_by_LogisticRegression_model.drop(columns='index',axis=1,inplace=True)



                    actual_values=y_test
                    actual_values=pd.DataFrame(actual_values)
                    actual_values=actual_values.reset_index()
                    actual_values.drop(columns='index',axis=1,inplace=True)

        
                    combind_Result=pd.concat([actual_values,Predication_by_LogisticRegression_model],axis=1)
                    st.table(combind_Result)
                



                with accuracy_Tab:


                    LogisticRegression_model_accuracy=accuracy_score(y_test,Predication_by_LogisticRegression_model)

                    LogisticRegression_model_accuracy=np.round(LogisticRegression_model_accuracy,1)*100


                    st.subheader(f"Model Accuracy : {LogisticRegression_model_accuracy} %")

                

                with Confusion_matrix_tab:
                    # Create the confusion matrix
                    cm = confusion_matrix(y_test, Predication_by_LogisticRegression_model)


                    # creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")
                    a=cm.tolist()


                    correct_by_LogisticRegression_model=(a[0][0])+(a[1][1])
                    wrong_by_LogisticRegression_model=(a[0][1])+(a[1][0])

                    # Create the result DataFrame
                    LogisticRegression_model_result = {
                        "Prediction Type": ["Correct Prediction", "Wrong Prediction"],
                        "Count": [correct_by_LogisticRegression_model, wrong_by_LogisticRegression_model]
                    }
                    LogisticRegression_model_result = pd.DataFrame(LogisticRegression_model_result)

                    st.markdown(
                    """
                    <h1 style='text-align: center;color:#DB1F48;'>Prediction Details</h1>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )  

                    st.table(LogisticRegression_model_result)
                    creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")





# DecisionTreeClassifier Model

            elif Model_selection=='DecisionTreeClassifier Model':


                st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                    """,
                    unsafe_allow_html=True
                )

            
                st.markdown(
                """
                <h1 style='text-align: center;color:#15B5B0;'>DecisionTreeClassifier Model</h1>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )       

                tablist=["Model accuracy ","Model Prediction","Confusion Matrix"]

                accuracy_Tab,Predication_tab,Confusion_matrix_tab=st.tabs(tablist)





                with Predication_tab :   
                

                    DecisionTreeClassifier_model=DecisionTreeClassifier()

                    # feed data to LogisticRegression_model

                    DecisionTreeClassifier_model.fit(x_train,y_train)

                    # Now Predicted from the LogisticRegression_model

                    Predication_by_DecisionTreeClassifier_model=DecisionTreeClassifier_model.predict(x_test)


                    




                    Predication_by_DecisionTreeClassifier_model=pd.DataFrame(Predication_by_DecisionTreeClassifier_model)
                    Predication_by_DecisionTreeClassifier_model.rename(columns={0:"Predicted Values"},inplace=True)
    
                    Predication_by_DecisionTreeClassifier_model=Predication_by_DecisionTreeClassifier_model.reset_index()

                    Predication_by_DecisionTreeClassifier_model.drop(columns='index',axis=1,inplace=True)



                    actual_values=y_test
                    actual_values=pd.DataFrame(actual_values)
                    actual_values=actual_values.reset_index()
                    actual_values.drop(columns='index',axis=1,inplace=True)

        
                    combind_Result=pd.concat([actual_values,Predication_by_DecisionTreeClassifier_model],axis=1)
                    st.table(combind_Result)
                



                with accuracy_Tab:


                    DecisionTreeClassifier_model_accuracy=accuracy_score(y_test,Predication_by_DecisionTreeClassifier_model)

                    DecisionTreeClassifier_model_accuracy=np.round(DecisionTreeClassifier_model_accuracy,1)*100


                    st.subheader(f"Model Accuracy : {DecisionTreeClassifier_model_accuracy} %")

                

                with Confusion_matrix_tab:
                    # Create the confusion matrix
                    cm_DecisionTreeClassifier_model = confusion_matrix(y_test, Predication_by_DecisionTreeClassifier_model)


                    # creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")
                    a=cm_DecisionTreeClassifier_model.tolist()


                    correct_by_DecisionTreeClassifier_model=(a[0][0])+(a[1][1])
                    wrong_by_DecisionTreeClassifier_model=(a[0][1])+(a[1][0])

                    # Create the result DataFrame
                    DecisionTreeClassifier_model_result = {
                        "Prediction Type": ["Correct Prediction", "Wrong Prediction"],
                        "Count": [correct_by_DecisionTreeClassifier_model, wrong_by_DecisionTreeClassifier_model]
                    }


                    DecisionTreeClassifier_model_result = pd.DataFrame(DecisionTreeClassifier_model_result)

                    st.markdown(
                    """
                    <h1 style='text-align: center;color:#FF8000;'>Prediction Details</h1>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )  

                    st.table(DecisionTreeClassifier_model_result)
                    creating_cunfusion_Matrix(cm_DecisionTreeClassifier_model,"viridis","DecisionTreeClassifier")










# RandomForestClassifier Model

            elif Model_selection=='RandomForestClassifier Model': 


                st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                    """,
                    unsafe_allow_html=True
                )

            
                st.markdown(
                """
                <h1 style='text-align: center;color:#15B5B0;'>RandomForestClassifier Model</h1>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )       

                tablist=["Model accuracy ","Model Prediction","Confusion Matrix"]

                accuracy_Tab,Predication_tab,Confusion_matrix_tab=st.tabs(tablist)





                with Predication_tab :   
                

                    RandomForestClassifier_model=RandomForestClassifier(n_estimators=100)

                    # feed data to LogisticRegression_model

                    RandomForestClassifier_model.fit(x_train,y_train)

                    # Now Predicted from the LogisticRegression_model

                    Predication_by_RandomForestClassifier_model=RandomForestClassifier_model.predict(x_test)


                    




                    Predication_by_RandomForestClassifier_model=pd.DataFrame(Predication_by_RandomForestClassifier_model)
                    Predication_by_RandomForestClassifier_model.rename(columns={0:"Predicted Values"},inplace=True)
    
                    Predication_by_RandomForestClassifier_model=Predication_by_RandomForestClassifier_model.reset_index()

                    Predication_by_RandomForestClassifier_model.drop(columns='index',axis=1,inplace=True)



                    actual_values=y_test
                    actual_values=pd.DataFrame(actual_values)
                    actual_values=actual_values.reset_index()
                    actual_values.drop(columns='index',axis=1,inplace=True)

        
                    combind_Result=pd.concat([actual_values,Predication_by_RandomForestClassifier_model],axis=1)
                    st.table(combind_Result)
                



                with accuracy_Tab:


                    RandomForestClassifier_model_accuracy=accuracy_score(y_test,Predication_by_RandomForestClassifier_model)

                    RandomForestClassifier_model_accuracy=np.round(RandomForestClassifier_model_accuracy,1)*100


                    st.subheader(f"Model Accuracy : {RandomForestClassifier_model_accuracy} %")

                

                with Confusion_matrix_tab:
                    # Create the confusion matrix
                    cm_RandomForestClassifier_model = confusion_matrix(y_test, Predication_by_RandomForestClassifier_model)


                    # creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")
                    a=cm_RandomForestClassifier_model.tolist()


                    correct_by_RandomForestClassifier_model=(a[0][0])+(a[1][1])
                    wrong_by_RandomForestClassifier_model=(a[0][1])+(a[1][0])

                    # Create the result DataFrame
                    RandomForestClassifier_model_result = {
                        "Prediction Type": ["Correct Prediction", "Wrong Prediction"],
                        "Count": [correct_by_RandomForestClassifier_model, wrong_by_RandomForestClassifier_model]
                    }


                    RandomForestClassifier_model_result = pd.DataFrame(RandomForestClassifier_model_result)

                    st.markdown(
                    """
                    <h1 style='text-align: center;color:#FF8000;'>Prediction Details</h1>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )  

                    st.table(RandomForestClassifier_model_result)
                    creating_cunfusion_Matrix(cm_RandomForestClassifier_model,"YlOrRd","RandomForestClassifier")












# GradientBoostingClassifier Model
            elif Model_selection=='GradientBoostingClassifier Model': 

                st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                    """,
                    unsafe_allow_html=True
                )

            
                st.markdown(
                """
                <h1 style='text-align: center;color:#15B5B0;'>GradientBoostingClassifier Model</h1>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )       

                tablist=["Model accuracy ","Model Prediction","Confusion Matrix"]

                accuracy_Tab,Predication_tab,Confusion_matrix_tab=st.tabs(tablist)





                with Predication_tab :   
                

                    GradientBoostingClassifier_model=GradientBoostingClassifier()

                    # feed data to LogisticRegression_model

                    GradientBoostingClassifier_model.fit(x_train,y_train)

                    # Now Predicted from the LogisticRegression_model

                    Predication_by_GradientBoostingClassifier_model=GradientBoostingClassifier_model.predict(x_test)


                    




                    Predication_by_GradientBoostingClassifier_model=pd.DataFrame(Predication_by_GradientBoostingClassifier_model)
                    Predication_by_GradientBoostingClassifier_model.rename(columns={0:"Predicted Values"},inplace=True)
    
                    Predication_by_GradientBoostingClassifier_model=Predication_by_GradientBoostingClassifier_model.reset_index()

                    Predication_by_GradientBoostingClassifier_model.drop(columns='index',axis=1,inplace=True)



                    actual_values=y_test
                    actual_values=pd.DataFrame(actual_values)
                    actual_values=actual_values.reset_index()
                    actual_values.drop(columns='index',axis=1,inplace=True)

        
                    combind_Result=pd.concat([actual_values,Predication_by_GradientBoostingClassifier_model],axis=1)
                    st.table(combind_Result)
                



                with accuracy_Tab:


                    GradientBoostingClassifier_model_accuracy=accuracy_score(y_test,Predication_by_GradientBoostingClassifier_model)

                    GradientBoostingClassifier_model_accuracy=np.round(GradientBoostingClassifier_model_accuracy,1)*100


                    st.subheader(f"Model Accuracy : {GradientBoostingClassifier_model_accuracy} %")

                

                with Confusion_matrix_tab:
                    # Create the confusion matrix
                    cm_GradientBoostingClassifie_model = confusion_matrix(y_test, Predication_by_GradientBoostingClassifier_model)


                    # creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")
                    a=cm_GradientBoostingClassifie_model.tolist()


                    correct_by_GradientBoostingClassifie_model=(a[0][0])+(a[1][1])
                    wrong_by_GradientBoostingClassifie_model=(a[0][1])+(a[1][0])

                    # Create the result DataFrame
                    GradientBoostingClassifie_model_result = {
                        "Prediction Type": ["Correct Prediction", "Wrong Prediction"],
                        "Count": [correct_by_GradientBoostingClassifie_model, wrong_by_GradientBoostingClassifie_model]
                    }


                    GradientBoostingClassifie_model_result = pd.DataFrame(GradientBoostingClassifie_model_result)

                    st.markdown(
                    """
                    <h1 style='text-align: center;color:#FF8000;'>Prediction Details</h1>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )  

                    st.table(GradientBoostingClassifie_model_result)
                    creating_cunfusion_Matrix(cm_GradientBoostingClassifie_model,"inferno","GradientBoostingClassifier")












# All Models

            elif Model_selection=='All Models': 





                # LogisticRegression model 

                st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                    """,
                    unsafe_allow_html=True
                )

            
                st.markdown(
                """
                <h1 style='text-align: center;color:#15B5B0;'>LogisticRegression Model</h1>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )       

                tablist=["Model accuracy ","Model Prediction","Confusion Matrix"]

                accuracy_Tab,Predication_tab,Confusion_matrix_tab=st.tabs(tablist)





                with Predication_tab :   
                

                    LogisticRegression_model=LogisticRegression()

                    # feed data to LogisticRegression_model

                    LogisticRegression_model.fit(x_train,y_train)

                    # Now Predicted from the LogisticRegression_model

                    Predication_by_LogisticRegression_model=LogisticRegression_model.predict(x_test)


                    




                    Predication_by_LogisticRegression_model=pd.DataFrame(Predication_by_LogisticRegression_model)
                    Predication_by_LogisticRegression_model.rename(columns={0:"Predicted Values"},inplace=True)
    
                    Predication_by_LogisticRegression_model=Predication_by_LogisticRegression_model.reset_index()
                    Predication_by_LogisticRegression_model.drop(columns='index',axis=1,inplace=True)



                    actual_values=y_test
                    actual_values=pd.DataFrame(actual_values)
                    actual_values=actual_values.reset_index()
                    actual_values.drop(columns='index',axis=1,inplace=True)

        
                    combind_Result=pd.concat([actual_values,Predication_by_LogisticRegression_model],axis=1)
                    st.table(combind_Result)
                



                with accuracy_Tab:


                    LogisticRegression_model_accuracy=accuracy_score(y_test,Predication_by_LogisticRegression_model)

                    LogisticRegression_model_accuracy=np.round(LogisticRegression_model_accuracy,1)*100


                    st.subheader(f"Model Accuracy : {LogisticRegression_model_accuracy} %")

                

                with Confusion_matrix_tab:
                    # Create the confusion matrix
                    cm = confusion_matrix(y_test, Predication_by_LogisticRegression_model)


                    # creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")
                    b=cm.tolist()


                    correct_by_LogisticRegression_model=(b[0][0])+(b[1][1])
                    wrong_by_LogisticRegression_model=(b[0][1])+(b[1][0])

                    # Create the result DataFrame
                    LogisticRegression_model_result = {
                        "Prediction Type": ["Correct Prediction", "Wrong Prediction"],
                        "Count": [correct_by_LogisticRegression_model, wrong_by_LogisticRegression_model]
                    }
                    LogisticRegression_model_result = pd.DataFrame(LogisticRegression_model_result)

                    st.markdown(
                    """
                    <h1 style='text-align: center;color:#DB1F48;'>Prediction Details</h1>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )  

                    st.table(LogisticRegression_model_result)
                    creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")



                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )    











# DecisionTreeClassifier Model

                st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                    """,
                    unsafe_allow_html=True
                )

            
                st.markdown(
                """
                <h1 style='text-align: center;color:#FA26A0;'>DecisionTreeClassifier Model</h1>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )       

                tablist=["Model accuracy ","Model Prediction","Confusion Matrix"]

                accuracy_Tab,Predication_tab,Confusion_matrix_tab=st.tabs(tablist)





                with Predication_tab :   
                

                    DecisionTreeClassifier_model=DecisionTreeClassifier()

                    # feed data to LogisticRegression_model

                    DecisionTreeClassifier_model.fit(x_train,y_train)

                    # Now Predicted from the LogisticRegression_model

                    Predication_by_DecisionTreeClassifier_model=DecisionTreeClassifier_model.predict(x_test)


                    




                    Predication_by_DecisionTreeClassifier_model=pd.DataFrame(Predication_by_DecisionTreeClassifier_model)
                    Predication_by_DecisionTreeClassifier_model.rename(columns={0:"Predicted Values"},inplace=True)
    
                    Predication_by_DecisionTreeClassifier_model=Predication_by_DecisionTreeClassifier_model.reset_index()

                    Predication_by_DecisionTreeClassifier_model.drop(columns='index',axis=1,inplace=True)



                    actual_values=y_test
                    actual_values=pd.DataFrame(actual_values)
                    actual_values=actual_values.reset_index()
                    actual_values.drop(columns='index',axis=1,inplace=True)

        
                    combind_Result=pd.concat([actual_values,Predication_by_DecisionTreeClassifier_model],axis=1)
                    st.table(combind_Result)
                



                with accuracy_Tab:


                    DecisionTreeClassifier_model_accuracy=accuracy_score(y_test,Predication_by_DecisionTreeClassifier_model)

                    DecisionTreeClassifier_model_accuracy=np.round(DecisionTreeClassifier_model_accuracy,1)*100


                    st.subheader(f"Model Accuracy : {DecisionTreeClassifier_model_accuracy} %")


                with Confusion_matrix_tab:
                    # Create the confusion matrix
                    cm_DecisionTreeClassifier_model = confusion_matrix(y_test, Predication_by_DecisionTreeClassifier_model)


                    # creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")
                    c=cm_DecisionTreeClassifier_model.tolist()


                    correct_by_DecisionTreeClassifier_model=(c[0][0])+(c[1][1])
                    wrong_by_DecisionTreeClassifier_model=(c[0][1])+(c[1][0])
                    

                    # Create the result DataFrame
                    DecisionTreeClassifier_model_result = {
                        "Prediction Type": ["Correct Prediction", "Wrong Prediction"],
                        "Count": [correct_by_DecisionTreeClassifier_model, wrong_by_DecisionTreeClassifier_model]
                    }


                    DecisionTreeClassifier_model_result = pd.DataFrame(DecisionTreeClassifier_model_result)

                    st.markdown(
                    """
                    <h1 style='text-align: center;color:#FF8000;'>Prediction Details</h1>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )  

                    st.table(DecisionTreeClassifier_model_result)
                    creating_cunfusion_Matrix(cm_DecisionTreeClassifier_model,"viridis","DecisionTreeClassifier")
                




# RandomForestClassifier Model

                st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                    """,
                    unsafe_allow_html=True
                )

            
                st.markdown(
                """
                <h1 style='text-align: center;color:#5DF15D;'>RandomForestClassifier Model</h1>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )       

                tablist=["Model accuracy ","Model Prediction","Confusion Matrix"]

                accuracy_Tab,Predication_tab,Confusion_matrix_tab=st.tabs(tablist)





                with Predication_tab :   
                

                    RandomForestClassifier_model=RandomForestClassifier(n_estimators=100)

                    # feed data to LogisticRegression_model

                    RandomForestClassifier_model.fit(x_train,y_train)

                    # Now Predicted from the LogisticRegression_model

                    Predication_by_RandomForestClassifier_model=RandomForestClassifier_model.predict(x_test)


                    




                    Predication_by_RandomForestClassifier_model=pd.DataFrame(Predication_by_RandomForestClassifier_model)
                    Predication_by_RandomForestClassifier_model.rename(columns={0:"Predicted Values"},inplace=True)
    
                    Predication_by_RandomForestClassifier_model=Predication_by_RandomForestClassifier_model.reset_index()

                    Predication_by_RandomForestClassifier_model.drop(columns='index',axis=1,inplace=True)



                    actual_values=y_test
                    actual_values=pd.DataFrame(actual_values)
                    actual_values=actual_values.reset_index()
                    actual_values.drop(columns='index',axis=1,inplace=True)

        
                    combind_Result=pd.concat([actual_values,Predication_by_RandomForestClassifier_model],axis=1)
                    st.table(combind_Result)
                



                with accuracy_Tab:


                    RandomForestClassifier_model_accuracy=accuracy_score(y_test,Predication_by_RandomForestClassifier_model)

                    RandomForestClassifier_model_accuracy=np.round(RandomForestClassifier_model_accuracy,1)*100


                    st.subheader(f"Model Accuracy : {RandomForestClassifier_model_accuracy} %")

                


                with Confusion_matrix_tab:
                    # Create the confusion matrix
                    cm_RandomForestClassifier_model = confusion_matrix(y_test, Predication_by_RandomForestClassifier_model)


                    # creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")
                    d=cm_RandomForestClassifier_model.tolist()


                    correct_by_RandomForestClassifier_model=(d[0][0])+(d[1][1])
                    wrong_by_RandomForestClassifier_model=(d[0][1])+(d[1][0])
                    

                    # Create the result DataFrame
                    RandomForestClassifier_model_result = {
                        "Prediction Type": ["Correct Prediction", "Wrong Prediction"],
                        "Count": [correct_by_RandomForestClassifier_model, wrong_by_RandomForestClassifier_model]
                    }


                    RandomForestClassifier_model_result = pd.DataFrame(RandomForestClassifier_model_result)

                    st.markdown(
                    """
                    <h1 style='text-align: center;color:#FF8000;'>Prediction Details</h1>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )  

                    st.table(RandomForestClassifier_model_result)
                    creating_cunfusion_Matrix(cm_RandomForestClassifier_model,"YlOrRd","RandomForestClassifier")



                



# GradientBoostingClassifier Model
                st.markdown(
                    """
                    <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                    """,
                    unsafe_allow_html=True
                )

            
                st.markdown(
                """
                <h1 style='text-align: center;color:#15B5B0;'>GradientBoostingClassifier Model</h1>
                """,
                unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                    """,
                    unsafe_allow_html=True
                )       

                tablist=["Model accuracy ","Model Prediction","Confusion Matrix"]

                accuracy_Tab,Predication_tab,Confusion_matrix_tab=st.tabs(tablist)





                with Predication_tab :   
                

                    GradientBoostingClassifier_model=GradientBoostingClassifier()

                    # feed data to LogisticRegression_model

                    GradientBoostingClassifier_model.fit(x_train,y_train)

                    # Now Predicted from the LogisticRegression_model

                    Predication_by_GradientBoostingClassifier_model=GradientBoostingClassifier_model.predict(x_test)


                    




                    Predication_by_GradientBoostingClassifier_model=pd.DataFrame(Predication_by_GradientBoostingClassifier_model)
                    Predication_by_GradientBoostingClassifier_model.rename(columns={0:"Predicted Values"},inplace=True)
    
                    Predication_by_GradientBoostingClassifier_model=Predication_by_GradientBoostingClassifier_model.reset_index()

                    Predication_by_GradientBoostingClassifier_model.drop(columns='index',axis=1,inplace=True)



                    actual_values=y_test
                    actual_values=pd.DataFrame(actual_values)
                    actual_values=actual_values.reset_index()
                    actual_values.drop(columns='index',axis=1,inplace=True)

        
                    combind_Result=pd.concat([actual_values,Predication_by_GradientBoostingClassifier_model],axis=1)
                    st.table(combind_Result)
                



                with accuracy_Tab:


                    GradientBoostingClassifier_model_accuracy=accuracy_score(y_test,Predication_by_GradientBoostingClassifier_model)

                    GradientBoostingClassifier_model_accuracy=np.round(GradientBoostingClassifier_model_accuracy,1)*100


                    st.subheader(f"Model Accuracy : {GradientBoostingClassifier_model_accuracy} %")

                                

                with Confusion_matrix_tab:
                    # Create the confusion matrix
                    cm_GradientBoostingClassifie_model = confusion_matrix(y_test, Predication_by_GradientBoostingClassifier_model)


                    # creating_cunfusion_Matrix(cm,"Blues","LogisticRegression")
                    e=cm_GradientBoostingClassifie_model.tolist()


                    correct_by_GradientBoostingClassifie_model=(e[0][0])+(e[1][1])
                    wrong_by_GradientBoostingClassifie_model=(e[0][1])+(e[1][0])
                    

                    # Create the result DataFrame
                    GradientBoostingClassifie_model_result = {
                        "Prediction Type": ["Correct Prediction", "Wrong Prediction"],
                        "Count": [correct_by_GradientBoostingClassifie_model, wrong_by_GradientBoostingClassifie_model]
                    }


                    GradientBoostingClassifie_model_result = pd.DataFrame(GradientBoostingClassifie_model_result)

                    st.markdown(
                    """
                    <h1 style='text-align: center;color:#FF8000;'>Prediction Details</h1>
                    """,
                    unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    )  

                    st.table(GradientBoostingClassifie_model_result)
                    creating_cunfusion_Matrix(cm_GradientBoostingClassifie_model,"inferno","GradientBoostingClassifier")










# Overall Model Evualuation

                st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:100%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    ) 
                st.markdown(
                    """
                    <h1 style='text-align: center;color:#FF8000;'>All Model Evaluation</h1>
                    """,
                    unsafe_allow_html=True
                    )

                st.markdown(
                        """
                        <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                        """,
                        unsafe_allow_html=True
                    ) 
                
                Full_info_tab,Visulization_tab=st.tabs(["Full Information ","Result Visulization"])


                with Full_info_tab:

                    st.markdown(
                            """
                            <h3 style='text-align: center;color:#FF8000;'>Overall Predication Table</h3>
                            """,
                            unsafe_allow_html=True
                            )

                    st.markdown(
                                """
                                <hr style="border: none; height: 2px;width:50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);margin: 0 auto;" />
                                """,
                                unsafe_allow_html=True
                            )   


                    Overall_Result_0f_Model={

                        "Model":["LogisticRegression","DecisionTreeClassifier ","RandomForestClassifier","GradientBoostingClassifier"],
                        "Correct Prediction":[correct_by_LogisticRegression_model,correct_by_DecisionTreeClassifier_model,correct_by_RandomForestClassifier_model,correct_by_GradientBoostingClassifie_model],
                        "Wrong Prediction":[wrong_by_LogisticRegression_model,wrong_by_DecisionTreeClassifier_model,wrong_by_RandomForestClassifier_model,wrong_by_GradientBoostingClassifie_model]

                    }    

                    Overall_Result_0f_Model=pd.DataFrame(Overall_Result_0f_Model)

                    Overall_Result_0f_Model=Overall_Result_0f_Model.sort_values(by="Correct Prediction",ascending=False).reset_index()
                    Overall_Result_0f_Model.drop(columns="index",axis=1,inplace=True)
                    st.table(Overall_Result_0f_Model)





                    # # Plot the combined comparison of all models
                    # Create the bar plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bar_width = 0.4
                    index = range(len(Overall_Result_0f_Model))

                    bar1 = ax.bar(index, Overall_Result_0f_Model['Correct Prediction'], bar_width, label='Correct Prediction', color='Orange')
                    bar2 = ax.bar([i + bar_width for i in index], Overall_Result_0f_Model['Wrong Prediction'], bar_width, label='Wrong Prediction', color='red')

                    # Add labels and title
                    ax.set_xlabel("Models")
                    ax.set_ylabel("Number of Predictions")
                    ax.set_title("Correct and Wrong Predictions by Models")
                    ax.set_xticks([i + bar_width / 2 for i in index])
                    ax.set_xticklabels(Overall_Result_0f_Model['Model'], rotation=0)
                    ax.legend()

                    # Display the plot in the Streamlit app
                    st.pyplot(fig)
                    




                with Visulization_tab:
                    # Create subplots
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 7))

                    # Plot each confusion matrix
                    sns.heatmap(cm, annot=True, fmt='d', cmap="viridis", ax=ax1)
                    ax1.set_xlabel('Predicted', fontsize=8)
                    ax1.set_ylabel('Actual', fontsize=8)
                    ax1.set_title('Confusion Matrix for LogisticRegression', fontsize=8)

                    sns.heatmap(cm_DecisionTreeClassifier_model, annot=True, fmt='d', cmap="plasma", ax=ax2)
                    ax2.set_xlabel('Predicted', fontsize=8)
                    ax2.set_ylabel('Actual', fontsize=8)
                    ax2.set_title('Confusion Matrix for DecisionTreeClassifier', fontsize=8)

                    sns.heatmap(cm_GradientBoostingClassifie_model, annot=True, fmt='d', cmap="plasma", ax=ax3)
                    ax3.set_xlabel('Predicted', fontsize=8)
                    ax3.set_ylabel('Actual', fontsize=8)
                    ax3.set_title('Confusion Matrix for GradientBoostingClassifier', fontsize=8)

                    sns.heatmap(cm_RandomForestClassifier_model, annot=True, fmt='d', cmap="plasma", ax=ax4)
                    ax4.set_xlabel('Predicted', fontsize=8)
                    ax4.set_ylabel('Actual', fontsize=8)
                    ax4.set_title('Confusion Matrix for RandomForestClassifier', fontsize=8)

                    # Adjust layout
                    plt.tight_layout()

                    # Display the plot in the Streamlit app
                    st.pyplot(fig)




    





                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )
                # Sutibal_model=all_Result.loc[,"Models"]

                Sutibal_model = Overall_Result_0f_Model.loc[Overall_Result_0f_Model.index[0],"Model"] 
 
                st.subheader(f"Most sutible Model For {DataSet_name1} DataSet are {Sutibal_model}")

                st.markdown(
                """
                <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                """,
                unsafe_allow_html=True
                    )  
                st.balloons()   






        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
    else:
        st.error("No file uploaded yet!")
        
        st.markdown(
                        """
                        <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
                        """,
                        unsafe_allow_html=True
                            )











hide_streamlit_style = """
<style>
 MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style,unsafe_allow_html=True)


