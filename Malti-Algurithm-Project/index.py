# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu




st.set_page_config(
        page_icon="https://www.svgrepo.com/show/390209/bug-insect-code-development.svg",
        page_title="Code Debugger | app",
        layout="wide"

        
        
            )


def heading(heading,color):
        heading_code=f"""

            <{heading} style='text-align: center;color: {color};'>Uplode DataSet</{heading}>


                    """
        st.markdown(heading_code,unsafe_allow_html=True)




def styled_paragraph(content, color="#37474F", font_size="14px"):


    # Define the CSS style block with dynamic values
    css_style = f"""
    <style>
        .custom-paragraph {{
            color: {color};
            font-size: {font_size};
        
        }}
    </style>
    """

    # Insert the CSS into the Streamlit app
    st.markdown(css_style, unsafe_allow_html=True)

    # Define the HTML content with the custom class
    html_content = f"""
    <p class="custom-paragraph">
        {content}
    </p>
    """

    # Display the HTML content in Streamlit
    st.markdown(html_content, unsafe_allow_html=True)








def check_datatype(dataframe,colmun):
    dtype = dataframe[colmun].dtype
    return dtype


# Function to find highly correlated columns
def find_highly_correlated_columns(df, threshold=0.8):
    corr_matrix = df.corr()  # Calculate correlation matrix
    # Create a boolean matrix of high correlations
    high_corr = corr_matrix.abs() > threshold
    # Get pairs of highly correlated columns
    highly_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j]) 
                         for i in range(len(corr_matrix.columns)) 
                         for j in range(i) 
                         if high_corr.iloc[i, j] and corr_matrix.iloc[i, j] != 1.0]
    return highly_corr_pairs


def extract_numerical_columns(df):

    # Select columns with numerical data types
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    return numerical_columns


# Function to plot graphs for a continuous variable
def numarical_Features(df, column_name):
    
    col_name=column_name[0]

    st.title(f"Analysis for {col_name}")

    graph1, graph2 = st.columns([2, 1])
    

    with graph1:
        # Color selection for Histogram
        st.write(f"##### Distribuation Of {col_name}")
        hist_color = st.color_picker("Select color for Histogram", "#1f77b4", key=f"hist_color_{column_name}")
        hist_fig = px.histogram(df, x=column_name, nbins=30, title=f"Histogram of {col_name}", color_discrete_sequence=[hist_color])
        st.plotly_chart(hist_fig)

    with graph2:
        # Color selection for Box Plot
        st.write(f"##### Outliers in {col_name}")
        box_color = st.color_picker("Select color for Box Plot", "#ff7f0e", key=f"box_color_{col_name}")
        box_fig = px.box(df, y=column_name, title=f"Box Plot of {col_name}", color_discrete_sequence=[box_color])
        st.plotly_chart(box_fig)

    Line_Break(100)
    # Scatter Plot (If there are at least two columns)
    if len(df.columns) > 1:
        other_columns = [col for col in df.columns if col != col_name]
        scatter_column = st.selectbox("Select another column for Scatter Plot", other_columns, key=f"scatter_column_{col_name}")
        
        st.write("##### Finding Relationship")
        scatter_color = st.color_picker("Select color for Scatter Plot", "#2ca02c", key=f"scatter_color_{col_name}")
        scatter_fig = px.scatter(df, x=col_name, y=scatter_column, title=f"Scatter Plot of {col_name} vs {scatter_column}", color_discrete_sequence=[scatter_color])
        scatter_fig.update_layout(
            width=800,
            height=600,
            xaxis_title=col_name,  # Set x-axis title
            yaxis_title=scatter_column  # Set y-axis title
        )
        scatter_fig.update_traces(texttemplate='%{x} <br>%{y}', textposition='top center')
        st.plotly_chart(scatter_fig)
  # Heatmap for correlation

    Line_Break(100)

    numrical_col=extract_numerical_columns(df)
    numrical_dataset=df[numrical_col]

    # Heatmap for correlation
    st.write("##### Correlation Heatmap")
    # Example usage of the function in your Streamlit app
    styled_paragraph("Select color scale for Heatmap")
    heatmap_colorscale = st.selectbox("", 
                                      ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Rainbow', 'RdBu'], 
                                      key=f"heatmap_colorscale_{col_name}")
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=numrical_dataset.corr().values,
        x=numrical_dataset.corr().columns,
        y=numrical_dataset.corr().index,
        colorscale=heatmap_colorscale,
        zmin=-1, zmax=1
    ))
    heatmap_fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title='Columns',  # Set x-axis title
        yaxis_title='Columns',  # Set y-axis title
        width=800,
        height=600,
    )
    heatmap_fig.update_traces(text=df.corr().values, texttemplate='%{text:.2f}', textfont_size=12)
    st.plotly_chart(heatmap_fig)
    










        




def Line_Break(width):
        line_code=f"""

            <hr style="border: none; height: 2px;width: {width}%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />


                    """
        st.markdown(line_code,unsafe_allow_html=True)

def Line_Break_start(width):
        line_code=f"""

            <hr style="border: none; height: 2px;width: {width}%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />


                    """
        st.markdown(line_code,unsafe_allow_html=True)


def missing_values_count(data):
        # Calculate the count of missing values for each column
        missing_counts = data.isnull().sum()
        return missing_counts.sum()

def duplicate_Values_count(df):
    # Identify duplicate rows
    num_duplicates = df.duplicated(keep=False).sum()

    return num_duplicates

def plot_data_type_distribution(dataframe):

    
    # Initialize counters for different data types
    float_count = 0
    int_count = 0
    object_count = 0

    # Count columns by data type
    for col in dataframe.columns:
        dtype = dataframe[col].dtype
        if dtype == "float64":
            float_count += 1
        elif dtype == "int64":
            int_count += 1
        elif dtype == "object":
            object_count += 1

    # Calculate total number of columns
    total_columns = len(dataframe.columns)

    # Calculate numerical and categorical feature counts and percentages
    numerical_count = float_count + int_count
    numerical_percentage = (numerical_count / total_columns) * 100
    categorical_percentage = (object_count / total_columns) * 100

    # Data for Pie Chart
    labels = ['Numerical Features', 'Categorical Features']
    values = [numerical_percentage, categorical_percentage]

    # Create Pie Chart with custom colors
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.4,
        marker=dict(colors=['#FFA500', '#1E90FF']),  # Custom colors: orange and blue
        textinfo='label+percent',  # Show labels and percentages on the chart
        hoverinfo='label+value+percent',  # Show all info on hover
        textfont=dict(size=16),  # Increase text font size for readability
    )])

    # Update Layout
    fig.update_layout(
        title=dict(
            text='Distribution of Numerical and Categorical Features',
            x=0.5,  # Center the title
            xanchor='center',
            font=dict(size=22)  # Increase title font size
        ),
        annotations=[dict(
            text='Data Types', 
            x=0.5, y=0.5, 
            font_size=18, 
            showarrow=False
        )],
        legend=dict(
            font=dict(size=14),  # Increase legend font size
            orientation='h',  # Horizontal legend
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=80, b=50, l=50, r=50),  # Adjust margins for spacing
        width=700,
        height=500,
 
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)




def plot_data_type_distribution_barchart(dataframe):

    # Initialize counters
    float_count = 0
    int_count = 0
    object_count = 0

    # Count columns by data type
    for col in dataframe.columns:
        dtype = dataframe[col].dtype
        if dtype == "float64":
            float_count += 1
        elif dtype == "int64":
            int_count += 1
        elif dtype == "object":
            object_count += 1

    # Data for Bar Chart
    labels = ['Float', 'Integer', 'Object']
    counts = [float_count, int_count, object_count]

    # Create Bar Chart
    fig = go.Figure(data=[go.Bar(x=labels, y=counts, marker_color=['blue', 'green', 'red'])])

    # Update Layout
    fig.update_layout(
        title='Count of Each Data Type',
        xaxis_title='Data Type',
        yaxis_title='Count',
        width=600,
        height=400
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

def plot_missing_values_percentage(dataframe):
    # Calculate the percentage of missing values in each column
    missing_percentages = dataframe.isnull().mean() * 100

    # Filter out columns with 0% missing values
    missing_percentages = missing_percentages[missing_percentages > 0]

    # Data for Pie Chart
    labels = missing_percentages.index
    values = missing_percentages.values

    # Create Pie Chart
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.3,
        marker=dict(colors=['orange', 'blue', 'red', 'green', 'purple'])  # Customize colors
    )])

    # Update Layout
    fig.update_layout(
        title='Percentage of Missing Values in Each Column',
        annotations=[dict(text='Missing Data', x=0.5, y=0.5, font_size=10, showarrow=False)],
        legend={'font': {'size': 14}},  # Legend font size
        margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for better spacing
        width=600,
        height=400
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)


def plot_missing_values_count_barchart(dataframe):

    # Calculate the count of missing values for each column
    missing_counts = dataframe.isnull().sum()

    # Data for Bar Chart
    labels = missing_counts.index
    counts = missing_counts.values

    # Create Bar Chart with different colors for each bar
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']

    fig = go.Figure(data=[go.Bar(x=labels, y=counts, marker_color=colors[:len(labels)])])

    # Update Layout
    fig.update_layout(
        title='Count of Missing Values in Each Column',
        xaxis_title='Columns',
        yaxis_title='Count of Missing Values',
        xaxis=dict(
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        ),
        yaxis=dict(
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        )
            ,
        width=500,
        height=600

    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

def plot_duplicate_vs_unique_pie_chart(df):
    # Identify duplicate rows
    num_duplicates = df.duplicated(keep=False).sum()
    num_unique = len(df) - num_duplicates

    # Prepare data for pie chart
    data_summary = pd.DataFrame({
        'Type': ['Unique Rows', 'Duplicate Rows'],
        'Count': [num_unique, num_duplicates]
    })

    # Create a pie chart with Plotly
    fig = px.pie(
        data_summary,
        names='Type',
        values='Count',
        title='Percentage of Unique vs Duplicate Rows',
        color_discrete_sequence=['blue', 'yellow'],
        hole=0.4  # Donut style
    )

    # Display the pie chart in Streamlit
    st.plotly_chart(fig)



def plot_duplicate_vs_unique_bar_chart(df):
    # Identify duplicate rows
    num_duplicates = df.duplicated(keep=False).sum()
    num_unique = len(df) - num_duplicates

    # Prepare data for bar chart
    data_summary = pd.DataFrame({
        'Type': ['Unique Rows', 'Duplicate Rows'],
        'Count': [num_unique, num_duplicates]
    })

    # Create a bar chart with Plotly
    fig = px.bar(
        data_summary,
        x='Type',
        y='Count',
        title='Count of Unique vs Duplicate Rows',
        color='Type',
        color_discrete_sequence=['orange', '#EF553B']
    )

    # Display the bar chart in Streamlit
    st.plotly_chart(fig)





def plot_class_distribution_pie_chart(df, column_name):
    # Calculate percentage of each class
    class_counts = df[column_name].value_counts(normalize=True) * 100
    class_percentages = class_counts.reset_index()
    class_percentages.columns = ['Class', 'Percentage']
    
    # Define custom colors (you can change these to any colors you like)
    custom_colors = ['orange', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2']

    # Create a pie chart with Plotly Express
    fig = px.pie(class_percentages, names='Class', values='Percentage', 
                 title='Class Distribution Percentage',
                 color_discrete_sequence=custom_colors)
    
    # Display the pie chart in Streamlit
    st.plotly_chart(fig)



def plot_class_distribution_bar_chart(df, column_name):
    # Calculate count of each class
    class_counts = df[column_name].value_counts()
    class_counts_df = class_counts.reset_index()
    class_counts_df.columns = ['Class', 'Count']
    
    # Create a bar chart with Plotly Express, assigning different colors to each class
    fig = px.bar(class_counts_df, x='Class', y='Count', 
                 title='Class Distribution Count', 
                 labels={'Class': 'Class', 'Count': 'Count'},
                 text='Count',
                 color='Class',  # Color by 'Class'
                 color_discrete_sequence=px.colors.qualitative.Plotly)  # Use Plotly's color sequence
    
    # Display the bar chart in Streamlit
    st.plotly_chart(fig)




def catgorical_varible(df,column_name):
        Line_Break(100)
        st.subheader(f'{column_name} Feature Analysis')

        graph1, graph2 = st.columns([2, 1])

        with graph1:
            plot_class_distribution_pie_chart(df,column_name)

        with graph2:
            plot_class_distribution_bar_chart(df, column_name)
        Line_Break(100)











# CSS styling for the Streamlit app
page_bg_img = f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
    background-repeat: no-repeat;
    background-attachment: fixed;
    background: rgb(18 18 18 / 0%);
}}

.st-emotion-cache-1gv3huu {{
    position: relative;
    top: 2px;
    background-color: #000;
    z-index: 999991;
    min-width: 244px;
    max-width: 550px;
    transform: none;
    transition: transform 300ms, min-width 300ms, max-width 300ms;
}}

.st-emotion-cache-1jicfl2 {{
    width: 100%;
    padding: 4rem 1rem 4rem;
    min-width: auto;
    max-width: initial;

}}


.st-emotion-cache-4uzi61 {{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background: rgb(240 242 246);
    box-shadow: 0 5px 8px #6c757d;
}}

.st-emotion-cache-1vt4y43 {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: #ffc107;
    border: 1px solid rgba(49, 51, 63, 0.2);
}}

.st-emotion-cache-qcpnpn {{
    border: 1px solid rgb(163, 168, 184);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background-color: rgb(38, 39, 48);
    MARGIN-TOP: 9PX;
    box-shadow: 0 5px 8px #6c757d;
}}


.st-emotion-cache-15hul6a {{
    user-select: none;
    background-color: #ffc107;
    border: 1px solid rgba(250, 250, 250, 0.2);
    
}}

.st-emotion-cache-1hskohh {{
    margin: 0px;
    padding-right: 2.75rem;
    color: rgb(250, 250, 250);
    border-radius: 0.5rem;
    background: #000;
}}

.st-emotion-cache-12pd2es {{
    margin: 0px;
    padding-right: 2.75rem;
    color: #f0f2f6;
    border-radius: 0.5rem;
    background: #000;
}}

p, ol, ul, dl {{
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 1rem;
    font-weight: 400;
    color: whitesmoke;
}}

.st-emotion-cache-1v7f65g .e1b2p2ww15 {{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    background: #212121;
    color: white;
}}

.st-emotion-cache-1aehpvj {{
    color: #f5deb3ab;
    font-size: 12px;
    line-height: 1.25;
}}

.st-emotion-cache-1ny7cjd {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: #FFA000;
    border: 1px solid rgba(49, 51, 63, 0.2);
}}

.st-cg {{
    caret-color: rgb(23 24 27);
 
    background: #bdbdbdc4;

}}

.st-emotion-cache-1jicfl2 {{
    width: 100%;
    padding: 2rem 1rem 4rem;
    min-width: auto;
    max-width: initial;
}}

.st-emotion-cache-ocqkz7 {{
    display: flex;
    flex-wrap: wrap;
    -webkit-box-flex: 1;
    flex-grow: 1;
    -webkit-box-align: stretch;
    align-items: stretch;
    gap: 1rem;
    padding: 20px;
}}

# .st-emotion-cache-ue6h4q {{
#     font-size: 14px;
#     color: rgb(49, 51, 63);
#     display: flex;
#     visibility: visible;
#     margin-bottom: 0.25rem;
#     height: auto;
#     min-height: 1.5rem;
#     vertical-align: middle;
#     flex-direction: row;
#     -webkit-box-align: center;
#     align-items: center;
#     display: none;
# }}


</style>
"""

# Apply CSS styling to the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # Display logo image
    st.image("Logo4.png", use_column_width=True)

    # Adding a custom style with HTML and CSS for sidebar
    st.markdown("""
        <style>
            .custom-text {
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                color:#ffc107
            }
            .custom-text span {
                color: #04ECF0; /* Color for the word 'Recommendation' */
            }
        </style>
    """, unsafe_allow_html=True)
    
 
    # Displaying the subheader with custom styling
    st.markdown('<p class="custom-text"><span>DashBoard</span> Pro</p>', unsafe_allow_html=True)

    heading("h3","White")
    Line_Break(100)

    file_upload_sucuss=0
    # Expander for file upload
    with st.expander("Insert your Data Here"):
        # DataSet_name=st.text_input(placeholder="DataSet Name",label="Insert Dataset Name")
        # File uploader widget
        uploaded_file = st.file_uploader(label="Upload CSV file", type=["csv"])

        # Check if a file has been uploaded
        if uploaded_file is not None:
            try:
                # Read the CSV file into a DataFrame
                data = pd.read_csv(uploaded_file)
                
                file_upload_sucuss=1
  
                st.success("File uploaded successfully!")
                
                file_upload_sucuss=1


                


            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
        else:
            st.info("Please upload a CSV file to proceed.")


        if file_upload_sucuss==1:

            # Multi-select widget for column selection
            selected_columns = st.multiselect(
                        label="Select Columns for Analysis",
                        options=data.columns
                    
                    )
            






    # HTML and CSS for the GitHub button
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

    # Display the GitHub button in the sidebar
    st.markdown(github_button_html, unsafe_allow_html=True)
    
    # Footer HTML and CSS
    footer_html = """
    <div style="padding:10px; text-align:center;margin-top: 10px;">
        <p style="font-size:20px; color:#ffffff;">Made with ❤️ by Salman Malik</p>
    </div>
    """

    # Display footer in the sidebar
    st.markdown(footer_html, unsafe_allow_html=True)




def Main_PAGE(data):
                    # Display the DataFrame
    st.title('Data Analytics Dashboard')
    
    # st.dataframe(data)

    rows=data.shape[0]
    Features=data.shape[1]
        
    # Get the data types of each column
    data_types = data.dtypes

    # Convert to a set to get unique data types
    unique_data_types = set(data_types)

    null_values = data.isnull().sum()
    null_values=null_values.sum()


    # Get the number of unique data types
    num_unique_data_types = len(unique_data_types)




    # Inject custom CSS
    st.markdown(
        """
        <style>
        /* Change the background color of the tab container */
        div[data-baseweb="tab-list"] {
            background-color: #00BCD4;
            padding: 5px;
            border-radius: 20px;
        }

        /* Change the color of the selected tab */
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #0008ff;
        color: white;
        border-radius: 20px;
        padding: 10px;
        border: none;
        }


        /* Change the color of non-selected tabs */
        div[data-baseweb="tab-list"] button {
            # background-color: #ffd54fdb;
            # color: black;
            border-radius: 10px;
            padding: 10px;
            border: none;
            margin: 0 5px;
            color: #fff;
            background-color: #007bff;
            border-color: #007bff;
        }

        /* Change the hover color of non-selected tabs */
        div[data-baseweb="tab-list"] button:hover {
            background-color: #ffcc00;
            color: white;
        }





        </style>
        """,
        unsafe_allow_html=True
    )



    if missing_values_count(data)  > 1 and duplicate_Values_count(data) >1:
        Line_Break(100)

        tab1, tab2,tab3 = st.tabs(["DataTypes Analysis", "Missing Values Analysis","Duplicate Values Analysis"])
        
        
        
        with tab1:
            Line_Break(100)
            
            st.subheader('DataTypes Analysis')
            graph1, graph2 = st.columns([2, 1])

            with graph1:
                plot_data_type_distribution(data)

            with graph2:
                plot_data_type_distribution_barchart(data)

        Line_Break(100)

        with tab2:
            Line_Break(100)
            
            st.subheader('Missing Values Analysis')
            graph3, graph4 = st.columns([2, 1])

            with graph3:
                plot_missing_values_percentage(data)

            with graph4:
                plot_missing_values_count_barchart(data)

        with tab3:
            Line_Break(100)
            
            st.subheader('Duplicate Values Analysis')
            graph3, graph4 = st.columns([2, 1])

            with graph3:
                plot_duplicate_vs_unique_pie_chart(data)

            with graph4:
                plot_duplicate_vs_unique_bar_chart(data)

        # Function to create boxplots
        def create_boxplots(data, columns):
            st.write("Here you can display a boxplot or any other analysis for", columns[0])
            # Example plot (you can replace this with actual boxplot code)
            st.line_chart(data[columns[0]])



        # Create dynamic tabs based on selected columns
        if selected_columns:
            tabs = st.tabs(selected_columns)
            for tab, column in zip(tabs, selected_columns):
                with tab:

                    


                    colmuns_datatype=check_datatype(data,column)



                    if colmuns_datatype =="object":
                        catgorical_varible(data,column)


                    elif colmuns_datatype =="float64" or colmuns_datatype =="int64":

                        numarical_Features(data,[column])
                    





    elif  missing_values_count(data)  > 1:
        Line_Break(100)
        tab1, tab2 = st.tabs(["DataTypes Analysis", "Missing Values Analysis"])

        with tab1:
            Line_Break(100)
            st.subheader('DataTypes Analysis')
            graph1, graph2 = st.columns([2, 1])

            with graph1:
                plot_data_type_distribution(data)

            with graph2:
                plot_data_type_distribution_barchart(data)

        Line_Break(100)

        with tab2:
            Line_Break(100)
            
            st.subheader('Missing Values Analysis')
            graph3, graph4 = st.columns([2, 1])

            with graph3:
                plot_missing_values_percentage(data)

            with graph4:
                plot_missing_values_count_barchart(data)

        # Function to create boxplots
        def create_boxplots(data, columns):
            st.write("Here you can display a boxplot or any other analysis for", columns[0])
            # Example plot (you can replace this with actual boxplot code)
            st.line_chart(data[columns[0]])



        # Create dynamic tabs based on selected columns
        if selected_columns:
            tabs = st.tabs(selected_columns)
            for tab, column in zip(tabs, selected_columns):

                with tab:
                    colmuns_datatype=check_datatype(data,column)
                    if colmuns_datatype =="object":
                        catgorical_varible(data,column)


                    elif colmuns_datatype =="float64" or colmuns_datatype =="int64":

                        numarical_Features(data,[column])
        Line_Break(100)




    else:
        Line_Break(100)
        # Create a single tab
        tab1 = st.tabs(["DataTypes Analysis"])[0]  # Access the first tab
        

        with tab1:
            Line_Break(100)
            st.subheader('DataTypes Analysis')
            
            graph1, graph2 = st.columns([2, 1])

            with graph1:
                plot_data_type_distribution(data)

            with graph2:
                plot_data_type_distribution_barchart(data)

        Line_Break(100)
        # Function to create boxplots
        def create_boxplots(data, columns):
            st.write("Here you can display a boxplot or any other analysis for", columns[0])
            # Example plot (you can replace this with actual boxplot code)
            st.line_chart(data[columns[0]])
            Line_Break(100)



        # Create dynamic tabs based on selected columns
        if selected_columns:
            tabs = st.tabs(selected_columns)
            for tab, column in zip(tabs, selected_columns):
                with tab:
                    Line_Break(100)

                    colmuns_datatype=check_datatype(data,column)

                    if colmuns_datatype =="object":
                        catgorical_varible(data,column)


                    elif colmuns_datatype =="float64" or colmuns_datatype =="int64":

                        numarical_Features(data,[column])
        














if file_upload_sucuss ==1:
    Main_PAGE(data)


else:

    def show_description():
    # HTML content with CSS styling
        description_html = """
        <div class="app-description">
            <h2>Streamlit EDA Web App</h2>
            <p>This web application is designed to provide a comprehensive and interactive platform for visualizing data and conducting Exploratory Data Analysis (EDA). Built using Streamlit, the app allows users to easily upload datasets, generate insightful visualizations, and uncover key patterns and trends in their data.</p>
            <p>With features such as dynamic histograms, scatter plots, box plots, and heatmaps, users can explore their data from multiple angles. The app also includes options for color customization, making it a versatile tool for data analysts and enthusiasts alike.</p>
            <p>Whether you are analyzing missing values, data types, or correlations, this app streamlines the process of EDA, helping you make informed decisions based on your data.</p>
        </div>
        """

        # Apply custom CSS
        st.markdown("""
        <style>
        .app-description {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px black;
            MARGIN-TOP: 80PX;
        }
        .app-description h1 {
            color: #333333;
            font-size: 24px;
            margin-bottom: 10px;
        }
        .app-description p {
            color: #555555;
            font-size: 18px;
            line-height: 1.6;
        }
                   
        </style>
        """, unsafe_allow_html=True)

        # Display the HTML content
        st.markdown(description_html, unsafe_allow_html=True)


    # Call the function in your main app code
    show_description()


