import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler 
 

def clean_data():           # A function to import and clean the dataset. 

    '''We can do this because our dataset is not that large 
    (Note: This method (i.e importing the dataset into the app)is not advisable / and not meant to be done for production purpose )'''

    data = pd.read_csv('data/Breast_cancer_data.csv')
    data = data.drop(columns = ['id','Unnamed: 32'], axis = 1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    return data 

def add_sidebar_to_webpage():              # A function to design the sidebar and return the value the user input in the widget

    st.sidebar.title('Cell Nuclei Measurement')

    data = clean_data()     # Assign the clean_data() function to a variable so it van be easily used

    # The Slider_label-- a list that stores the label we want to use for our sidebar alongside the name of the columns in our dataset
    slider_label = [ ('Radius (mean)','radius_mean'), 
                ('Texture (mean)', 'texture_mean'), 
                ('Perimeter (mean)', 'perimeter_mean'),
                ('Area (mean)', 'area_mean'),
                ('Smoothness (mean)', 'smoothness_mean'),
                ('Compactness (mean)', 'compactness_mean'),
                ('Concavity (mean)', 'concavity_mean'),
                ('Concave points (mean)', 'concave points_mean'),
                ('Symmetry (mean)', 'symmetry_mean'),
                ('Fractal_dimension (mean)', 'fractal_dimension_mean'),
                ('Radius (se)', 'radius_se'),
                ('texture (se)', 'texture_se'),
                ('Perimeter (se)', 'perimeter_se'),
                ('Area (se)', 'area_se'),
                ('Smoothness (se)', 'smoothness_se'),
                ('Compactness (se)', 'compactness_se'),
                ('Concavity (se)', 'concavity_se'),
                ('Concave points (se)', 'concave points_se'),
                ('Symmetry (se)', 'symmetry_se'),
                ('Fractal dimension (se)', 'fractal_dimension_se'),
                ('Radius (worst)', 'radius_worst'),
                ('Texture (worst)', 'texture_worst'),('Perimeter_worst', 'perimeter_worst'),
                ('Area (worst)', 'area_worst'),
                ('Smoothness (worst)', 'smoothness_worst'),
                ('Compactness (worst)', 'compactness_worst'), 
                ('Concavity (worst)', 'concavity_worst'),
                ('Concave points (worst)', 'concave points_worst'),
                ('Symmetry (worst)', 'symmetry_worst'),
                ('Fractal_dimension (worst)', 'fractal_dimension_worst')  ]

    
    input_dict = { }        # a Dictionary to store the current value the user select in the slider
    input_dict2 = { }       # a Dictionary to store the current value the user input in the number input 

        #Create a slider or number_input to put in the sidebar so as to receive input from the users. 
        #The label should be the name of the columns in our dataset i.e original colname_indataset e.g 'radius_mean'. 
        # sidebar_label is the name that will appear in the sidebar e.g. 'Radius (mean)'
        #Easiest way to do that is to loop through all the labels in the slider_label variable since it contains all the columns in our dataset

    for sidebar_label, colname_indataset in slider_label:
       # input_dict[colname_indataset] = st.sidebar.slider ( label = sidebar_label, min_value= float(0), max_value= float(data[colname_indataset].max()) )
       
       # or we can also use a number input instead of a slider

        input_dict2[colname_indataset] = st.sidebar.number_input(label = sidebar_label, value= float(data[colname_indataset].mean()) ) 

    return input_dict2  # We return the dictionary because what we need is the value the users input and its the dictionary that stores that


def plot_radar_chart(user_input): 

    # Create a function that will scale the value the user enter to be between 0 and 1
    user_input = scale_the_values(user_input) 

    #We are going to plot this inside the first column i.e col1 
    categories = ['Radius ', 'Texture','Perimeter',
                 'Area', 'Smoothness','Compactness', 'Concavity',
                'Concave points', 'Symmetry', 'Fractal_dimension'] 
    fig = go.Figure()   
    
    fig.add_traces(go.Scatterpolar(
        r= [
            user_input['radius_mean'], user_input['texture_mean'], user_input['perimeter_mean'], user_input['area_mean'], 
            user_input['smoothness_mean'], user_input['compactness_mean'], user_input['concavity_mean'], user_input['concave points_mean'], 
            user_input['symmetry_mean'], user_input['fractal_dimension_mean'] ],
        theta = categories,
        fill = 'tonext', 
        fillcolor='rgb(100,50,85)',
        name = 'Mean Value'  )) 
 
    fig.add_traces(go.Scatterpolar(
        r= [user_input['radius_se'], user_input['texture_se'], user_input['perimeter_se'], user_input['area_se'], 
            user_input['smoothness_se'], user_input['compactness_se'], user_input['concavity_se'], user_input['concave points_se'], 
            user_input['symmetry_se'], user_input['fractal_dimension_se'] ],
        theta = categories,
        fill = 'toself', fillcolor='rgb(90,500,100)',
         name = 'Standard Error'  )) 

    fig.add_traces(go.Scatterpolar(
        r= [user_input['radius_worst'], user_input['texture_worst'], user_input['perimeter_worst'], user_input['area_worst'], 
            user_input['smoothness_worst'], user_input['compactness_worst'], user_input['concavity_worst'], user_input['concave points_worst'], 
            user_input['symmetry_worst'], user_input['fractal_dimension_worst'] ],
        theta = categories,
        fill= 'toself',
        fillcolor='rgb(217,092,100)',
        name = 'Worst' )) 


    fig.update_layout(
        polar = dict( radialaxis = dict(visible = True, range = [0, 1 ])), 
        showlegend = True 
    )

    return fig


def scale_the_values(input_dict2): # You can also use sklearn standard scaler to do this, remember that input_dict2 stores the value the user passes to the number_input in the sidebar
    data = clean_data()
    X = data.drop(columns=['diagnosis'], axis= 1)

    scaled_dict = {}
    for col_name, col_value in input_dict2.items():
        min_val = X[col_name].min()
        max_val =  X[col_name].max()

        scaled_values = (col_value - min_val) / (max_val - min_val)
        scaled_dict[col_name] = scaled_values
    return scaled_dict 
    
def make_prediction(user_input):
    # We import the model we have already built so we can use it to make prediction 
    RF_model = joblib.load('model\RF.pkl')
    scaler =  joblib.load('model\scaler.pkl')

    # We convert the dictionary output from key&value pairs to an array
    user_input_array = np.array(list(user_input.values())).reshape(1, -1)
    
    #scale the value that comes from user_input_array
    user_input_array_scaled = scaler.transform(user_input_array) 

    st.subheader("Cancerous Growth Prediction")
    #st.write('Based on the parameter you supplied, the Growth is')

    prediction = RF_model.predict(user_input_array_scaled)
    if prediction == 0:
        st.write(' Based on the parameter you supplied, the Growth is ***Benign**')
    else:
        st.write('Based on the parameter you supplied, the Growth is ***Malicious**')

    st.write('-------------------------')
    
    # You can also display the probabililty of it been Benign or Malicious
    st.write('Probability of Cancerous Growth been ***Benign** is', RF_model.predict_proba(user_input_array_scaled)[0][0])
    st.write('--------------------------')
    st.write('Probability of Cancerous Growth been ***Malicious** is', RF_model.predict_proba(user_input_array_scaled)[0][1])
    
    st.write('................................................................................')

    st.write('This App should only be used for diagnosis purpose only> The final decision should always be left to the specialist to make')
    st.write('We wish you a life free of health challenges')

def main():
    #Change the  default page title 
    st.set_page_config(page_title= 'Breast-Cancer-Predictor', page_icon='heart', layout = 'wide', initial_sidebar_state = 'expanded')


        # Inject CSS for full app styling
    st.markdown("""
        <style>
            /* General page style */
            body {
                background-color: #f9f9f9;
                font-family: 'Segoe UI', sans-serif;
            }

            /* Sidebar styling */
            section[data-testid="stSidebar"] {
                background-color: #f0f2f6;
                padding: 20px;
            }

            /* Titles */
            h1, h2, h3 {
                color: #4B0082; /* Indigo */
            }

            /* Subheaders */
            h4, h5, h6 {
                color: #444;
            }

            /* Buttons */
            div.stButton > button {
                background-color: #4B0082;
                color: white;
                border-radius: 8px;
                height: 3em;
                font-weight: bold;
                transition: 0.3s;
                border: none;
            }
            div.stButton > button:hover {
                background-color: #6a0dad;
                color: #fff;
            }

            /* Data display boxes */
            .stDataFrame, .stTable {
                background-color: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }

            /* Chart container */
            .stPlotlyChart {
                border-radius: 10px;
                padding: 5px;
                background-color: white;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }

            /* Tabs styling */
            button[data-baseweb="tab"] {
                font-weight: bold;
                color: #4B0082 !important;
            }

            /* Input widgets */
            input, select, textarea {
                border: 1px solid #ccc !important;
                border-radius: 6px !important;
                padding: 6px !important;
            }

            /* Sliders */
            .stSlider > div[data-baseweb="slider"] > div > div {
                background-color: #4B0082 !important;
            }

            /* Number Inputs */
            input[type=number] {
                border: 1px solid #4B0082 !important;
            }

            /* Checkbox */
            label[data-baseweb="checkbox"] div {
                border: 2px solid #4B0082 !important;
                border-radius: 4px !important;
            }
            label[data-baseweb="checkbox"] div[aria-checked="true"] {
                background-color: #4B0082 !important;
            }

            /* Select boxes */
            div[data-baseweb="select"] > div {
                border-color: #4B0082 !important;
            }
        </style>
    """, unsafe_allow_html=True)
           



    with st.container(border = True):
        # set title for the application
        st.title('Breast Cancer Predictor')
        st.write('Please Connect This App To Your Cytolab To Help Diagnose Breast Cancer. This App Uses A Machine Learning Model To Help Predict Whether A Breast Mass Is Benign Or Maligant.')
        st.write('You Can Adjust The Measurement Manually Using The Slider In The Sidebar')
 

    # call the add_sidebar_to_webpage function to plot the sidebar. (Recall that the output of the function is a dictionary that stores the current value the user input)
    # Named it user_input because its output is the value the user inputed into the widget in the sidebar
    user_input = add_sidebar_to_webpage() 


    # Divide the webpage into two tabs for easy navigation
    tab1, tab2 = st.tabs(tabs= ['📈 Chart', '🗃 Medical Test Value Confirmation'])
    with tab1: 
        st.header('Record Visuals')
        # Create a column and Plot the radar chart there
        col = st.columns(spec=1, border = True) 
        with col[0]:
            radar_chart = plot_radar_chart(user_input) 
            # use the plotly_chart function to plot the chart
            st.plotly_chart(radar_chart)

    with tab2:
        col1, col2 = st.columns(spec = [3, 1.5], border = True)
        # display the value the user enter in the number_input widget so they can confirm before making prediciton
        with col1:
            st.header('Medical Test input')
            st.subheader('Please Confirm Your Input Before Proceeding To Next Step') 
            st.write(user_input)
        
        with col2:
            make_prediction(user_input)

if __name__ == '__main__':
    main() 