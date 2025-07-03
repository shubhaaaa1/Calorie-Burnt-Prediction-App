import numpy as np
import streamlit as st
import pickle

model = pickle.load(open('model.pkl','rb'))

st.set_page_config(page_title='Calorie Burnt Prediction' , layout='centered')
st.title('Calorie Burnt Prediction App')
st.write("""
This app predicts the **Calories Burnt** based on your exercise session.
Provide your details in the sidebar and get an instant prediction!
""")



st.sidebar.header('Enter Person Details')
gender = st.sidebar.text_input("Enter the Gender Of the Person: ")
age = st.sidebar.number_input('Enter the Age of the Person : ',min_value=0,max_value=120,step=1)
height = st.sidebar.number_input('Enter the Height of the Person :',min_value=100.0,max_value=230.0,step=1.0,value=100.0)
weight = st.sidebar.number_input('Enter the Weight of the Person :',min_value=0.0,max_value=150.0,step=1.0)
duration = st.sidebar.number_input('Enter the Duration of The Exercise : ',min_value=0.0,step=0.1)
heart_rate = st.sidebar.number_input('Enter the Heart Rate of the Person : ',min_value=0.0,max_value=300.0,step=0.1)
body_temp = st.sidebar.number_input('Enter the Body Temp Of the Person : ',min_value=0.0,max_value=100.0,step=0.1)

if st.button("Predict Your Burnt Calorie"):
    gender_map = {'male' : 0,'female' : 1}
    feature = (gender_map[gender],age,height,weight,duration,heart_rate,body_temp)
    numpy_array = np.asarray(feature)
    reshaped = numpy_array.reshape(1,-1)
    pred = model.predict(reshaped)
    st.subheader('✅ Prediction Result')
    st.success(f'Estimated Calories Burnt: {pred[0]:.2f} kcal')

    # Additional info
    st.info('Stay healthy! Regular exercise is key to a fit life. 💪')

st.markdown('---')
st.write('Created with ❤️ by Shubham Agnihotri')
