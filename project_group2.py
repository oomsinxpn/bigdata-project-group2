import streamlit as st
import pandas as pd 
import pickle

st.write("""
# Big Data Management and Analytics
Show **TCAS information** of students of Mae Fah Lung University in the **academic year 2020**
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

def get_input():
   #widgets
   v_Faculty = st.sidebar.selectbox('Faculty', ['School of Liberal Arts', 'School of Science',
       'School of Management', 'School of Information Technology',
       'School of Agro-industry', 'School of Law',
       'School of Cosmetic Science', 'School of Health Science',
       'School of Nursing', 'School of Medicine', 'School of Dentistry',
       'School of Social Innovation', 'School of Sinology',
       'School of Integrative Medicine'])

   v_GPAX = st.sidebar.number_input('GPAX', min_value=0.00, max_value=4.00)

   st.sidebar.write('Expectation for studying in MFU:') 
   v_Q1 = st.sidebar.checkbox('beautiful scenary and atmosphere')
   v_Q2 = st.sidebar.checkbox('quality of life')
   v_Q3 = st.sidebar.checkbox('campus and facilitiese')
   v_Q4 = st.sidebar.checkbox('modern and ready-to-use learning support and facilities')
   v_Q5 = st.sidebar.checkbox('sources of student scholarship')
   v_Q6 = st.sidebar.checkbox('demand by workforce market')

   st.sidebar.write('Factor to apply for MFU:') 
   v_Q23 = st.sidebar.checkbox('easy/convenient transportation')
   v_Q24 = st.sidebar.checkbox('suitable cost')
   v_Q25 = st.sidebar.checkbox('graduates with higher language/academic competency than other universities')
   v_Q26 = st.sidebar.checkbox('learning in English')
   v_Q27 = st.sidebar.checkbox('quality/reputation of university')
   v_Q28 = st.sidebar.checkbox('excellence in learning support and facilities')
   v_Q29 = st.sidebar.checkbox('provision of preferred major')
   v_Q30 = st.sidebar.checkbox('environment and setting motivate learning')
   v_Q31 = st.sidebar.checkbox('experienced and high-quality instructors')
   v_Q32 = st.sidebar.checkbox('suggestion by school teacher/friend/relative')
   v_Q33 = st.sidebar.checkbox('suggestion by family')

   st.sidebar.write('If your application fails, will you try again?') 
   v_Q34 = st.sidebar.checkbox('try the same major')
   v_Q35 = st.sidebar.checkbox('try a different major')
   v_Q36 = st.sidebar.checkbox('will not try again')   
    
   

   #change into num
   if v_Q1 == 1: v_Q1 = '1'
   else: v_Q1 = '0'
   if v_Q2 == 1: v_Q2 = '1'
   else: v_Q2 = '0'
   if v_Q3 == 1: v_Q3 = '1'
   else: v_Q3 = '0'
   if v_Q4 == 1: v_Q4 = '1'
   else: v_Q4 = '0'
   if v_Q5 == 1: v_Q5 = '1'
   else: v_Q5 = '0'
   if v_Q6 == 1: v_Q6 = '1'
   else: v_Q6 = '0'

   if v_Q23 == 1: v_Q23 = '1'
   else: v_Q23 = '0'
   if v_Q24 == 1: v_Q24 = '1'
   else: v_Q24 = '0'
   if v_Q25 == 1: v_Q25 = '1'
   else: v_Q25 = '0'
   if v_Q26 == 1: v_Q26 = '1'
   else: v_Q26 = '0'
   if v_Q27 == 1: v_Q27 = '1'
   else: v_Q27 = '0'
   if v_Q28 == 1: v_Q28 = '1'
   else: v_Q28 = '0'
   if v_Q29 == 1: v_Q29 = '1'
   else: v_Q29 = '0'
   if v_Q30 == 1: v_Q30 = '1'
   else: v_Q30 = '0'
   if v_Q31 == 1: v_Q31 = '1'
   else: v_Q31 = '0'
   if v_Q32 == 1: v_Q32 = '1'
   else: v_Q32 = '0'
   if v_Q33 == 1: v_Q33 = '1'
   else: v_Q33 = '0'

   if v_Q34 == 1: v_Q34 = '1'
   else: v_Q34 = '0'
   if v_Q35 == 1: v_Q35 = '1'
   else: v_Q35 = '0'
   if v_Q36 == 1: v_Q36 = '1'
   else: v_Q36 = '0'



   #dictionary
   data = {'FacultyName': v_Faculty,
            'GPAX': v_GPAX,
            'Q1': v_Q1,
            'Q2': v_Q2,
            'Q3': v_Q3,
            'Q4': v_Q4,
            'Q5': v_Q5,
            'Q6': v_Q6,
            'Q23': v_Q23,
            'Q24': v_Q24,
            'Q25': v_Q25,
            'Q26': v_Q26,
            'Q27': v_Q27,
            'Q28': v_Q28,
            'Q29': v_Q29,
            'Q30': v_Q30,
            'Q31': v_Q31,
            'Q32': v_Q32,
            'Q33': v_Q33,
            'Q34': v_Q34,
            'Q35': v_Q35,
            'Q36': v_Q36
            }

   #create data frame
   data_df = pd.DataFrame(data, index=[0]) 
   return data_df


st.write(""" **User Input :** """)

df = get_input()
st.write(df)


data_sample = pd.read_csv('test_TCAS.csv')
df = pd.concat([df, data_sample],axis=0)

# st.write(""" **Normalizationed Input :** """)

cat_data = pd.get_dummies(df[['FacultyName']])

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)

st.write(""" **Normalization Input :** """)
# Drop un-used feature
X_new = X_new.drop(columns=['FacultyName'])

st.write(X_new)

st.write(""" **Prediction :** """)

# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)