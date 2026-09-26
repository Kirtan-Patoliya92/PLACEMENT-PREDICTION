import streamlit as st
import joblib as jb
import pandas as pd

model = jb.load('model_placed.pkl')
columns = jb.load('placed_columns.pkl')
scaler = jb.load('placed_scaler.pkl')

st.title("Placement Prediction ML Model")

age = st.number_input('Enter your age :'.upper(),min_value=18,max_value=99)

ssc_percentage = st.number_input(
    'Enter your scc percentage '.upper(),
    min_value=0.00,
    max_value=100.00
)

hsc_percentage = st.number_input(
    'Enter your hsc percentage '.upper(),
    min_value=0.00,
    max_value=100.00
)

hsc_stream = st.selectbox(
    'Select your stream ',
    ['Commerce','Science','Arts']
)

list = ['Commerce','Science','Arts']

for i in range(len(list)):
    if hsc_stream == list[i]:
        hsc_stream = i


degree_percentage = st.number_input(
    'Enter your degree percentage'.upper(),
    max_value=100.00,
    min_value=0.00
)

degree_field = st.selectbox(
    'select your degree field',
    ['Engineering','Business','Other','Arts','Science']
)

list1 = ['Engineering','Business','Other','Arts','Science']

for i in range(len(list1)):
    if degree_field == list1[i]:
        degree_field = i

mba_percentage = st.number_input(
    'Enter your mba percentage',
    min_value=0.00,
    max_value=100.00
)                                                         

internships_count = st.selectbox(
    'select your intern sheep count'.upper(),
    [0,1,2,3,4,5]
)

projects_count = st.selectbox(
    'how many projects you have done '.upper(),
    [0,1,2,3,4,5,6,7,8,9,10]
)

certifications_count = st.selectbox(
    'how many cerificats achive in your oven life',
    [0,1,2,3,4,5,6,7,8]
                                                                        
)

technical_skills_score = st.number_input(
    'Enter your technical skills score give by your collage after completing degree',
    min_value=0.0,
    max_value=10.00
)

soft_skills_score = st.number_input(
    'Enter your soft skills score give by your collage after completing degree',
    min_value=0.00,
    max_value=10.00
)

aptitude_score = st.number_input(
    'Enter your aptitude score give by your collage after completing degree',
    min_value=0.00,
    max_value=100.00
)

communication_score = st.number_input(
   'Enter your communication score give by your collage after completing degree',    
    min_value=0.00,
    max_value=10.00
)

work_experience_months = st.number_input(
    'enter your work experience in months',
    min_value=0,
    max_value=40
)

leadership_roles = st.selectbox(
    'select how many time you have selected of any kind of leadership',
    [0,1,2,3,4,5]
)

extracurricular_activities = st.selectbox(
    'select how many times you are participeted in for any kind of extracurricular activities'.upper(),
    [0,1,2,3,4,5,6,7,8,9,10]
)

backlogs = st.number_input(
    'enter how many time to strking againts backss'.upper(),
    min_value=0,
    max_value=10
)

if st.button('PREDICT'):

    row_data = {
        'age' : age,
        'ssc_percentage' : ssc_percentage,
        'hsc_percentage' : hsc_percentage,
        'hsc_stream' : hsc_stream,
        'degree_percentage' : degree_percentage,
        'degree_field' : degree_field,
        'mba_percentage' : mba_percentage,
        'internships_count' : internships_count,
        'projects_count' : projects_count,
        'certifications_count' : certifications_count,
        'technical_skills_score' : technical_skills_score,
        'soft_skills_score' : soft_skills_score,
        'aptitude_score' : aptitude_score,
        'communication_score' : communication_score,
        'work_experience_months' : work_experience_months,
        'leadership_roles' : leadership_roles,
        'extracurricular_activities' : extracurricular_activities,
        'backlogs' : backlogs
    }

    final_data = pd.DataFrame([row_data])

    input_columns = [
        col for col in columns
        if col != 'placed'
    ]

    final_data = final_data.reindex(
        columns=input_columns,
        fill_value=0
    )

    final_data = final_data.astype(float)

    final_data = scaler.transform(final_data)

    prediction = model.predict(final_data)

    prediction = prediction[0]

    if prediction == 1:
        st.success("Congretulations you will have placed ".upper())

    elif prediction == 0:
        st.error("Sorry, i think you should be improuve your skill".upper())