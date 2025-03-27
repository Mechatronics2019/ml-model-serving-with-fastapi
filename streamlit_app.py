import streamlit as st
import requests
from scripts import s3


### Define the API endpoint
API_URL = "http://127.0.0.1:8000/api/v1/"
headers = {
    "Content-Type": 'application/json'
}

st.title("ML Model Serving Over REST API")


model = st.selectbox(label="Select Model",
                     options=["Sentiment Classifier", "Disaster Classifier", "Pose Classifer"])

if model == "Sentiment Classifier":
    text = st.text_area("Enter your movie Review.")
    user_id = st.text_input("Enter user id",value="udemy@udemy.com")

    data = {
        "text" : [text],
        "user_id" : user_id
        }
    
    url = API_URL + "sentiment_analysis"

elif model == "Disaster Classifier":
    text = text = st.text_area("Enter your Tweet.")
    user_id = st.text_input("Enter user id",value="udemy@udemy.com")

    data = {
        "text" : [text],
        "user_id" : user_id
        }

    url = API_URL + "disaster_classifier"


elif model == "Pose Classifer":
    select_file = st.radio("Select the image source",["Local","URL"])

    if select_file == "URL":
        url = st.text_input("Enter your Image URL")
    else:
        image = st.file_uploader("Upload the image",type=["jpg","jpeg","png"])
        file_name = "images/temp.jpg"

        if image is not None:
            with open(file_name,"wb") as f:
                f.write(image.read())
        url = file_name
        # url = s3.upload_image_to_s3(file_name=file_name)

    user_id = st.text_input("Enter user id",value="udemy@udemy.com")

    data = {
        "url" : [url],
        "user_id" : user_id
        }
    
    url = API_URL + "pose_classifier"

if st.button("Predict"):
    with st.spinner("Predicting... Please wait!!!"):
        response = requests.post(url= url,
                                 json=data)
        
        output = response.json()

    st.write(output)