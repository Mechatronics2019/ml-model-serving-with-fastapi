import warnings
warnings.filterwarnings("ignore")

from scripts.data_model import NLPDataInput, NLPDataOutput, ImageDataInput, ImageDataOutput
from scripts import s3

import os
import torch
from transformers import pipeline
from transformers import AutoImageProcessor

from fastapi import FastAPI
import uvicorn
import time

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

## Loading Image Processor for FineTuned Vit Model
model_image_processor = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_image_processor, use_fast=True)

############# Download ML Models #############

force_download = False ## True

models = ["tinybert-sentiment-analysis/", "tinybert-disaster-tweet/", "vit-human-pose-classification"]
LOCAL_PATH = "ml-models/"
for model_name in models:
    local_path = LOCAL_PATH + model_name
    if not os.path.isdir(local_path) or force_download:
        s3.download_dir(local_path=local_path, model_name=model_name)
    
############# Download Ends Here ##############



############# Loading Models ##################

sentiment_model = pipeline("text-classification", model=LOCAL_PATH + models[0], device = device)
twitter_model = pipeline("text-classification", model=LOCAL_PATH + models[1], device = device)
pose_model = pipeline("image-classification", model=LOCAL_PATH + models[2], device = device, image_processor=image_processor)

############ Loading Ends Here ###############


@app.get("/")
def read_root():
    return "Hello, I'm up."

@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput):

    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int((end - start)*1000)

    labels = [prediction['label'] for prediction in output]
    scores = [prediction['score'] for prediction in output]

    output = NLPDataOutput(
        model_name = "tinybert-sentiment-analysis",
        text = data.text,
        labels = labels,
        scores = scores,
        prediction_time = prediction_time
    )
    return output


@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = twitter_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [prediction['label'] for prediction in output]
    scores = [prediction['score'] for prediction in output]

    output = NLPDataOutput(
        model_name = "tinybert-disaster-tweet",
        text = data.text,
        labels = labels,
        scores = scores,
        prediction_time = prediction_time
    )
    return output


@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput):
    start = time.time()
    urls = [str(url) for url in data.url]
    output = pose_model(urls)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [prediction[0]['label'] for prediction in output]
    scores = [prediction[0]['score'] for prediction in output]

    output = ImageDataOutput(
        model_name = "vit-human-pose-classification",
        url = data.url,
        labels = labels,
        scores = scores,
        prediction_time = prediction_time
    )

    return output



if __name__=="__main__":
    uvicorn.run(app ="app:app", port=8000, reload=True)