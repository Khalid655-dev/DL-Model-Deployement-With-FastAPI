from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import numpy as np
from pathlib import Path
from starlette.staticfiles import StaticFiles



# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from werkzeug.utils import secure_filename


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define a FastAPI app
app = FastAPI(title="Tomato Leaf Disease Detection")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model saved with Keras model.save()
MODEL_PATH =  os.path.join(BASE_DIR, 'ReTrained98.h5')

# Load your trained model
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)


def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    label = np.argmax(preds,axis=1)
    preds = label[0]
    if preds == 0:
        preds = "Tomato___Bacterial_spot"
    if preds == 1:
        preds = "Tomato___Early_blight"
    if preds == 2:
        preds = "Tomato___Late_blight"
    if preds == 3:
        preds = "Tomato___Leaf_Mold"
    if preds == 4:
        preds = "Tomato___Septoria_leaf_spot"
    if preds == 5:
        preds = "Tomato___Spider_mites Two-spotted_spider_mite"
    if preds == 6:
        preds = "Tomato___Target_Spot"
    if preds == 7:
        preds = "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    if preds == 8:
        preds = "Tomato___mosaic_virus"
    elif preds == 9:
        preds = "Tomato__healthy"   
    return preds


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    # Main page
    return templates.TemplateResponse('index.html', {"request": request})


@app.post('/predict')
def upload(file: UploadFile = File(description="A file read as UploadFile")):
   
        import shutil


        if os.path.exists('uploads'):
            directory = "uploads"

            # Path
            path = os.path.join(os.path.dirname(__file__), directory)

            # Remove the Directory
            shutil.rmtree(path)

        f = file

        basepath = os.path.dirname(__file__)
        if not os.path.exists('uploads'):
            os.mkdir('uploads')
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        path: Path = Path(file_path)
        try:
            with path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        finally:
            file.close()
        # Prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
   
