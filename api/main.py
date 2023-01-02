from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn 
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model('D:\\Coding\\jupyter\\PlantDiseaseDetection\\models\\1')
CLASS_NAMES = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Septoria_leaf_spot', 'Tomato__Target_Spot']

@app.get("/ping")
async def ping():
  return "Hello, from server"

def read_file_as_image(data) -> np.ndarray:
  image = np.array(Image.open(BytesIO(data)))
  return image

@app.post("/predict")
async def predict(
  file: UploadFile = File(...)
):
  image = read_file_as_image(await file.read())
  img_batch = np.expand_dims(image, 0)

  prediction = MODEL.predict(img_batch)

  predicted_class = CLASS_NAMES[np.argmax(prediction[0])]

  confidence = np.max(prediction[0])
  
  return {
    'class': predicted_class,
    'confidence': float(confidence)
  }
  

if __name__ == "__main__":
  uvicorn.run(app, host='localhost', port=8080)