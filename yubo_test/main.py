from fastapi import File, FastAPI, UploadFile, HTTPException
import json
import numpy as np 
from PIL import Image
import requests
import asyncio
from asyncio import gather
import logging
import io
import httpx
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

serving_url = "http://localhost:8501/v1/models/model:predict"
header = {'Content-Type': 'application/json'}
categories = 'Python_Engineer/categories_places365.txt'
app= FastAPI()


@app.post("/predict")
async def predict(files:List[UploadFile]= File(...)):
    """
    Predicts categories for a list of uploaded images.

    This route accepts multiple images in JPEG format and returns their predicted categories.

    """
    logger.info(f"start predicting")
    async_process = []
    index =0
    for image in files:
        if image.content_type != "image/jpeg":
            
            logger.info(f"image content {image.filename} : {image.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Wrong format: {image.filename} should be formatted in JPEG"
                )
        async_process.append(asyncio.create_task(process(image,index+1)))
        index +=1
    return await gather(*async_process)

async def process(image: UploadFile ,index:int):
    """
    Processes an uploaded image and predicts its category.

    :param image: The image to be processed.
    :param index: The index of the image in the list of uploaded files.
    :return: The predicted category for the image.
    """
    logger.info(f"Starting image {index} processing ")
    #filename = image
    shape = (224, 224)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    im = Image.open(io.BytesIO(await image.read())).convert("RGB") 
    im = im.resize(shape)
    im = np.array(im, dtype=np.float32)
    im /= 255.
    im -= mean
    im /= std
    im = np.transpose(im, (2, 0, 1))
    data = [im.tolist()]
    request = {"inputs":data}
    logger.info(f"image {image.filename} : requesting tensorflow serving")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(serving_url, json=request, headers=header)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"Tensorflow serving response error  for image {image.filename}: {e}")
            return {f"serving_error_{index}": str(e)}
        except httpx.RequestError as e:
            logger.error(f"Tensorflow serving request error  for image {image.filename}: {e}")
            return {f"serving_error_{index}": str(e)}

    logger.info(f"image {index} : get response tensorflow serving") 
    vector = response.json()["outputs"][0]
    category = get_category(vector,index)
    logger.info(f"image {image.filename} : get category -> {category}") 
    logger.info(f"Ending image {image.filename} processing ")
    with open(f"response/response_{image.filename.split(".jpeg")[0]}.json", 'w') as fichier:
        json.dump(response.json(), fichier, indent=4)
    return {f"{image.filename}":category}

def get_category(vector:List[float],index)->str:
    """
    Determines the category of an image from a vector of scores.

    :param vector: List of prediction scores for the image.
    :param index: The index of the image in the list of uploaded files.
    :return: The category of the image determined from the prediction scores.
    """
    indice_max = 0
    val_max = vector[0]

    for n, val in enumerate(vector):
        try:
            if val > val_max:
                val_max = val
                indice_max = n
        except Exception as e:
            logger.error(f"Error when trying to getback the category of the image {index}: {e}")
            return f"Category_Error: {e}"
            
    with open(categories, 'r') as file:
        for i, ligne in enumerate(file, start=1):
            if i == indice_max +1:
                category = ligne.strip()
    return category

@app.get("/test")
def test_endpoint():
    """
    Test the endpoint of the API
    """
    logger.info(f"ping")
    return {"message": "Test successful"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)