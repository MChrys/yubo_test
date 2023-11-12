from fastapi import File, FastAPI, UploadFile
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
app= FastAPI()


@app.post("predict/")
async def predict(images:List[UploadFile]= File(...)):
    """
    Predicts categories for a list of uploaded images.

    This route accepts multiple images in JPEG format and returns their predicted categories.

    """

    async_process = [asyncio.create_task(process(image,num)) for num,image in enumerate(images)]
    return await gather(*async_process)

async def process(image: UploadFile ,num:int):
    """
    Processes an uploaded image and predicts its category.

    :param image: The image to be processed.
    :param index: The index of the image in the list of uploaded files.
    :return: The predicted category for the image.
    """
    logger.info(f"Starting image {num} processing ")
    filename = image
    shape = (224, 224)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    im = Image.open(filename).convert("RGB") 
    im = im.resize(shape)
    im = np.array(im, dtype=np.float32)
    im /= 255.
    im -= mean
    im /= std
    im = np.transpose(im, (2, 0, 1))
    data = [im.tolist()]
    logger.info(f"image {num} : requesting tensorflow serving")
    async with httpx.AsyncClient() as client:
        response = await client.post(serving_url, json=data, headers=header)
    logger.info(f"image {num} : get response tensorflow serving")        
    logger.info(f"Ending image {num} processing ")
    return response

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