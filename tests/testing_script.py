import httpx
import asyncio
import os
from pathlib import Path


url = "http://localhost:8000/predict"  
repo = Path.cwd()
relative = Path("Python_Engineer/test_images/")
images_path =  repo /  relative 
# how many time we send the execution
nb_executions = 6

def jpeg_list(chemin_dossier):
    
    fichiers_jpeg = []


    for fichier in os.listdir(chemin_dossier):

        if fichier.lower().endswith(('.jpeg', '.jpg')):
            fichiers_jpeg.append(fichier)

    return fichiers_jpeg



jpeg_files = jpeg_list(images_path)



async def test_script():

    images_path =  repo /  relative 
    jpeg_files = jpeg_list(images_path)
    files = [('files', (image_path, open(images_path / image_path, 'rb'), 'image/jpeg')) for image_path in jpeg_files]
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, files=files)

    return response.json()


async def run_tests_concurrently():
    
    tasks = [test_script() for _ in range(nb_executions)]
    results = await asyncio.gather(*tasks)
    return results

async def display_result(index, result):
    print(f"Test result {index}: {result}")

async def main():
    results = await run_tests_concurrently()
    result_tasks = [display_result(i, result) for i, result in enumerate(results, start=1)]
    await asyncio.gather(*result_tasks)

if __name__ == "__main__":
    asyncio.run(main())