from fastapi.testclient import TestClient
from yubo_test.main import app  
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = TestClient(app)

def test_ping_api():
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "Test successful"}

def test_upload_file():
    repo = Path.cwd()
    relative = Path("Python_Engineer/test_images/")
    images_path =  repo /  relative 
    request = {"files": ( "1.jpeg", open(images_path / "1.jpeg", "rb"), "image/jpeg")}
    response = client.post(
        "/predict",
        files=request
    )
    logger.info(response.json())
    assert response.status_code == 200