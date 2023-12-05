import os
import shutil

from minio import Minio
from dotenv import load_dotenv

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
BUCKET_NAME = 'knowledge'

load_dotenv()

minio_endpoint = os.getenv('MINIO_ENDPOINT')
minio_access_key = os.getenv('ACCESS_KEY')
minio_secret_key = os.getenv('SECRET_KEY')

client = Minio(endpoint=minio_endpoint,
                   secret_key=minio_secret_key,
                   access_key=minio_access_key,
                   secure=False)

def delete_data():
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)

def get_file():
    objects = client.list_objects(BUCKET_NAME)
    for obj in objects:
        pdf_path = os.path.join(DATA_PATH, obj.object_name)
        doc = client.fget_object(bucket_name=BUCKET_NAME, object_name=obj.object_name, file_path=pdf_path)
