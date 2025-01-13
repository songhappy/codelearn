from transformers import pipeline
import requests
import urllib.parse as urlparse
from PIL import Image
from requests import Response

import os

classifier = pipeline('sentiment-analysis')
# classified = classifier('We are very happy to introduce pipeline to the transformers repository.')
# print(classified)


url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
# image_data = requests.get(url, stream=True).raw
# image = Image.open(image_data)
# image.show()
# object_detector = pipeline('object-detection')
# detected = object_detector(image)
# print(detected)


from datasets import load_dataset
