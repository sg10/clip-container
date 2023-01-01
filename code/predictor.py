import base64
import io
import json
import logging
import logging.config
import os
import shutil
import uuid
from urllib.parse import urlparse

import numpy as np
import open_clip
import requests
import torch
import transformers
from PIL import Image
from flask import Flask, request, Response
from multilingual_clip import pt_multilingual_clip

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"


model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
# Load Model & Tokenizer
model_text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name, device=device)
preprocess_text = transformers.AutoTokenizer.from_pretrained(model_name)

model_image_encoder, _, preprocess_image = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model_image_encoder.to(device)



app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# gunicorn_error_logger = logging.getLogger('gunicorn.error')
# app.logger.handlers.extend(gunicorn_error_logger.handlers)
# app.logger.setLevel(logging.DEBUG)

# check if inside docker container
def is_inside_docker():
    return os.path.exists("/.dockerenv")

PREFIX_PATH = "/opt/ml/" if is_inside_docker() else "/tmp/clipservice/"

IMAGES_FOLDER = os.path.join(PREFIX_PATH, "images")
os.makedirs(IMAGES_FOLDER, exist_ok=True)


def download_image(request_id, image):
    def download_file_from_url(folder, url):
        filename = os.path.join(folder, os.path.basename(urlparse(url).path))
        try:
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)

            return filename
        except Exception:
            return None

    logging.info(f'Downloading image "{image}"...')

    folder = os.path.join(IMAGES_FOLDER, request_id)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    fragments = urlparse(image, allow_fragments=False)
    if fragments.scheme in ("http", "https"):
        filename = download_file_from_url(folder, image)
    else:
        filename = image

    if filename is None:
        raise Exception(f"There was an error downloading image {image}")

    return Image.open(filename).convert("RGB")


def delete_images(request_id):
    directory = os.path.join(IMAGES_FOLDER, request_id)

    try:
        shutil.rmtree(directory)
    except OSError as e:
        logging.error(f"Error deleting image directory {directory}.")


def predict(texts, images):
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'

    # Load Model & Tokenizer
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    text_embeddings = model.forward(texts, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    model.to(device)

    images = [
        preprocess(image).unsqueeze(0).to(device)
        for image in images
    ]
    # stack all images in a single tensor
    images = torch.cat(images, dim=0)

    with torch.no_grad():
        image_features = model.encode_image(images)

    # normalize features
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # compute similarity
    similarity = (100.0 * image_features @ text_embeddings.T).softmax(dim=-1).tolist()

    result = []

    top_k = 3

    """ structure: [
          [
              <top 3 text label indices>, 
              <top 3 text original labels>,
              <top 3 text label scores>
          ],
          [
              ...
          ]   
    ]        
    """

    for i, image in enumerate(images):
        result.append([
            np.argsort(similarity[i])[::-1].tolist()[:top_k],
            [texts[j] for j in np.argsort(similarity[i])[::-1]][:top_k],
            np.sort(similarity[i])[::-1].tolist()[:top_k]
        ])

    return result


@app.route("/ping", methods=["GET"])
def ping():
    """This endpoint determines whether the container is working and healthy."""
    logging.info("Ping received...")

    health = True

    status = 200 if health else 404
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invoke():
    if request.content_type != "application/json":
        return Response(
            response='{"reason" : "Request should be application/json"}',
            status=400,
            mimetype="application/json",
        )

    request_id = uuid.uuid4().hex

    data = request.get_json()

    images = []
    for im in data["images"]:
        fragments = urlparse(im, allow_fragments=False)
        if fragments.scheme in ("http", "https", "file"):
            image = download_image(request_id, im)
        else:
            image = Image.open(io.BytesIO(base64.b64decode(im)))

        images.append(image)

    result = predict(texts=data["classes"], images=images)

    delete_images(request_id=request_id)

    return Response(
        response=json.dumps(result),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
