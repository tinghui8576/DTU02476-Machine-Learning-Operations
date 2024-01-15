from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from http import HTTPStatus
from typing import Optional

import cv2

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}
app = FastAPI()

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
    """Simple function using open-cv to resize an image."""
    with open("image.jpg", "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    img = cv2.imread("image.jpg")

    response = {
        "input": data,
        "output": predict_step(img),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def predict_step(img):

    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds
