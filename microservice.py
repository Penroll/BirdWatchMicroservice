import io
import os

import torch
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='bird_image_classifier.pt').cpu()
model.eval()


def delete_image(file_path: str):
    try:
        if os.path.exists(file_path):  # Check if the file exists
            os.remove(file_path)  # Delete the file
            print(f"File {file_path} has been deleted.")
        else:
            print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"Error deleting file: {str(e)}")


# Function to save the received image
def save_image(input_image: Image):
    # Save the received image to received_images/temp.jpg
    output_dir = "received_images"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(output_dir, f"received_image_{timestamp}.jpg")
    input_image.save(file_path)
    print(f"Saved image to {file_path}")

    # Return the path to the image for the model to use, and later for deletion
    return file_path


def make_prediction(input_image: Image):
    # Save image as a file (Model only reads filenames)
    input_filepath = save_image(input_image)
    # Run prediction on the image
    prediction = model(input_filepath)

    # Isolate the detected classes and their probabilities
    results_df = prediction.pandas().xyxy[0]
    detections = [
        {"class": row["name"], "confidence": round(row["confidence"], 2)}
        for _, row in results_df.iterrows()
    ]

    # Delete the original image to preserve storage space
    delete_image(input_filepath)

    return detections


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image data
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        # Make the prediction, receive detected birds in image
        detections = make_prediction(input_image)

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
