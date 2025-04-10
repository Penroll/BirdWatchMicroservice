# BirdWatchMicroservice
Python Microservice that recieves an image from server, and returns the identified birds' Species (Sometimes Male/Female distinctions) and the confidence. 

## Model
This model was trained on the YOLOv5s model, with a dataset from Cornell University. If you want to look into training a similar model, perhaps on a more recent version of the YOLO models, check out Quentin Young's repository for how he preprocessed the images from Cornell: https://github.com/qlyoung/yolobirds

## Running the Service
To run this service locally, I use uvicorn, and start from the command line using this command:
uvicorn microservice:app --host 0.0.0.0 --port 8000
