import os
from google.cloud import storage, vision


def detect_text(event, context):
    storage_client = storage.Client()
    vision_client = vision.ImageAnnotatorClient()

    # Get the image file name from the event
    file_name = event['name']
    bucket_name = event['bucket']
    print(f"Processing file: {file_name} {bucket_name}")

    blob_uri = f"gs://{bucket_name}/{file_name}"
    blob_source = vision.Image(source=vision.ImageSource(image_uri=blob_uri))

    # Perform text detection on the image
    response = vision_client.text_detection(image=blob_source)
    text_annotations = response.text_annotations

    # Extract the detected text from the response
    detected_text = ""
    for text_annotation in text_annotations:
        detected_text += text_annotation.description + "\n"

    # TODO: Save the response to Google Cloud Firestore or Datastore

    print(f"Detected text: {detected_text}")

