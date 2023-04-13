import boto3
import json
import random
import string
from decimal import Decimal

import cv2
import numpy as np
from timeit import default_timer as timer

s3_client = boto3.client('s3')
dynamodb_client = boto3.resource('dynamodb')
# Save the filtered image to another S3 bucket
output_bucket = 'eq-preprocessed-test-images'
table = dynamodb_client.Table('preprocessing_metrics')


def generate_random_string(length):
    # Define the possible characters to use in the string
    characters = string.ascii_letters + string.digits
    
    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for i in range(length))
    
    return random_string


def upload_preprocessed_image(preprocessed_image, file_name, random_string, operation_performed, steps):
    ''' Upload the preprocessed image to another bucket'''
    filename, ext = file_name.split('.')
    output_filename = f"{filename}_{random_string}/{operation_performed}_{steps}.{ext}"
    success, buffer = cv2.imencode('.jpg', preprocessed_image)
    filtered_image_bytes = buffer.tobytes()
    s3_client.put_object(Bucket=output_bucket, Key=output_filename, Body=filtered_image_bytes)
    # Get the URL of the filtered image
    location = s3_client.get_bucket_location(Bucket=output_bucket)['LocationConstraint']
    preprocessed_image_path = "https://%s.s3.amazonaws.com/%s" % (output_bucket, output_filename)
    print("Filtered image saved to S3 bucket:", output_bucket, "at key:", output_filename)
    print("Filtered image URL:", preprocessed_image_path)

    return str(preprocessed_image_path)


def scale_image(image, file_name, random_string, preprocessing_metrics):
    start_time = timer()  # Record the start time of the processing

    # Apply scaling steps from 10 to 100 with steps of 20
    for scale_percent in range(10, 100, 20):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        scaled_image = cv2.resize(image, (width, height))
        preprocessed_image_path = upload_preprocessed_image(scaled_image, file_name, random_string, 'scaled', scale_percent)

        preprocessing_metrics[f'scaled_{scale_percent}_path'] = preprocessed_image_path

    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['scaling_operation_time'] = processing_time


def blur_image(image, file_name, random_string, preprocessing_metrics):
    start_time = timer()  # Record the start time of the processing

    # Apply blurring steps from 10 to 100 with steps of 20
    for blur_step in range(1, 20, 4):
        # Apply the normalized box filter with kernel size equal to blur_step
        blurred_image = cv2.boxFilter(image, -1, (blur_step, blur_step), normalize=True)
        preprocessed_image_path = upload_preprocessed_image(blurred_image, file_name, random_string, 'blurred', blur_step)

        preprocessing_metrics[f'blurred_{blur_step}_path'] = preprocessed_image_path
    
    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['bluring_operation_time'] = processing_time


def adjust_brightness(image, file_name, random_string, preprocessing_metrics):
    start_time = timer()  # Record the start time of the processing
   
    # Define gamma correction values to use
    gamma_values = [0.25, 1.5, 3.0]

    # Apply gamma correction using each value and save the resulting images
    for gamma in gamma_values:
        # Calculate the gamma correction lookup table
        gamma_inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma_inv) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # Apply the gamma correction to the image
        gamma_corrected = cv2.LUT(image, table)
        preprocessed_image_path = upload_preprocessed_image(gamma_corrected, file_name, random_string, 'brightness', gamma)

        preprocessing_metrics[f'brightness_{gamma}_path'] = preprocessed_image_path
    
    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['brightness_adjustment_time'] = processing_time


def adjust_salt_and_pepper_noise(image, file_name, random_string, preprocessing_metrics):
    start_time = timer()  # Record the start time of the processing

    # Apply salt and pepper noise to the image with different probabilities
    for value in range(1, 10, 2):
        noise_probability = value * 0.1
        noisy_image = np.copy(image)
        height, width, channels = image.shape
        noise_mask = np.random.choice([0, 1, 2], size=(height, width),
                                       p=[1 - noise_probability, noise_probability / 2, noise_probability / 2])
        noisy_image[noise_mask == 1] = [255, 255, 255]  # Salt noise
        noisy_image[noise_mask == 2] = [0, 0, 0]  # Pepper noise

        preprocessed_image_path = upload_preprocessed_image(noisy_image, file_name, random_string, 'noise', value)

        preprocessing_metrics[f'noise_{value}_path'] = preprocessed_image_path
    
    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['noise_adjustment_time'] = processing_time


def preprocess_image(event, context):
    start_time = timer()  # Record the start time of the processing

    preprocessing_metrics = {}

    # Get the S3 bucket and key of the uploaded image
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_name = event['Records'][0]['s3']['object']['key']
    
    # Get the uploaded image from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    img = response['Body'].read()

    # blob = storage_client.bucket(bucket_name).get_blob(file_name)
    # file_name = blob.name
    # _, temp_local_filename = tempfile.mkstemp()
    # print(f"local filenmame: {temp_local_filename}")

    # blob.download_to_filename(temp_local_filename)
    # print(f'Image {file_name} was downloaded to {temp_local_filename}.')

    image = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)

    # Generate random string to be used for unique identity for images
    random_string = generate_random_string(8)

    # Scaling operations
    scale_image(image, file_name, random_string, preprocessing_metrics)

    # Bluring Operations
    blur_image(image, file_name, random_string, preprocessing_metrics)

    # Brightness Adjustment Operations
    adjust_brightness(image, file_name, random_string, preprocessing_metrics)

    # Noise Adjustment Operations
    adjust_salt_and_pepper_noise(image, file_name, random_string, preprocessing_metrics)

    # os.remove(temp_local_filename)

    end_time = timer()  # Record the end time of the processing
    processing_time = end_time - start_time
    preprocessing_metrics['total_preprocessing_time'] = processing_time
    
    db_id = f"{file_name.split('.')[0]}_{random_string}"
    preprocessing_metrics['id'] = db_id

    # Write the preprocessing_metrics to DynamoDB
    converted_preprocessing_metrics = json.loads(json.dumps(preprocessing_metrics), parse_float=Decimal)
    table.put_item(Item=converted_preprocessing_metrics)
