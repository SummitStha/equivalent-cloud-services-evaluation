import json
import os
from timeit import default_timer as timer

# from google.cloud import storage, vision, datastore, firestore
from google.cloud import storage, vision, firestore

validation_dataset = {
    '1.jfif': 'PureMichigan DNJ 0955',
    '2.jfif': 'EXP:040917',
    '3.jfif': 'Exp. date: 02-2023',
    '4.jfif': 'EXP 06.10.2016',
    '7.jfif': 'Illinois 977 4224',
    '10.jfif': 'California 5XOR829',
    '11.jfif': '2120 MIDLAKE DRIVE',
    '12.jfif': '2190 ORCHARD RIDGE',
    '13.jfif': "1027 Brooks Road The Oliver's",
    '14.jfif': '4727 NORTH MONTANA AVENUE THE BRISTOWS',
    '15.jfif': '6768 BLUE LAKE ROAD',
    '16.jpeg': 'Thank you.',
    '17.jpeg': 'I am Really sorry.',
    '18.jpeg': 'you are beautiful',
    '19.jpeg': 'Get well soon.',
    '20.jpeg': 'Eid mubarak',
    '23.jfif': 'Kentucky 552 WDN',
    '28.jfif': 'Indiana 825ZYJ',
    '31.jpg': 'ONE WAY',
    '32.jpg': 'SPEED LIMIT 30',
    '33.jpg': 'STOP',
    '34.jpg': 'ROAD CLOSED',
    '50.jfif': '2120 MIDLAKE DRIVE'
    }


def detect_text(event, context):
    storage_client = storage.Client()
    vision_client = vision.ImageAnnotatorClient()
    # datastore_client = datastore.Client()
    firestore_client = firestore.Client()

    # Get the image file name from the event
    file_name = event['name']
    bucket_name = event['bucket']
    print(f"Processing file: {file_name} {bucket_name}")

    blob_uri = f"gs://{bucket_name}/{file_name}"
    blob_source = vision.Image(source=vision.ImageSource(image_uri=blob_uri))

    # Perform text detection on the image
    start_time = timer()
    response_text_annotations = vision_client.text_detection(image=blob_source).text_annotations
    end_time = timer()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time}')
    text_annotations = [text.description.lower() for text in response_text_annotations]

    detected = False
    splitted_filename = file_name.split('/')
    exact_filename = splitted_filename[0].replace(f"_{splitted_filename[0].split('_')[-1]}", "") + '.' + splitted_filename[-1].split('.')[-1]
    check_str = ''
    print('EXXCCTTCT: ', exact_filename)
    print('TEXTTT ANNNOOTTATIONS :', text_annotations)
    if validation_dataset.get(exact_filename):
        ground_truth = validation_dataset[exact_filename].lower()
        print('GROUNNDDDD: ', ground_truth)
        if ground_truth in text_annotations:
            detected = True
        elif set(ground_truth.strip().split()).issubset(set(text_annotations)):
            detected = True
        else:
            for text in text_annotations:
                if text in ground_truth:
                    check_str += f' {text}'
            print('CHECKK STTRR: ', check_str)
            if (check_str != '') and (set(check_str.strip().split()) == set(ground_truth.strip().split())):
                detected = True

    results = {'id': file_name, 'image_id': splitted_filename[0], 'image_uri': f'https://storage.googleapis.com/{bucket_name}/{file_name}', 'execution_time': execution_time, 'result': text_annotations, 'detected': detected}
    doc_ref = firestore_client.collection('cloud_vision_results').document(file_name.replace('/', '_'))
    doc_ref.set(results)

