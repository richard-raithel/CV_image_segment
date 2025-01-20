import base64
import csv
import io
from pathlib import Path
import cv2
from django.core.paginator import Paginator
from django.shortcuts import render, redirect
from django.urls import reverse
from django.core.cache import cache
from django.http import StreamingHttpResponse
from django.http.response import JsonResponse, HttpResponse, FileResponse
from django.views.decorators import gzip
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from PIL import Image
from pymongo import MongoClient
import bson
from Utility.onnx_inference1 import get_chip_prediction, get_bearing_prediction
from .forms import RegistrationForm
import numpy as np
import json
import os
from types import SimpleNamespace
import sys
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.cosmos import PartitionKey, CosmosClient
from pymongo import ASCENDING
from scipy.sparse import csr_matrix, save_npz, load_npz
import time
from django.conf import settings
from jinja2 import Environment, FileSystemLoader
from datetime import date
from docxtpl import DocxTemplate
# from dotenv import load_dotenv


# load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
snapshot = False
img = np.zeros((500, 500, 1), dtype="uint8")
np_arr = np.zeros((500, 500, 1), dtype="uint8")
RESOURCE_NAME = 'computervisionimages'


def to_data_uri(img_buf):
    img_64 = base64.b64encode(img_buf.getvalue())
    uri = img_64.decode('utf-8')
    img_buf.close()
    return f'data:image/png;base64,{uri}'


def to_pil_img(img_str):
    search_for = 'data:image/png;base64,'
    img_str = img_str.replace(search_for, '')

    if "==" in img_str[-2:]:
        img_bytes = base64.b64decode(img_str)
        pil_img = Image.open(io.BytesIO(img_bytes))
    elif "=" in img_str[-2:]:
        img_bytes = base64.b64decode(img_str + "=")
        pil_img = Image.open(io.BytesIO(img_bytes))
    else:
        img_bytes = base64.b64decode(img_str + "==")
        pil_img = Image.open(io.BytesIO(img_bytes))

    return pil_img.convert("RGB")

def pil_to_buffer(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="png")
    pil_image.close()
    return buffer


def UserLoggedIn(request):
    if request.user.is_authenticated == True:
        username = request.user
    else:
        username = None
    return username


def get_cosmos_connection():
    DB_NAME = 'computervisionimages-results'
    cosmos_client = MongoClient(
        os.getenv("COSMOS_URI")
    )

    cosmos_db = cosmos_client[DB_NAME]

    if DB_NAME not in cosmos_client.list_database_names():
        cosmos_db.command({"customAction": "CreateDatabase", "offerThroughput": 400})
        print(f"Created db {DB_NAME} with shared throughput.")
    else:
        print(f"Using database: {DB_NAME}")

    return cosmos_db


def get_blob_connection():
    # # USING MANAGED IDENTITIES
    # account_url = f"https://{RESOURCE_NAME}.blob.core.windows.net/"
    # default_credential = DefaultAzureCredential()
    # Create the BlobServiceClient object
    # blob_service_client = BlobServiceClient(account_url, credential=default_credential)

    # USING CONNECTION STRING
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=os.getenv("BLOB_CONNECTION_STRING"))

    return blob_service_client


# Can convert the masks to base64 to save some additional space
def numpy_array_to_base64(array):
    return base64.b64encode(array.tobytes()).decode('utf-8')


# Can convert the masks to base64 to save some additional space
def base64_to_numpy_array(base64_string, dtype, shape):
    return np.frombuffer(base64.b64decode(base64_string), dtype=dtype).reshape(shape)


def estimate_document_size(document):
    return len(bson.BSON.encode(document)) / (1024*1024)


# Convert the mask to scipy sparse matrix and then save the output to a dictionary for use in mongdb
def convert_mask_to_sparse(mask):
    sparse_matrix = csr_matrix(mask)
    return {
        'shape': list(sparse_matrix.shape),
        'data': sparse_matrix.data.tolist(),
        'indices': sparse_matrix.indices.tolist(),
        'indptr': sparse_matrix.indptr.tolist()
    }


# Convert the loaded mask dictionary holding the sparse matrix details
def convert_sparse_to_mask(mask_dict):
    shape = tuple(mask_dict['shape'])
    data = np.array(mask_dict['data'])
    indices = np.array(mask_dict['indices'])
    indptr = np.array(mask_dict['indptr'])

    return csr_matrix((data, indices, indptr), shape)


def push_to_azure(img, outputs, model, filename, username, metrics):

    blob_name = os.path.splitext(filename)
    blob_name = blob_name[0] + time.strftime("_%Y%m%d_%H%M%S") + blob_name[-1]

    blob_service_client = get_blob_connection()
    container_name = str(username).replace(".", "").lower()
    existing_containers = blob_service_client.list_containers()
    container_names = [container.get('name') for container in list(existing_containers)]
    if container_name in container_names:
        pass
    else:
        try:
            blob_service_client.create_container(container_name)
        except Exception as err:
            print(err)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    tags = {
        "formulation": str(metrics.get("Formulation")),
        "concentration": str(metrics.get("Concentration")),
        "userrating": str(metrics.get("UserRating")),
        "model": str(model),
        "modelrating": str(metrics.get("Rating")),
        "modelpercentage": str(metrics.get("Percentage")),
        "batchid": str(metrics.get("BatchID"))
    }

    try:
        img.seek(0)
        blob_client.upload_blob(img)
    except Exception as err:
        print(str(err) + "---- or the Blob already exists.")

    try:
        blob_client.set_blob_metadata(tags)
    except Exception as err:
        print(str(err) + "--- the tags couldn't be set")

    model_masks = outputs['masks']
    sparse_masks = [convert_mask_to_sparse(mask.squeeze()) for mask in model_masks]

    outputs['masks'] = sparse_masks

    cosmos_metadata = {
        'blob': blob_name,
        'model': str(model),
    }

    cosmos_metadata = cosmos_metadata | outputs | metrics

    try:
        cosmos_db = get_cosmos_connection()
    except Exception as err:
        cosmos_db = None
        print("No Cosmos Connection Setup")
        print(err)

    if cosmos_db is not None:
        all_collections = cosmos_db.list_collection_names()
        if container_name not in all_collections:
            # Create a collection based on the username logged in
            cosmos_db.command({"customAction": "CreateCollection", "collection": container_name})
            # try:
            #     # Update new collection to incorporate two types of index
            #     index = [
            #         {"key": {"_id": 1}, "name": "_id_1"},
            #         {"key": {"blob": 2}, "blob": "_id_2"}
            #     ]
            #     cosmos_db.command({"customAction": "UpdateCollection", "collection": container_name, "indexes": index})
            #     print(f"Created Collection {container_name}.")
            # except Exception as err:
            #     print("Could not add indices to Collection due to: ", err)

            # Connect to collection
            cosmos_collection = cosmos_db[container_name]
        else:
            cosmos_collection = cosmos_db[container_name]
            print(f"Using Collection {container_name}.")

        try:
            cosmos_collection.insert_one(cosmos_metadata)
        except Exception as err:
            print(f"Unable to upload model results due to:\n {err}")


def get_blob_cosmos_metadata(blob_name, cosmos_collection):
    allProductsQuery = {"blob": blob_name}
    doc = cosmos_collection.find(allProductsQuery)
    if doc:
        try:
            metadata = doc[0]
        except Exception as err:
            print(f"No Metadata found for blob ({blob_name}) due to error:", err)
            metadata = dict(blob=blob_name)
    else:
        metadata = dict(blob=blob_name)

    return metadata


def convert_mask_to_transparent_color(mask_pil_image):
    width = mask_pil_image.size[0]
    height = mask_pil_image.size[1]
    for i in range(0, width):  # process all pixels
        for j in range(0, height):
            data = mask_pil_image.getpixel((i, j))
            if data[:3] == (255, 255, 255):
                mask_pil_image.putpixel((i, j), (37, 134, 206, 255))
            else:
                mask_pil_image.putpixel((i, j), (0, 0, 0, 0))

    return mask_pil_image


def reformat_metadata(metadata):
    _ = metadata.pop("_id")  # remove mongoid

    # width = source_size[0]
    # height = source_size[1]

    reformatted_metadata = metadata

    masks_dict = metadata.pop('masks')

    if masks_dict:
        # Reload each individual mask instance
        mask_array = np.array([convert_sparse_to_mask(mask_dict).toarray() for mask_dict in masks_dict])

        # Merge the masks into a single image for overlay
        mask_array = np.sum(mask_array, axis=0)
        # Convert merged mask back to color range
        mask_array[mask_array > 0.5] = 255

        mask_image = Image.fromarray(mask_array).convert("RGBA")       #.resize((width, height))
        mask_image = convert_mask_to_transparent_color(mask_image)
        mask_buffer = pil_to_buffer(mask_image)
        mask_uri = to_data_uri(mask_buffer)

    else:
        mask_array = np.zeros((256,256))
        mask_array[mask_array > 0.5] = 255

        mask_image = Image.fromarray(mask_array).convert("RGBA")
        mask_image = convert_mask_to_transparent_color(mask_image)
        mask_buffer = pil_to_buffer(mask_image)
        mask_uri = to_data_uri(mask_buffer)

    reformatted_metadata['masks'] = mask_uri

    return reformatted_metadata


def get_blob_data(blob, blob_client, cosmos_client):
    stream = io.BytesIO()
    blob_client.download_blob(blob).readinto(stream)
    stream.seek(0)

    # try:
    #     #read binary stream into pillow
    #     image = Image.open(stream).convert('RGB').resize((256, 256))
    #     np_image = np.array(image).astype(np.float32)
    #     np_image = np_image[None, :]
    #     np_image = np.transpose(np_image, (0, 3, 1, 2))
    #
    #     #manipulate image
    #     circle_extractor_model = BASE_DIR.joinpath('SampleAnalysisApp').joinpath('mrcnn_circle.onnx')
    #     ort_session = onnxruntime.InferenceSession(str(circle_extractor_model))
    #
    #     ort_inputs = {ort_session.get_inputs()[0].name: np_image}
    #     ort_outs = ort_session.run(None, ort_inputs)
    #     sample_found = image.copy()
    #     box_found = ort_outs[0][0]
    #     image_cropped = sample_found.crop(box_found)
    #
    #     # Create a BytesIO object to hold the image data
    #     image_bytes = io.BytesIO()
    #
    #     # Save the image to the BytesIO object in a specific format
    #     image_cropped.save(image_bytes, format='png')
    #
    # except Exception:
    #     image_bytes = stream

    source_image = Image.open(stream)
    image_buffer = pil_to_buffer(source_image)
    img = to_data_uri(image_buffer)

    metadata = get_blob_cosmos_metadata(blob.name, cosmos_client)
    if 'masks' in metadata.keys():
        metadata = reformat_metadata(metadata)
    else:
        metadata = metadata

    blob_data = {
        'src': img,
    }

    blob_data = blob_data | metadata
    blob_attributes = list(blob_data.keys())

    return blob_data, blob_attributes


def fetch_profile_from_azure(username):
    blob_service_client = get_blob_connection()
    container_name = str(username).replace(".", "").lower()
    print(container_name)

    cosmos_db = get_cosmos_connection()
    cosmos_collection = cosmos_db[container_name]

    existing_containers = blob_service_client.list_containers()
    container_names = [c.name for c in existing_containers]

    if container_name in container_names:
        print("Found container: ", container_name)
        blob_container_client = blob_service_client.get_container_client(container_name)
    else:
        print("No container found, attempting to create for: ", container_names)
        blob_container_client = blob_service_client.create_container(container_name)

    print('loading all blobs')
    all_blobs = blob_container_client.list_blobs(marker=None)

    profile_objects = []
    for blob in all_blobs:
        blob_dict, blob_keys = get_blob_data(blob, blob_container_client, cosmos_collection)

        blob_object = SimpleNamespace(**blob_dict)
        profile_objects.append(blob_object)

    return profile_objects


# def get_mongo_connection():
#     mongodb_atlas_client = MongoClient(
#         os.getenv("MONGO_IMAGE_URI"),
#         tls=True,
#         tlsCertificateKeyFile=str(BASE_DIR.joinpath("X509-cert-8736505645657480051.pem"))
#     )
#
#     mongodb_atlas_db = mongodb_atlas_client['ImageUploads']
#     return mongodb_atlas_db, mongodb_atlas_client


# def push_to_mongo(img_buf, ort_outs, model, filename, username, metrics):
#     try:
#         mongo_db, mongo_client = get_mongo_connection()
#         mongo_repo = GridFS(mongo_db, username)
#
#         image_contents = img_buf.getvalue()
#
#         image_meta = {
#             'model': model,
#             'bbox': np.array_str(ort_outs[0]),
#             'labels': np.array_str(ort_outs[1]),
#             'scores': np.array_str(ort_outs[2]),
#             'masks': np.array_str(ort_outs[3]),
#             'metrics': json.dumps(metrics)
#         }
#
#         id = mongo_repo.put(image_contents, metadata=image_meta, filename=filename)
#         mongo_client.close()
#         img_buf.close()
#         return id
#     except Exception as err:
#         print(f"Not able to push to Mongo due to: {err}")
#         return ''

# Create your views here.
@gzip.gzip_page
def signup(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('index')
    else:
        form = RegistrationForm()
    return render(request, 'registration/signup.html', {'form': form})


@csrf_exempt
@gzip.gzip_page
def entry(request):
    return render(request, 'SampleAnalysisApp/index.html')


@csrf_exempt
@gzip.gzip_page
def index(request):
    return render(request, 'SampleAnalysisApp/index.html')


@csrf_exempt
@gzip.gzip_page
def analyze_test(request):
    return render(request, 'SampleAnalysisApp/analyze_test.html')


@csrf_exempt
@gzip.gzip_page
@login_required(login_url='/accounts/login/')
def analyze_user(request):
    return render(request, 'SampleAnalysisApp/analyze_user.html')


@csrf_exempt
@gzip.gzip_page
def coming_soon(request):
    return render(request, 'SampleAnalysisApp/coming_soon.html')


@csrf_exempt
@gzip.gzip_page
def gallery(request):
    username = str(request.user)
    try:
        user_blobs = fetch_profile_from_azure(username)
        user_blobs = sorted(user_blobs, key=lambda x: x.BatchID, reverse=True)

        unique_batch = []
        for item in user_blobs:
            if item.BatchID not in unique_batch:
                unique_batch.append(item.BatchID)
        unique_formulation = []
        for item in user_blobs:
            if item.Formulation not in unique_formulation:
                unique_formulation.append(item.Formulation)

    except Exception as err:
        print(f"No Images available due to: {err}")
        user_blobs = []
        unique_batch = []
        unique_formulation = []


    return render(request, 'SampleAnalysisApp/gallery.html', context={'user_blobs': user_blobs, 'unique_batch': unique_batch, 'unique_formulation': unique_formulation})
# @gzip.gzip_page
# @login_required(login_url='/accounts/login/')
# def profile(request):
#     username = str(request.user)
#     try:
#         mongo_db, mongo_client = get_mongo_connection()
#         user_collection = GridFS(mongo_db, username)
#
#         user_files = user_collection.find().sort("uploadDate", -1)
#         user_images = [user_collection.get(ObjectId(doc._id)).read().decode('utf-8') for doc in user_files]
#         # tmp_mongo_id = '63e2c99862abe918602e9156'
#         # debug_data = user_collection.get(ObjectId(tmp_mongo_id))
#         # user_images = [debug_data]
#         # user_images = ''
#     except Exception as err:
#         print(f"No Images available due to: {err}")
#         mongo_client = None
#         user_images = None
#
#     try:
#         return render(request, 'profile.html', context={'images': user_images})
#     finally:
#         if mongo_client is not None:
#             mongo_client.close()


@csrf_exempt
@gzip.gzip_page
# @login_required(login_url='/accounts/login/')
def base_template(request):
    return render(request, 'SampleAnalysisApp/base_template.html')


@gzip.gzip_page
# @login_required(login_url='/accounts/login/')
def chip_corrosion_detection(request):
    if request.method == 'POST' and request.FILES['myfile']:
        # FETCH USERS FILE AND OPEN IMAGE FOR PROCESSING
        myfile = request.FILES['myfile']
        batch_id = "fileupload_single_batch%s".format(time.strftime("_%Y%m%d_%H%M%S"))

        filename = myfile.name

        username = str(request.user)

        img = Image.open(myfile).convert('RGB')

        # FETCH BUFFER FROM PREDICTION WORKFLOW
        obb_img, obb_percent, obb_model, obb_detections, obb_metrics, save_image = get_chip_prediction(img, filename)
        img.close()

        # CONVERT IMAGE USED BY MODEL FOR DETECTION TO BUFFER
        img_buf = pil_to_buffer(save_image)

        # CAST DETECTED IMAGE TO BUFFER FOR EASY LOAD TO WEBPAGE AND CLOSE IMAGE
        obb_buf = pil_to_buffer(obb_img)

        # CAST BYTES BUFFER INTO IMAGE URI FOR HTML LOAD AND CLOSE BUFFERS
        obb_uri = to_data_uri(obb_buf)
        obb_buf.close()

        print("GENERATED ALL IMAGE DATA")
        print("Pushing to Database")
        obb_metrics["Percentage"] = obb_percent
        obb_metrics["Formulation"] = None  # formulation
        obb_metrics["Concentration"] = None #concentration
        obb_metrics["UserRating"] = None  # rating
        obb_metrics["BatchID"] = batch_id

        push_to_azure(img_buf, obb_detections, obb_model, filename, username, obb_metrics)

        img_buf.close()

        obb_details = f"{obb_percent}% based on {obb_model} model"
        # RETURN HTML WITH HTML DATA URI IMAGES
        return render(request, 'SampleAnalysisApp/chip_corrosion_detection.html', context={
            'obb_img': obb_uri, 'obb_details': obb_details, 'obb_metrics': obb_metrics})

    return render(request, 'SampleAnalysisApp/chip_corrosion_detection.html')


@gzip.gzip_page
# @login_required(login_url='/accounts/login/')
def bearing_corrosion_detection(request):
    if request.method == 'POST' and request.FILES['myfile']:
        # FETCH USERS FILE AND OPEN IMAGE FOR PROCESSING
        myfile = request.FILES['myfile']
        batch_id = "fileupload_single_batch%s".format(time.strftime("_%Y%m%d_%H%M%S"))

        filename = myfile.name

        username = str(request.user)

        img = Image.open(myfile).convert('RGB')

        # FETCH BUFFER FROM PREDICTION WORKFLOW
        obb_img, obb_percent, obb_model, obb_detections, obb_metrics, save_image = get_bearing_prediction(img, filename)
        img.close()

        # CONVERT IMAGE USED BY MODEL FOR DETECTION TO BUFFER
        img_buf = pil_to_buffer(save_image)

        # CAST DETECTED IMAGE TO BUFFER FOR EASY LOAD TO WEBPAGE AND CLOSE IMAGE
        obb_buf = pil_to_buffer(obb_img)

        # CAST BYTES BUFFER INTO IMAGE URI FOR HTML LOAD AND CLOSE BUFFERS
        obb_uri = to_data_uri(obb_buf)
        obb_buf.close()

        print("GENERATED ALL IMAGE DATA")
        print("Pushing to Database")
        obb_metrics["Percentage"] = obb_percent
        obb_metrics["Formulation"] = None  # formulation
        obb_metrics["Concentration"] = None #concentration
        obb_metrics["UserRating"] = None  # rating
        obb_metrics["BatchID"] = batch_id

        push_to_azure(img_buf, obb_detections, obb_model, filename, username, obb_metrics)

        img_buf.close()

        obb_details = f"{obb_percent}% based on {obb_model} model"
        # RETURN HTML WITH HTML DATA URI IMAGES
        return render(request, 'SampleAnalysisApp/bearing_corrosion_detection.html', context={
            'obb_img': obb_uri, 'obb_details': obb_details, 'obb_metrics': obb_metrics})

    return render(request, 'SampleAnalysisApp/bearing_corrosion_detection.html')


@gzip.gzip_page
# @login_required(login_url='/accounts/login/')
def livestream(request):
    if request.method == 'POST':
        username = str(request.user).lower()
        filename = username + '_framecapture.png'
        batch_id = "livestream_single_batch%s".format(time.strftime("_%Y%m%d_%H%M%S"))

        img_string = request.POST["frame"]

        img = to_pil_img(img_string)

        # FETCH BUFFER FROM PREDICTION WORKFLOW
        obb_img, obb_percent, obb_model, obb_detections, obb_metrics, save_image = get_prediction(img, filename)
        img.close()

        # CONVERT IMAGE USED BY MODEL FOR DETECTION TO BUFFER
        img_buf = pil_to_buffer(save_image)

        # CAST DETECTED IMAGE TO BUFFER FOR EASY LOAD TO WEBPAGE AND CLOSE IMAGE
        obb_buf = pil_to_buffer(obb_img)

        # CAST BYTES BUFFER INTO IMAGE URI FOR HTML LOAD AND CLOSE BUFFERS
        obb_uri = to_data_uri(obb_buf)
        obb_buf.close()

        print("GENERATED ALL IMAGE DATA")
        print("Pushing to Database")
        obb_metrics["Percentage"] = obb_percent
        obb_metrics["Formulation"] = None  # formulation
        obb_metrics["Concentration"] = None # concentration
        obb_metrics["UserRating"] = None  # rating
        obb_metrics["BatchID"] = batch_id

        push_to_azure(img_buf, obb_detections, obb_model, filename, username, obb_metrics)

        img_buf.close()

        obb_details = f"{obb_percent}%"  # based on {obb_model} model

        # RENDER HTML WITH IMAGE DISPLAY
        # return HttpResponse(obb_uri, content_type='application/json')
        return render(request, 'SampleAnalysisApp/livestream.html', context={
            'obb_img': obb_uri, 'obb_details': obb_details, 'obb_metrics': obb_metrics})

        # return redirect(reverse('livestream_detection'))
        # kwargs={'orig_img': img_uri, 'obb_img': obb_uri}))

    else:
        return render(request, 'SampleAnalysisApp/livestream.html')


# @gzip.gzip_page
# # #@login_required(login_url='/accounts/login/')
# def livestream_ar(request):
#     if request.method == 'POST':
#         username = str(request.user)
#         filename = username + '.FrameCapture.png'
#
#         img_string = request.POST["frame"]
#
#         img = to_pil_img(img_string)

#         # FETCH BUFFER FROM PREDICTION WORKFLOW
#         obb_img, obb_percent, obb_model, obb_detections, obb_metrics, save_image = get_prediction(img, filename)
#         img.close()
#
#         # CONVERT IMAGE USED BY MODEL FOR DETECTION TO BUFFER
#         img_buf = pil_to_buffer(save_image)
#
#         # CAST DETECTED IMAGE TO BUFFER FOR EASY LOAD TO WEBPAGE AND CLOSE IMAGE
#         obb_buf = pil_to_buffer(obb_img)
#
#         # CAST BYTES BUFFER INTO IMAGE URI FOR HTML LOAD AND CLOSE BUFFERS
#         obb_uri = to_data_uri(obb_buf)
#         obb_buf.close()
#
#         print("GENERATED ALL IMAGE DATA")
#         print("Pushing to Database")
#         obb_metrics["Percentage"] = obb_percent
#         obb_metrics["Formulation"] = None  # formulation
#         obb_metrics["Concentration"] = None  # concentration
#         obb_metrics["UserRating"] = None  # rating
#         obb_metrics["BatchID"] = batch_id
#
#         push_to_azure(img_buf, obb_detections, obb_model, filename, username, obb_metrics)
#
#         img_buf.close()
#
#         obb_details = f"Result: {obb_percent}% based on {obb_model} model"
#
#         # RENDER HTML WITH IMAGE DISPLAY
#         # return HttpResponse(obb_uri, content_type='application/json')
#         return render(request, 'SampleAnalysisApp/livestream_ar.html', context={
#             'obb_img': obb_uri, 'obb_details': obb_details, 'obb_metrics': obb_metrics})
#
#         # return redirect(reverse('livestream_detection'))
#         # kwargs={'orig_img': img_uri, 'obb_img': obb_uri}))
#
#     else:
#         print("return")
#
#         return render(request, 'SampleAnalysisApp/livestream_ar.html')

#
# @gzip.gzip_page
# # @login_required(login_url='/accounts/login/')
# def video(request):
#     return redirect(reverse('index'))


@gzip.gzip_page
@login_required(login_url='/accounts/login/')
def batch(request):
    samples = request.session.get("batch", {})

    return render(request, 'SampleAnalysisApp/batch.html', context={'samples': samples})


@gzip.gzip_page
@login_required(login_url='/accounts/login/')
def batchSubmit(request):
    if request.method == "POST":
        batch_id = f"batch{time.strftime('_%Y%m%d_%H%M%S')}"

        username = str(request.user)
        # progress_key = f'upload_progress_{username}'

        if 'batch' not in request.session:
            request.session['batch'] = samples = {}

        else:
            samples = request.session.get("batch")

        print(samples)

        unique_indices = set()
        for key in list(request.POST.keys()):
            index = int(key.split("[")[1].split("]")[0])
            unique_indices.add(index)

        ### CONVERT THIS CODE TO PERFORM A BATCH PREDICTION ON ALL SAMPLES IN ONE GO THEN A BATCH PUSH TO AZURE
        total_samples = len(unique_indices)

        for i in unique_indices:
            if key.startswith("samples["):
                formulation = request.POST.get(f'samples[{i}][formulation]')
                concentration = request.POST.get(f'samples[{i}][concentration]')
                rating = request.POST.get(f'samples[{i}][rating]')
                image = request.POST.get(f'samples[{i}][img_data]')

                if 'not provided' in str(formulation).lower():
                    formulation = 'na'
                if 'not provided' in str(rating).lower():
                    rating = 'na'

                img = to_pil_img(image)
                filename = formulation + '_' + rating + '_' + f'sample{i}.png'

                # FETCH BUFFER FROM PREDICTION WORKFLOW
                obb_img, obb_percent, obb_model, obb_detections, obb_metrics, save_image = get_prediction(img, filename)
                img.close()

                # CONVERT IMAGE USED BY MODEL FOR DETECTION TO BUFFER
                img_buf = pil_to_buffer(save_image)

                # CAST DETECTED IMAGE TO BUFFER FOR EASY LOAD TO WEBPAGE AND CLOSE IMAGE
                obb_buf = pil_to_buffer(obb_img)

                # CAST BYTES BUFFER INTO IMAGE URI FOR HTML LOAD AND CLOSE BUFFERS
                obb_uri = to_data_uri(obb_buf)
                obb_buf.close()

                print("GENERATED ALL IMAGE DATA")
                print("Pushing to Database")
                obb_metrics["Percentage"] = obb_percent
                obb_metrics["Formulation"] = formulation
                obb_metrics["Concentration"] = concentration
                obb_metrics["UserRating"] = rating
                obb_metrics["BatchID"] = batch_id

                push_to_azure(img_buf, obb_detections, obb_model, filename, username, obb_metrics)

                img_buf.close()

                samples[i] = {'src': obb_uri} | obb_metrics
                request.session['batch'] = samples
                request.session.save()

        return JsonResponse({"status": "success"})
    else:
        print("Not a POST request")


@login_required(login_url='/accounts/login/')
def reset_session_data(request):
    request.session.pop('batch', None)
    return JsonResponse({'status': 'success'})


def perform_secondary_cv_detection(image, output):
    orig_image = image.copy()
    try:
        print('Using Hough Circle Edge Detection')
        open_cv_image = np.array(image)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        (thresh, blackAndWhiteImage) = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        edged = cv2.Canny(blackAndWhiteImage, 200, 650)  # 150, 650

        # define a (3, 3) structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # apply the dilation operation to the edged image
        dilate = cv2.dilate(edged, kernel, iterations=1)
        closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

        circles = cv2.HoughCircles(closing, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=int((256 / 2) - 50), maxRadius=int(256 / 2))

        circles = np.uint8(np.around(circles))
        mask = np.zeros(open_cv_image.shape[:2], dtype='uint8')
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

        masked = cv2.bitwise_and(open_cv_image, open_cv_image, mask=mask)
        detected_circle = Image.fromarray(masked).convert('RGB')

        return detected_circle, True
    except TypeError:
        print("Wasn't Able to Find Any Additional Contours with Current Image Size")
        return orig_image, False


# def export_images_to_csv(request):
#     username = str(request.user)
#     try:
#         images = fetch_profile_from_azure(username)
#
#     except Exception as err:
#         print(f"No Images available due to: {err}")
#         images = []
#
#     # Create a response object with appropriate headers for a CSV file
#     response = HttpResponse(content_type='text/csv')
#     response['Content-Disposition'] = 'attachment; filename="gallery_data.csv"'
#
#     # Create a CSV writer and write the header row
#     writer = csv.writer(response)
#     writer.writerow(['Image Name', 'User Rating', 'Model Rating', 'Defect Percentage', 'Number of Small Defects', 'Number of Medium Defects', 'Number of Large Defects'])  # Modify as needed
#
#     # Write the data from images to the CSV file
#     for image in images:
#         # Check for None values before accessing properties
#         blob = image.blob if image.blob is not None else ""
#         user_rating = image.UserRating if image.UserRating is not None else ""
#         rating = image.Rating if image.Rating is not None else ""
#         percentage = image.Percentage if image.Percentage is not None else ""
#         small = image.Small if image.Small is not None else ""
#         medium = image.Medium if image.Medium is not None else ""
#         large = image.Large if image.Large is not None else ""
#
#         # Write the data to the CSV file
#         writer.writerow([blob, user_rating, rating, percentage, small, medium, large])
#
#     return response

def export_to_csv(request):
    if request.method == 'POST':
        # Get the JSON data from the request and convert it to a list of dictionaries
        data = json.loads(request.body)

        # Set up the HTTP response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="export.csv"'

        # Create the CSV writer
        writer = csv.writer(response)

        # Write the header row
        writer.writerow(['Name', 'User Rating', 'Percent Defect', 'Model Rating'])

        # Write the data rows
        for item in data:
            writer.writerow([item['name'], item['user_rating'], item['percent'], item['model_rating']])

        return response


def create_report(request):
    username = str(request.user).replace('.', ' ').title()
    if request.method == 'POST':
        # Get the JSON data from the request and convert it to a list of dictionaries
        data = json.loads(request.body)

        template_path = str(BASE_DIR.joinpath("templates/chip_report_template.docx"))
        print(template_path)
        # images_path = str(BASE_DIR.joinpath("static").joinpath('images').joinpath('report_images'))
        doc = DocxTemplate(template_path)

        # set jinja template variables
        content = {
            'data': data['data'],
            'customerName': data['customerName'],
            'comments': data['comments'],
            'date': date.today().strftime("%B %d, %Y"),
            'operator': username,
        }

        # Render the data in the template
        doc.render(content)

        # Write the generated .docx to a BytesIO buffer
        docx_buf = io.BytesIO()
        doc.save(docx_buf)
        docx_buf.seek(0)

        filename = 'chip_corrosion_report.docx'

        # Return the .docx file as a response
        return FileResponse(docx_buf, as_attachment=True, filename=filename,
                            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')


# def create_report(request):
#     username = str(request.user).replace('.', ' ').title()
#     if request.method == 'POST':
#         # Get the JSON data from the request and convert it to a list of dictionaries
#         data = json.loads(request.body)
#
#         # # Convert base64 strings of the images and masks to PIL.Image objects
#         # for item in data:
#         #     image_data = base64.b64decode(item['image'].split(',')[1])
#         #     mask_data = base64.b64decode(item['mask'].split(',')[1])
#         #
#         #     item['image'] = Image.open(io.BytesIO(image_data))
#         #     item['mask'] = Image.open(io.BytesIO(mask_data))
#
#         template_path = str(BASE_DIR.joinpath("templates"))
#         images_path = str(BASE_DIR.joinpath("static").joinpath('images').joinpath('report_images'))
#
#         environment = Environment(loader=FileSystemLoader(template_path))
#         report_template = environment.get_template("report.html")
#
#         # write the header image to the buffer
#         header_buf = io.BytesIO()
#         header = Image.open(images_path + '/report_header.png')
#         header.save(header_buf, format='png')
#         header_uri = to_data_uri(header_buf)
#         header.close()
#         header_buf.close()
#
#         # set jinja template variables for pt_report.html
#         content = report_template.render(
#             rigname='Chip Corrosion',
#             data=data['data'],
#             customerName=data['customerName'],
#             comments=data['comments'],
#             date=date.today().strftime("%B %d, %Y"),
#             operator=username,
#             header=header_uri
#         )
#
#         # write the generated html file to the buffer
#         html_buf = io.StringIO()
#         html_buf.write(content)
#         html_buf.seek(0)
#
#         # write the generated pdf to the buffer
#         pdf_buf = io.BytesIO()
#         HTML(string=html_buf.read()).write_pdf(pdf_buf)
#         pdf_buf.seek(0)
#
#         # response = FileResponse(pdf_buf)
#         response = HttpResponse(pdf_buf, content_type='text/pdf')
#         response['Content-Type'] = 'application/pdf'
#         response['Content-Disposition'] = 'attachment; filename="report.pdf"'
#
#         return response
