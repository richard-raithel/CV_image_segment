import base64
import json
import io
import os

import torch
from PIL import Image, ImageFile, ImageDraw
from torchvision import transforms
import onnxruntime
from pathlib import Path
import numpy as np
import cv2
from shapely.geometry import Polygon
from dotenv import load_dotenv
import albumentations as A
from albumentations.pytorch import ToTensorV2


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SIZE = 256


def model_version_lookup(model_name):
    models = {
        'chip': 'mrcnn_chip_gan.onnx',
        'bearing': 'mrcnn_bearing.onnx'
    }

    model_used = models.get(model_name)
    print(f"Using model: {model_used}")
    model_path = BASE_DIR.joinpath('Utility').joinpath(model_used)
    return model_path


def transform_image(image, IMAGE_SIZE):
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ]
    )
    return transform_pipeline(image).unsqueeze(0)


def prepare_bearing_image(image, IMAGE_SIZE):
    transforms = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            ToTensorV2()
        ]
    )

    transformed = transforms(image=np.array(image))

    img = transformed["image"].to(torch.float32) / 255.0

    # Convert the tensor to numpy array
    np_img = img.cpu().numpy()

    # Add a dimension to simulate a batch. The model expects a shape of [N, C, H, W] (batch size, channels, height, width)
    np_img = np.expand_dims(np_img, axis=0)

    return np_img


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def rating_calc(percent_defect, num_masks, pixel_dim):
    if 0 < num_masks <= 3:
        if pixel_dim < 1:
            rating = 1
        else:
            rating = 2
    elif num_masks > 3 and percent_defect < 1:
        rating = 2
    elif 1 < percent_defect < 5:
        rating = 3
    elif percent_defect > 5:
        rating = 4
    else:
        rating = 'Invalid'
    return rating


def chip_astm_mapping(defects, pixel_metric, percentage):

    defect_checks = {
        "Small": 0,
        "Medium": 0,
        "Large": 0
    }
    if defects:
        try:
            boxes = defects[0]
            mask = np.array(defects[-1][0, 0, :, :])
            mask_im = Image.fromarray(mask)
        except IndexError:
            boxes = []
            mask = []
            mask_im = []
        defects_mm = []
        for box in boxes:
            x = box[0]
            y = box[1]

            if box[2] > box[0]:
                w = box[2]
            else:
                w = box[0] + box[2]

            if box[3] > box[1]:
                h = box[3]
            else:
                h = box[1] + box[3]

            defect = mask_im.copy().crop((x, y, w, h))
            defects_mm.append([dim * pixel_metric for dim in defect.size])

        for defect in defects_mm:
            if defect[0] > 5 or defect[1] > 5:
                defect_checks["Large"] += 1
            elif defect[0] > 1 or defect[1] > 1:
                defect_checks["Medium"] += 1
            else:
                defect_checks["Small"] += 1

        if defects_mm:
            max_defect_size = np.array(defects_mm).max()
            lab_rating = rating_calc(percentage, len(defects_mm), max_defect_size)
        else:
            lab_rating = 0
    else:
        lab_rating: 0

    defect_checks['Rating'] = lab_rating
    return defect_checks


def get_defect_percentage(total_area, results):

    masks = results[-1]
    # Adjust this calculation to only count the non-transparent pixels for total area
    masks[masks > 0.5] = 1
    masks[masks < 0.5] = 0
    mask_area = []
    for k in range(len(masks)):
        temp_mask = masks[k][0, :, :]
        mask_area.append(sum(sum(temp_mask)))
    defect_area = sum(mask_area)
    defect_percentage = round((defect_area/total_area)*100, 2)

    return defect_percentage, defect_area


def view(image, output, percentage, name, n=1, std=1, mean=0, cmap='Greys'):

    if len(output[0]) > 0:
        # RESIZE SOURCE IMAGE TO MASK SIZE
        mask_image = image.resize((256,256)).convert("RGBA")

        # Plot output (predicted) masks
        output_im = output[-1][0][0, :, :]
        for k in range(len(output[-1])):
            output_im2 = output[-1][k][0, :, :]
            output_im2[output_im2 > 0.5] = 1
            output_im2[output_im2 < 0.5] = 0
            output_im = output_im + output_im2

        output_im[output_im > 0.5] = 255
        output_im[output_im < 0.5] = 0
        masks = Image.fromarray(output_im).convert("RGBA")

        ### FEED THIS CONSOLIDATED MASK INTO A CALCULATION OF AREA

        width = masks.size[0]
        height = masks.size[1]
        for i in range(0, width):  # process all pixels
            for j in range(0, height):
                data = masks.getpixel((i, j))
                if data[:3] == (255,255,255):
                    masks.putpixel((i, j), (0, 191, 255, 125))
                else:
                    masks.putpixel((i, j), (0, 0, 0, 0))

        # overlay = Image.blend(mask_image, masks, alpha=0.2)
        overlay = Image.alpha_composite(mask_image, masks)

        # # Plot output (predicted) bounding boxes
        # l=output[0]
        # scores = output[2]
        # l[:,2]=l[:,2]-l[:,0]
        # l[:,3]=l[:,3]-l[:,1]
        # draw = ImageDraw.Draw(overlay)
        # for j in range(len(l)):
        #     x = l[j][0]
        #     y = l[j][1]
        #     w = l[j][0]+l[j][2]
        #     h = l[j][1]+l[j][3]
        #     draw.rectangle((x, y, w, h), fill=None, outline="red")
        #     # plt.text(l[j][0], l[j][1], str(round(scores[j]*100, 2)), fontdict=font)

        masks.close()
        mask_image.close()

        # percent_overlay = f"Percentage Defect: {str(round(percentage,2))}"
        # # SCALE TEXT TO IMAGE DIMENSIONS
        # textsize = int(width*0.05)
        # h_center = int(height/2)
        # w_center = int((width/2)-(width*.25))
        # draw = ImageDraw.Draw(overlay)
        # text_path = BASE_DIR.joinpath('Corrosion').joinpath('LiberationSerif-Regular.ttf')
        # font = ImageFont.truetype(str(text_path), textsize)
        # percent_overlay = f"Defect %: {str(round(percentage,2))}"
        # draw.text((h_center, w_center), percent_overlay, (0,0,0), font)

        return overlay
    else:
        return image


def to_data_uri(img_buf):
    img_64 = base64.b64encode(img_buf.getvalue())
    uri = img_64.decode('utf-8')
    return f'data:image/png;base64,{uri}'


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


def get_image_class(image):
    transformclassifier = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_to_classify = transformclassifier(image)
    img_to_classify = img_to_classify[None, :]
    classifier_path = BASE_DIR.joinpath('Utility').joinpath('classifier.onnx')
    ort_sess = onnxruntime.InferenceSession(str(classifier_path))

    ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_to_classify)}
    ort_outs = ort_sess.run(None, ort_inputs)

    if ort_outs[0][0][0] > ort_outs[0][0][1]:
        model = 'bearing'
    else:
        model = 'chip'

    return model


def get_chip_prediction(image, filename):
    #Classifier for model selection
    width, height = image.size

    # model_name = get_image_class(image)

    model_name = 'chip'
    model_path = model_version_lookup(model_name)

    img_tensor = transform_image(image, IMAGE_SIZE)

    circle_extractor_model = BASE_DIR.joinpath('Utility').joinpath('mrcnn_circle.onnx')
    ort_session = onnxruntime.InferenceSession(str(circle_extractor_model))

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_tensor)}

    ort_outs = ort_session.run(None, ort_inputs)

    sample_found = image.resize((IMAGE_SIZE, IMAGE_SIZE)).copy()

    box_found = ort_outs[0][0]
    sample_found = sample_found.crop(box_found)
    resize_sample = sample_found.resize((IMAGE_SIZE, IMAGE_SIZE))

    # CALCULATE TOTAL CIRCLE AREA
    detected_circle_mask = ort_outs[-1]
    detected_circle_mask[detected_circle_mask > 0] = 1
    detected_circle_mask[detected_circle_mask < 0.5] = 0
    mask_area = []
    for k in range(len(detected_circle_mask)):
        temp_mask = detected_circle_mask[k][0, :, :]
        mask_area.append(sum(sum(temp_mask)))
    object_area_found = sum(mask_area)

    isolated_specimen, found = perform_secondary_cv_detection(resize_sample, ort_outs)

    transform_circle = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform_circle(isolated_specimen).unsqueeze(0)
    show_image = isolated_specimen.copy()

    paper_diameter = 90                                                   # mm
    circle_pixels_in_diameter = 256                                       # based on fixed Image input size
    pixel_per_mm = paper_diameter / circle_pixels_in_diameter             # mm/pixel

    save_image = show_image.copy()

    ort_session = onnxruntime.InferenceSession(str(model_path))

    input_image = to_numpy(img_tensor)
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}

    ort_outs = ort_session.run(None, ort_inputs)

    # ##    Filter by confidence
    # boxes = ort_outs[0]
    # labels = ort_outs[1]
    # scores = ort_outs[2]
    # masks = ort_outs[3]

    defect_percentage, defect_area = get_defect_percentage(object_area_found, ort_outs)

    pil_image = view(show_image, ort_outs, defect_percentage, 'Predicted')
    pil_image = pil_image.resize((512, 512)).convert("RGB")

    metrics = chip_astm_mapping(ort_outs, pixel_per_mm, defect_percentage)

    results = {
        'boxes': ort_outs[0].tolist(),
        'labels': ort_outs[1].tolist(),
        'scores': ort_outs[2].tolist(),
        'masks': ort_outs[3],
    }

    return pil_image, defect_percentage, model_name, results, metrics, save_image

    # finally:
    #     mongo_connection = setup_mongodb()
    #     mongo_db = mongo_connection['computer_vision_results']
    #
    #     img_buf = io.BytesIO()
    #     image.save(img_buf, format='jpeg')
    #     image.close()
    #
    #     img_uri = to_data_uri(img_buf)
    #     img_buf.close()
    #
    #     boxes = ort_outs[0].tolist()
    #     classes = ort_outs[1].tolist()
    #     scores = ort_outs[2].tolist()
    #     masks = ort_outs[-1].tolist()
    #
    #     result_data = {
    #         'image': img_uri,
    #         'mask': json.dumps(masks),
    #         'bboxes': boxes,
    #         'classes': classes,
    #         'scores': scores
    #     }
    #
    #     if 'chip' in model_used.lower():
    #         chip_collection = mongo_db['Chip']
    #         chip_collection.insert_one(result_data)
    #
    #     elif 'bearing' in model_used.lower():
    #         bearing_collection = mongo_db['Bearing']
    #         bearing_collection.insert_one(result_data)


def get_bearing_prediction(image, filename):
    #Classifier for model selection
    width, height = image.size

    # model_name = get_image_class(image)

    model_name = 'bearing'
    model_path = model_version_lookup(model_name)

    # input_image = prepare_bearing_image(image, IMAGE_SIZE)

    img_tensor = transform_image(image, IMAGE_SIZE)
    input_image = to_numpy(img_tensor)

    show_image = image.copy().resize((IMAGE_SIZE, IMAGE_SIZE))

    ort_session = onnxruntime.InferenceSession(str(model_path))

    ort_inputs = {ort_session.get_inputs()[0].name: input_image}

    ort_outs = ort_session.run(None, ort_inputs)

    # ##    Filter by confidence
    # boxes = ort_outs[0]
    # labels = ort_outs[1]
    # scores = ort_outs[2]
    # masks = ort_outs[3]

    # CALCULATE TOTAL IMAGE AREA (IMAGE DIMENSIONS)
    object_area_found = show_image.size[0] * show_image.size[1]

    place_holder_dim = 100  # mm
    image_dim = 256  # based on fixed Image input size
    pixel_per_mm = place_holder_dim / image_dim  # mm/pixel

    save_image = show_image.copy()

    defect_percentage, defect_area = get_defect_percentage(object_area_found, ort_outs)

    pil_image = view(show_image, ort_outs, defect_percentage, 'Predicted')
    pil_image = pil_image.resize((512, 512)).convert("RGB")

    metrics = {"Small": 0, "Medium": 0, "Large": 0, "Rating": 0}

    results = {
        'boxes': ort_outs[0].tolist(),
        'labels': ort_outs[1].tolist(),
        'scores': ort_outs[2].tolist(),
        'masks': ort_outs[3],
    }

    return pil_image, defect_percentage, model_name, results, metrics, save_image



# image = 'test_2.jpg'
# #
# image = '../static/images/img141_7.jpg'
# pil_image = Image.open(image)
# #
# prediction, percent, model, outputs, metrics, save = get_prediction(pil_image, 'sample')
# print(percent)
# prediction.save(f"bearing_sample_metrics_{percent}.png")
#
# width = 512*2
# height = 512
#
# combined = Image.new("RGB", (width, height))
#
# combined.paste(pil_image.resize((512,512)))
# combined.paste(prediction, (512, 0))
#
# combined.save(f"{image.split('/')[-1]}_bearing_sample_metrics_{percent}.png")