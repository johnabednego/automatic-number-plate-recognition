import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import util


# define constants

model_cfg_path = "C:/Users/jilim/Desktop/automatic-number-plate-recognition-python-master/yolov3-from-opencv-object-detection/model/cfg/darknet-yolov3.cfg"
model_weights_path="C:/Users/jilim/Desktop/automatic-number-plate-recognition-python-master/yolov3-from-opencv-object-detection/model/model.weights"
class_names_path="C:/Users/jilim/Desktop/automatic-number-plate-recognition-python-master/yolov3-from-opencv-object-detection/model/class.names"

input_dir = 'C:/Users/jilim/Desktop/automatic-number-plate-recognition-python-master/data'

for img_name in os.listdir(input_dir):

    img_path = os.path.join(input_dir, img_name)

    # load class names
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()

    # load model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # load image

    img = cv2.imread(img_path)

    H, W, _ = img.shape

    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # apply nms
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    # plot
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            15)

        # Draw a green rectangle arounded the detected number plate
        final_license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Pass the image through pytesseract
    text = pytesseract.image_to_string(final_license_plate)

    # Print the extracted text
    print(text[:-1])
    plt.figure()
    plt.imshow(cv2.cvtColor(final_license_plate, cv2.COLOR_BGR2RGB))

    plt.show()
