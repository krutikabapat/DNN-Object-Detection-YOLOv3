import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time


c_threshold = 0.5
nms = 0.4
width = 416
height = 416

parser = argparse.ArgumentParser(description = 'YOLO')
parser.add_argument('image', help = 'Path to Image File')
args = vars(parser.parse_args())

classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

COLORS = np.random.randint(0, 255, size=(len(classes), 3),
	dtype="uint8")
 
# initialize a list of colors to represent each possible class label

config = './yolov3.cfg'
weight = './yolov3.weights'


net = cv.dnn.readNetFromDarknet(config, weight)
image = cv.imread(args['image'])

(H,W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv.dnn.blobFromImage(image,1/255,(416,416),swapRB=True, crop=False)
net.setInput(blob)


start = time.time()

layersOutputs = net.forward(ln)
print(layersOutputs)

end = time.time()

boxes = []
confidences = []
classIDs = []

for output in layersOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
 
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > c_threshold:
			
			box = detection[0:4]* np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
 
			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
 
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)
			

idxs = cv.dnn.NMSBoxes(boxes, confidences, c_threshold, nms)

if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
 
		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
		cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
 
# show the output image
cv.imshow("Image", image)
cv.waitKey(0)


