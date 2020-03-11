# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


r_eye = slice(36, 42)
l_eye = slice(42, 48)

jaw = slice(0, 17)
v_nose = slice(27, 31)
h_nose = slice(31, 36)

key_points = [0, 8, 16, 36, 45, 48, 54]
center = 30

def get_bounds(points):
	min_x = max_x = points[0][0]
	min_y = max_y = points[0][1]
	for (x,y) in points:
		max_x = x if x > max_x else max_x
		min_x = x if x < min_x else min_x
		max_y = y if y > max_y else max_y
		min_y = y if y < min_y else min_y
	return (min_x, min_y, max_x - min_x, max_y - min_y)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def get_key_points(face_points):
	points = []
	for idx in key_points:
		points.append(face_points[idx])
	center_point = face_points[center]

	(min_x, min_y, w, h) = get_bounds(points + [center_point])
	max_x = min_x + w
	max_y = min_y + h

	centered_points = [[point[0] - center_point[0], point[1] - center_point[1]] for point in points]

	room_left = center_point[0] - min_x
	room_right = max_x - center_point[0]
	room_top = center_point[1] - min_y
	room_down = max_y = center_point[1]

	factor = max([room_left, room_right, room_top, room_down])

	final_points = [[point[0]/factor, point[1]/factor] for point in centered_points]
	final_points.append([0.0, 0.0])

	return final_points
	

def blob_process(img, threshold):
	detector_params = cv2.SimpleBlobDetector_Params()
	detector_params.filterByArea = True
	detector_params.maxArea = 1500
	detector = cv2.SimpleBlobDetector_create(detector_params)
    # print(threshold)

	# knock out red channel
	img[:, :, 2] = 0

	# for bright in range(15, 256, 12):
	# 	print(bright)
	# 	print("Before brighten")
	# 	cv2.imshow('before',img)
	# 	# cv2.waitKey(0)
	# 	img2 = increase_brightness(img, bright)
	# 	print("After brighten")
	# 	cv2.imshow('after',img2)
	# 	cv2.waitKey(0)	
	
	gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print("Gray")
	cv2.imshow('image',gray_frame)
	cv2.waitKey(0)
	_, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
	print("Before processing")
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	img = cv2.erode(img, None, iterations=1)
	print("After erode")
	cv2.imshow('image',img)
	cv2.waitKey(0)
	# img = cv2.dilate(img, None, iterations=2)
	# print("After dilate")
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	img = cv2.medianBlur(img, 5)
	print("After Blur")
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	keypoints = detector.detect(img)
	return keypoints



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
# image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	assert len(shape) == 68
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image

	# for eye in [r_eye, l_eye]:

	# 	for (x, y) in shape[eye]:
	# 		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	'''
	# pupils:
	(x, y, w, h) = get_bounds(shape[eye])
	# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
	# eye = image[x:x + w, y:y + h]

	eps = 5
	eye = image[y - eps:y + h + eps, x - eps:x + w + eps]

	threshold = 25
	# for threshold in range(20, 31, 5):
	print(threshold)
	keypoint = blob_process(eye, threshold)
	print(keypoint)
	'''

	for idx in key_points + [center]:
		(x, y) = shape[idx]
		cv2.circle(image, (x,y), 1, (0, 255, 0), -1)

	important_points = get_key_points(shape)
	for (x, y) in important_points:
		cv2.circle(image, (int(x*100.0 + 100.0), int(y*100.0 + 100.0)), 1, (0, 255, 0), -1)


	# for (x, y) in shape[l_eye]:
	# 	cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

	# for (x, y) in shape[jaw]:
	# 	cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

	# for (x, y) in shape[v_nose]:
	# 	cv2.circle(image, (x, y), 1, (0, 200, 200), -1)

	# for (x, y) in shape[h_nose]:
	# 	cv2.circle(image, (x, y), 1, (200, 0, 200), -1)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)


