import joblib
import base64
import json
import numpy as np
import cv2
from wavelet import w2d

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
	face_cascade = cv2.CascadeClassifier('/opencv/haarcascades/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('/opencv/haarcascades/haarcascade_eye.xml')

	if image_path:
		img=cv2.imread(image_path)
	else: # if image is not saved on disk
		img = base64.b64decode(image_base64_data)
		img = np.fromstring(img, np.uint8)
		img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	img = cv2.imread(image_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		if len(eyes) >= 2:
			return roi_color
		
def classify_image(image_base64_data, file_path=None):

	imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

	result = []
	for img in imgs:
		scalled_raw_img = cv2.resize(img, (32, 32))
		img_har = w2d(img, 'db1', 5)
		scalled_img_har = cv2.resize(img_har, (32, 32))
		combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

		len_image_array = 32*32*3 + 32*32

		final = combined_img.reshape(1,len_image_array).astype(float)
		result.append({
			'class': class_number_to_name(__model.predict(final)[0]),
			'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
			'class_dictionary': __class_name_to_number
		})

	return result

def classify_image(img_b64,file_path=None):
	# img_b64 = base64.b64encode(img)
	# img_b64 = img_b64.decode('utf-8')
	# print(img_b64)