##### ABOUT THE FILE ######
'''
This python file is a part of the proposed tentative solution to the problem statement 'Face Recognition - Age, Ethnicity and Emotion classification'.
Note: The model used for Face Recognition will be improved in the the comming version.
'''

import numpy as np
import cv2

# Loading output_mapper
# This dictionary maps the predicted output with the text
import pickle
output_mapper = pickle.load( open( "output_mapper.p", "rb" ) ) 

# Loading model
# A CNN model trained on the dataset provided (Face_Recognition.json)
from keras.models import load_model
model = load_model('production_model.h5')

IMAGE_WIDTH = IMAGE_HEIGHT = 100 # Depends on model the model used
IMAGE_CHANNELS=1 # Depends on model the model used

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Haar Cascade for face detection
########################################################################################################

# SET VALUES AS PER REQUIREMENT
LOCATION_IMAGE = '' # For non real time prediction (i.e., not from live stream or video )
LOCATION_VIDEO = '/absolute/path/to/the/testvideo.mp4' # For real time prediction. Set to 0 (zero) if using webcamp or else the string format absolute path to the video file

# Comment this code block of code and uncomment the next block for Realtime (live) prediction from webcam
###############################################################################
# img = cv2.imread(LOCATION_IMAGE)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 1)
# if len(faces)==0:
#     print('No face detected')
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     roi_gray = cv2.resize(roi_gray,(IMAGE_HEIGHT, IMAGE_WIDTH))
#     roi_gray = roi_gray / 255.0
#     pred = model.predict(np.array(roi_gray).reshape(1,IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
#     pred_dict = dict(zip(['age', 'race', 'emotion', 'gender'], pred ))
#     print('-------------Another Face-----------') # Showing output in Terminal
#     text = ''
#     for key, value in pred_dict.items():
#         text+= str(output_mapper[key][np.argmax(value[0])]) + ''
#         print(output_mapper[key][np.argmax(value[0])]) # Showing output in Terminal
#     cv2.putText(img,text, (x,y),cv2.FONT_HERSHEY_DUPLEX,0.5,(200,0,0),1)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
###############################################################################



# Uncomment the following code for RealTime (live) prediction from the Webcam
###############################################################################
cap=cv2.VideoCapture()
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 1)
    if len(faces)==0:
        print('No face detected')
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray,(IMAGE_HEIGHT, IMAGE_WIDTH))
        roi_gray = roi_gray / 255.0
        pred = model.predict(np.array(roi_gray).reshape(1,IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        pred_dict = dict(zip(['age', 'race', 'emotion', 'gender'], pred ))
        print('-------------Another Face-----------') # Showing output in Terminal
        text = ''
        for key, value in pred_dict.items():
            text+= str(output_mapper[key][np.argmax(value[0])]) + ''
            print(output_mapper[key][np.argmax(value[0])]) # Showing output in Terminal
        cv2.putText(img,text, (x,y),cv2.FONT_HERSHEY_DUPLEX,0.5,(200,0,0),1)
    cv2.imshow('Face Recognition',img)
    if cv2.waitKey(10) == ord('q'): # wait until 'q' key is pressed
        break
cap.release()
cv2.destroyAllWindows
###############################################################################