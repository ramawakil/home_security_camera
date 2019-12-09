import cv2
import numpy as np
from PIL import Image
import os


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


# Path for face image database
path = 'dataset'
# function to get the images and label data


def image_labels(path):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for imagePath in image_paths:
        pil_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(pil_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return face_samples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = image_labels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
# Print the number of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))