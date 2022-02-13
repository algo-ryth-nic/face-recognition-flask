import cv2
import time 
import os 
import numpy as np
from PIL import Image

PATH_HAAR_CASCADE = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(PATH_HAAR_CASCADE)


def generate_dataset(id, max_samples=15) -> None:
    """
    Generates images for the given id that will be used later for training
    """
    sample_images = 0
    # intialize the camera
    cap = cv2.VideoCapture(0)
    
    cap.read()
    input('Press Enter to start')
    
    while True:
        # read the image
        ret, img = cap.read()
        # convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect the faces
        faces = detector.detectMultiScale(gray, 1.3, 5)
        # draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # increment the sample image
            sample_images += 1
            print(sample_images)
            
            # save the captured image into the datasets folder
            cv2.imwrite("data/user." + str(id) + '.' + str(sample_images) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('SAMPLE_IMAGE_SAVED', gray[y:y+h,x:x+w])
        
        
        cv2.imshow('Live-video-feed', img)

        if sample_images >= max_samples:
            print('Captured ' + str(sample_images) + ' images')
            break

        if cv2.waitKey(1500) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train() -> None:
    """
    Trains the dataset
    """
    # create the training data folder
    if not os.path.exists('data/'):
        raise Exception('No training data found')

    # get the list of all the available images
    images = os.listdir('data/')
    # create the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # variables
    faces = []
    labels = []
    # loop over the images
    for image in images:
        # get the label of the image
        label = int(image.split('.')[1])
        # get the path of the image
        image_path = os.path.join('data/', image)
        # read the image 
        image_pil = Image.open(image_path)
        # convert the image into numpy array
        image = np.array(image_pil, 'uint8')
        
        print(detector.detectMultiScale(image))
        # get the face from the image
        try: 
            face = detector.detectMultiScale(image)
            # draw the rectangle around the face
            for (x, y, w, h) in face:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('FACE to be trained', image)
                cv2.waitKey(100)

            # add the face to the list of faces
            faces.append(image[y:y+h,x:x+w])
            # add the label
            labels.append(label)
        except Exception as e:
            print(e)

    
    print(faces, labels)

    # train the recognizer
    recognizer.train(faces, np.array(labels))

    # save the trained data into trainer.yml
    recognizer.save('trainer.yml')

    cv2.destroyAllWindows()


def start_face_detection(names):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    cap=cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if os.path.exists('trainer.yml'):
        recognizer.read('trainer.yml')
   

    sample_images = 0

    # looping for live video feed
    while(True):
        # reading image captured from webcam
        ret, img = cap.read()
    
        # converting to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detecting the faces in the image using haar cascade classifier
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # for each face detected, draws a rectangle around it
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+2,y-5), font, 1, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(confidence), (x-30,y+h-5), font, 1, (0,255,0), 1, cv2.LINE_AA) 
            
            sample_images += 1
            print(sample_images)
        

        # displaying the image
        cv2.imshow('live-video-feed',img)
        
        # if the user presses q, it exits the loop 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # releasing the webcam
    cap.release()
    # destroying all the windows
    cv2.destroyAllWindows()


def gen_frames(names):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    cap=cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if os.path.exists('trainer.yml'):
        recognizer.read('trainer.yml')
   

    sample_images = 0

    # looping for live video feed
    while(True):
        # reading image captured from webcam
        ret, img = cap.read()
    
        # converting to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detecting the faces in the image using haar cascade classifier
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # for each face detected, draws a rectangle around it
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+2,y-5), font, 1, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(confidence), (x-30,y+h-5), font, 1, (0,255,0), 1, cv2.LINE_AA) 
            

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    pass
    # generate_dataset(4, max_samples=30)
    # train()
    # start_face_detection(['NONE','PJ'])