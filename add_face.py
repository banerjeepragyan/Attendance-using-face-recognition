import cv2
import os
import joblib
import numpy as np
import time
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from keras_facenet import FaceNet

try:os.makedirs('faces')
except:pass

try:os.makedirs('faces/train')
except:pass

try:os.makedirs('faces/val')
except:pass

name = input('Enter your name --> ')
id = input('Enter your id no --> ')
name = id

if id in os.listdir('faces/train'):
    print('User already exist in faces')

else:    
    os.makedirs('faces/train/'+id)
    os.makedirs('faces/val/'+id)

    cap = cv2.VideoCapture(0)
    i = 0
    print()
    for i in range(5):
        print(f'Capturing starts in {5-i} seconds...')
        time.sleep(1)
    print('Taking photos...')
    while i<=200:
        ret,frame = cap.read()
        cv2.imshow('taking your pictures',frame)
        if i%5==0 and i<=150 and i!=0:
            cv2.imwrite('faces/train/'+id+'/'+str(i)+'.png',frame)
        elif i%5==0 and i>150:
            cv2.imwrite('faces/val/'+id+'/'+str(i)+'.png',frame)
        i+=1
            
        if cv2.waitKey(1)==27:  
            break

    cv2.destroyAllWindows()
    cap.release()
    print('Successfully taken your photos...')

embedder = FaceNet()
print('Embedding Model Loaded')

detector = MTCNN()

def find_face(img,img_size=(160,160)):
    img = cv2.imread(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.asarray(img) 
    faces = detector.detect_faces(img)
    if faces:
        x,y,w,h = faces[0]['box']
        x,y=abs(x),abs(y)
        face = img[y:y+h,x:x+w]
        face = Image.fromarray(face) 
        face = face.resize(img_size) 
        face = np.asarray(face)      
        return face
    return None

def embed(face):
    face = face.astype('float32')
    fm,fs = face.mean(),face.std()
    face = (face-fm)/fs 
    face = np.expand_dims(face,axis=0)
    embs = embedder.embeddings(face)
    return embs[0]

def load_dataset(path):
    X = []
    y = []
    for people in os.listdir(path):
        for people_images in os.listdir(path+people):
            face = find_face(path+people+'/'+people_images)
            if face is None:continue
            emb = embed(face)
            X.append(emb)
            y.append(people)
        print('Loaded {} images of {}'.format(len(os.listdir(path+'/'+people)),people)) 
    return np.asarray(X),np.asarray(y)

print('Loading train data...')
X_train, y_train = load_dataset('faces/train/')

print()

print('Loading test data...')
X_test, y_test = load_dataset('faces/val/')

l2_normalizer = Normalizer('l2')

X_train = l2_normalizer.transform(X_train)
X_test  = l2_normalizer.transform(X_test)

label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train)
y_test = label_enc.transform(y_test)

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

joblib.dump(rfc,'models/face_prediction_model.sav')
print()

print('Random Forest Model saved successfully!!')