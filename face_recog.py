import numpy as np
import cv2
import os
from PIL import Image


class FaceRecognition:

    def __init__(self):
        self.harcasscase = './env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(self.harcasscase)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.mkdirs = ["anchors","positives","dataset","models"]
    
    def construct_directories(self):
        for dir in self.mkdirs:
            if not os.path.isdir(dir):
                os.mkdir(dir)

    def VideoFeed(self):
        cap = cv2.VideoCapture(0)
        cap.set(3,640) 
        cap.set(4,480)
        while True:
            ret, img = cap.read()
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
            yield img
        cap.release()
        cv2.destroyAllWindows()
    
    def CollectFaces(self):
        self.construct_directories()
        cap = cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
        pcount = 1
        acount = 1
        
        while True:
            ret,frame = cap.read()
            k = cv2.waitKey(30) & 0xff
            img = frame[120:120+250,200:200+250,:]
            if k == 27:
                break
            if k == ord('p'):
                cv2.imwrite('./positives/img_positive_{}.jpg'.format(pcount),img)
                print(f'Positive count - {pcount}')
                pcount += 1
            if k == ord('a'):
                cv2.imwrite('./anchors/img_aanchor_{}.jpg'.format(acount),img)
                print(f'Anchor count - {acount}')
                acount += 1
            cv2.imshow('face',img)
        cap.release()
        cv2.destroyAllWindows()
    
    def DetectFaces(self,img,labels=None,trained_model_path=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20, 20))
        if trained_model_path and len(labels) > 1:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(trained_model_path)
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                face_id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                if confidence < 100:
                    face_name = labels[face_id]
                else:
                    face_name = "unknown"
                cv2.putText(img, str(face_name), (x+5,y-5), self.font, 1, (255,0,0), 2)
        else:        
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        return img
    
    def CaptureFacesLive(self,max_frames=None,save_path=None):
        if max_frames and max_frames > 0:
            count = 0
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            for frame in self.VideoFeed():
                count = self.SaveImages(frame,count,save_path)
                print(f"Saved {count} of {max_frames} frames")
                if count == max_frames:
                    cv2.destroyAllWindows()
                    return
        else:
            for frame in self.VideoFeed():
                face_detected = self.DetectFaces(frame)
                cv2.imshow('Detected',face_detected)
    
    def RecognizeFacesLive(self,labels,trained_model_path):
        for frame in self.VideoFeed():
            face_recognized = self.DetectFaces(frame,labels,trained_model_path)
            cv2.imshow('Recognized',face_recognized)
        
        
    def SaveImages(self,img,count,savepath,face_id=1):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
        for (x,y,w,h) in faces:
            count += 1
            cv2.imwrite(f"./{savepath}/user_{face_id}_{count}.jpg",gray[y:y+h,x:x+w])
        return count
    
    def TrainFaces(self,model_name,path="dataset"):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces,ids = self.getImagesAndLabels(path,self.faceCascade)
        recognizer.train(faces,np.array(ids))
        recognizer.write(f'./models/{model_name}')
        print(f"{len(np.unique(ids))} faces trained.")
            
    def getImagesAndLabels(self,path,detector):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            face_id = int(os.path.split(imagePath)[-1].split("_")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(face_id)
        return faceSamples,ids