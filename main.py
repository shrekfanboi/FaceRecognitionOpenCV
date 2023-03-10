from face_recog import FaceRecognition
import os
import cv2

def get_your_face_recognized(name,model_name="trained_faces.yml"):
    faceRecog = FaceRecognition()
    faceRecog.CollectFaces()
    faceRecog.CaptureFacesLive(max_frames=30,save_path="dataset")
    model_path = os.path.join(os.getcwd(),"models",model_name)
    faceRecog.TrainFaces(model_name=model_name)
    faceRecog.RecognizeFacesLive(labels=[None,name],trained_model_path=model_path)
    cv2.waitKey(0)