import cv2
import os
from insightface.app import FaceAnalysis
from numpy.linalg import norm 
import numpy as np


# Computes the cosine similarity between the two embeddings 
def cos_similarity(emb1, emb2):
    return emb1.dot(emb2) / (norm(emb1) * norm(emb2)) 


# Detect faces and extract embeddings 
def load_embedding_from_database_image(db_img_path, recognizer): 

    # Read the image 
    img = cv2.imread(db_img_path) 

    # Detect faces in this image 
    faces = recognizer.get(img) 

    # There should be exactly one face dectected per image 
    if len(faces) == 1: 
        # Extract embedding for the detected face 
        embedding = faces[0].embedding 
        return embedding 

# Initialize face recognition model      
def initialize_model(): 
    # Create a FaceAnalysis instance (the core interface of InsightFace) 
    # providers=['CPUExecutionProvider'] means we are using CPU. 
    # If want to use GUP, add 'CUDAExecutionProvider' at the beginning of the providers array to prioritize GPU 
    recognizer = FaceAnalysis(providers=['CPUExecutionProvider']) 
    # Use GPU (ctx_id=0) or CPU (ctx_id=-1)
    recognizer.prepare(ctx_id=-1) 

    return recognizer 


# face recognition function 
def face_recognition(image_path, db_img_path, threshold=0.5): 
    # Initialize the recognization model 
    recognizer = initialize_model() 

    # detect faces from the input image 
    img = cv2.imread(image_path)
    faces = recognizer.get(img) 

    if len(faces) == 0:
        print("No face detected in the input image.")
        return 
    
    elif len(faces) > 1:
        print("More than one faces detected in the input image.")
        return 
    
    # When there are exactly one face in the input image, compare its embedding with that of the database images  
    db_embedding = load_embedding_from_database_image(db_img_path, recognizer) 
    
    embedding = faces[0].embedding 

    sim_score = 0 
    
    print(type(db_embedding))
    # iterate through all database images to find the best matched image 
    if isinstance(db_embedding, np.ndarray): 
        sim_score = cos_similarity(embedding, db_embedding) 
    
    # output the result of recognizer 
    if sim_score > threshold: 
        return True 
    else: 
        return False 

        
# compares input image to the entire database. When integrating to our project, we could only compare with that one specific image. 
if __name__ == "__main__":
    # for n in ["Benedict.jpg", "Emma_Watson.jpg", "Jim_carrey.jpg", "Kim.jpg", "Leonardo.jpg", "Tom_Selleck.jpg"]: 
    #     image_path = "./test_images/" + n  
    #     database_folder = "./student_images"       
    #     face_recognition(image_path, database_folder)

    # image_path = "./test_images/Kiernan_Shipka.jpg"  
    # db_img_path = "./student_images/Mckenna_Grace.jpg"       
    image_path = "./test_images/Benedict.jpg"  
    db_img_path = "./student_images/Benedict.jpg" 
    result = face_recognition(image_path, db_img_path)
    print(result)
    

