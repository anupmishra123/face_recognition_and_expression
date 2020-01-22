from mtcnn import MTCNN
face = MTCNN()

def face_detect(image):
    faces = face.detect_faces(image)
    face_detected=[]
    
    if len(faces) > 0:
        for x,y,w,h in faces:
            face_detected.append(image[y:y+h, x:x+w])
        return face_detected
    return "No face detected"
           
def face_locations(image):
    faces = face.detect_faces(image) 
    if len(faces) > 0:
        return faces
    return "No face detected"