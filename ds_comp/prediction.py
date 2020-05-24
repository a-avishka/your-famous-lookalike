import face_recognition
from pathlib import Path
from PIL import Image


class Prediction:
    def __init__(self, file):
        self.file = file

    def make_pred(self):
        face = face_recognition.load_image_file(self.file)

        face_encoded = face_recognition.face_encodings(face)[0]

        closest_face_distance = 1.0
        closest_face_image = None
        closest_face_image_path = ""
        for image_path in Path('static').glob('*.jpg'):

            new_image = face_recognition.load_image_file(image_path)

            new_image_encoded = face_recognition.face_encodings(new_image)
            try:
                new_image_distance = face_recognition.face_distance(new_image_encoded, face_encoded)[0]
                if new_image_distance < closest_face_distance:
                    closest_face_distance = new_image_distance
                    closest_face_image = new_image
                    closest_face_image_path = image_path.name

            except IndexError:
                print(image_path)

        closest_face = Image.fromarray(closest_face_image)
        # closest_face.show()
        print(1.0 - closest_face_distance)
        return closest_face_image_path
