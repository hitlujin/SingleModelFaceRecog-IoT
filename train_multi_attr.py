
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from celebAload import load_celebA_data
from faceattr import build_face_attribute_model
from blazeface.py import BlazeFaceDetector
def train_multi_attributes(epochs, batch_size):
    # Load data
    x_train, y_train = load_celebA_data()  # Assuming this function exists and is properly implemented
    # Load pre-trained face detection model
    face_detector = BlazeFaceDetector()
    face_detector.load_weights('multi_task_face_model.h5')
    # Build the multi-attribute model
    model = build_face_attribute_model(face_detector)
    
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Training the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    # Save the trained model
    model.save('multi_task_face_attributes_model.h5')
if __name__ == "__main__":
    train_multi_attributes(epochs=10, batch_size=32)