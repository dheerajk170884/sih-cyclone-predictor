import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# Dataset Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def create_data_generators(data_dir, val_split=0.2):
    """
    data_dir: Folder jisme class-wise cyclone images stored hain
    Structure:
      data/
        ├── Depression/
        ├── Deep Depression/
        ├── Cyclonic Storm/
        ├── Severe Cyclonic Storm/
        ├── Very Severe Cyclonic Storm/
        └── Super Cyclonic Storm/
    """
    # Data Augmentation & Normalization
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_split
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

if __name__ == "__main__":
    sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    print(f"TensorFlow Data Pipeline configured for: {sample_data_path}")