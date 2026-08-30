import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB0

# Constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 6
BATCH_SIZE = 32

def build_cyclone_model(num_classes=NUM_CLASSES):
    # Pretrained EfficientNetB0 backbone
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Fine-tuning ke liye freeze rakhein initially

    # Classification Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model = build_cyclone_model()
    model.summary()
    print("TensorFlow EfficientNet-B0 Model built successfully!")
    
    # Model architecture save karne ke liye
    model.save("../models/cyclone_model_baseline.keras")
    print("Baseline model saved in models/ folder.")