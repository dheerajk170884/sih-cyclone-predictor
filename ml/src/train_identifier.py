import os
import tensorflow as tf
from tensorflow.keras import layers, models

def train_identifier():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "../data/Identifier")
    save_path = os.path.join(base_dir, "../models/cyclone_identifier.keras")

    # Load dataset (0: cyclone or non_cyclone based on folder names)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary"
    )

    # Simple Transfer Learning Model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="Cyclone_Identifier")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("Training Identifier Model...")
    model.fit(train_ds, validation_data=val_ds, epochs=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model successfully saved at: {save_path}")

if __name__ == "__main__":
    train_identifier()