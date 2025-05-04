import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import constants as constants
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import cv2


# Custom RBC to grayscale preprocessing
def rbc_to_grayscale(image):
    """Convert image to enhanced grayscale emphasizing pest visibility"""
    # Convert to numpy array if it's a tensor
    if tf.is_tensor(image):
        image = image.numpy()

    # Special weighted conversion for pest visibility
    # Emphasizes red and blue channels where pests often contrast
    gray = 0.5 * image[..., 0] + 0.2 * image[..., 1] + 0.3 * image[..., 2]

    # Normalize and enhance contrast
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-7)
    gray_uint8 = (gray * 255).astype('uint8')

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_uint8)

    # Convert back to float32 and stack to 3 channels
    enhanced = enhanced.astype('float32') / 255.0
    return np.stack([enhanced, enhanced, enhanced], axis=-1)


def preprocess_function(image):
    """Main preprocessing function combining normalization and RBC conversion"""
    image = image / 255.0  # Normalize first
    return rbc_to_grayscale(image)


# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="reflect",
    validation_split=0.2
)

# Validation/Test data generator
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function,
    validation_split=0.2
)

# Load datasets
train_data = train_datagen.flow_from_directory(
    constants.TRAIN_PATH,
    target_size=constants.IMG_SIZE,
    batch_size=constants.BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_data = val_datagen.flow_from_directory(
    constants.TRAIN_PATH,
    target_size=constants.IMG_SIZE,
    batch_size=constants.BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

# Load Pretrained ResNet50
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Enhanced classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=output)

# Compile with custom optimizer settings
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=True
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
)

# Enhanced Callbacks
# Callbacks without EarlyStopping
callbacks = [
    ModelCheckpoint(
        constants.MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# Compute class weights for imbalanced data
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Initial training (frozen base)
initial_epochs = constants.EPOCHS // 2
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=initial_epochs,
    callbacks=callbacks,
    class_weight=class_weights
)

# Fine-tuning (unfreeze top layers)
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    initial_epoch=history.epoch[-1] + 1,
    epochs=constants.EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Evaluation
loss, accuracy = model.evaluate(val_data)
print(f"\nValidation Accuracy: {accuracy:.2%}")

# Save final model
model.save("final_pest_model_rbc_preprocessed.h5")
print("Model saved as final_pest_model_rbc_preprocessed.h5")


# Visualization
def plot_results(initial_history, fine_tune_history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(initial_history.history['accuracy'] + fine_tune_history.history['accuracy'],
             label='Training Accuracy')
    plt.plot(initial_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'],
             label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(initial_history.history['loss'] + fine_tune_history.history['loss'],
             label='Training Loss')
    plt.plot(initial_history.history['val_loss'] + fine_tune_history.history['val_loss'],
             label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


plot_results(history, fine_tune_history)


# Visualize preprocessing
def visualize_preprocessing_samples():
    sample_images, _ = next(train_data)

    plt.figure(figsize=(15, 6))
    for i in range(3):
        # Original (reverse normalization)
        plt.subplot(2, 3, i + 1)
        plt.imshow(sample_images[i] * 0.5 + 0.5)
        plt.title(f"Original Sample {i + 1}")
        plt.axis('off')

        # Processed (show first channel)
        plt.subplot(2, 3, i + 4)
        plt.imshow(sample_images[i][..., 0], cmap='gray')
        plt.title(f"Processed Sample {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('preprocessing_samples.png')
    plt.show()


visualize_preprocessing_samples()