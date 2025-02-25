# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, enable=True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.compat.v1.Session(config=config)

# %%
"""
### Model for colorizing
"""

# %%
def unet_model_colorization(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4_2)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv3, up2], axis=3)
    conv7_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7_2)
    
    up3 = UpSampling2D(size=(2, 2))(conv7)
    merge3 = concatenate([conv2, up3], axis=3)
    conv8_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8_2)
    
    up4 = UpSampling2D(size=(2, 2))(conv8)
    merge4 = concatenate([conv1, up4], axis=3)
    conv9_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9_2)
    
    output = Conv2D(2, (1, 1), activation='tanh')(conv9)

    model_output = concatenate([inputs[...,0:1], output], axis=-1)

    model = Model(inputs=inputs, outputs=model_output)
    return model

# %%
"""
### Training
"""

# %%
def preprocess_input(image):
    # Resize the image and normalize
    image_rescaled = (image).astype(np.uint8)
    image_copy = cv2.resize(image_rescaled, (256, 256))
    image_copy = image_copy / 255.0  # Normalize to [0, 1]
    return image_copy[..., np.newaxis]

def preprocess_target(image):
    # Resize the image and convert to LAB color space
    image_rescaled = (image).astype(np.uint8)
    image_copy = cv2.resize(image_rescaled, (256, 256))
    lab_image = cv2.cvtColor(image_copy, cv2.COLOR_RGB2LAB)
    
    lab_image = lab_image.astype(np.float32)

    # Normalize L channel to [0, 1]
    l_channel = lab_image[..., 0:1] / 255.0
    
    # Normalize ab channels to [-1, 1]
    ab_channels = (lab_image[..., 1:3] - 128) / 128.0

    # Combine L and ab channels into one array with shape (256, 256, 3)
    combined_output = np.concatenate([l_channel, ab_channels], axis=-1)

    return combined_output

# %%
file_path = "data"
input_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
target_datagen = ImageDataGenerator()

train_input_generator = input_datagen.flow_from_directory(
    file_path + "/train",
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode=None,
    batch_size=4,
    shuffle=False,
)

train_target_generator = target_datagen.flow_from_directory(
    file_path + "/train",
    target_size=(256, 256),
    class_mode=None,
    batch_size=4,
    shuffle=False,
)

# %%
val_input_generator = input_datagen.flow_from_directory(
    file_path + "/validation",
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode=None,
    batch_size=4,
    shuffle=False,
)

val_target_generator = target_datagen.flow_from_directory(
    file_path + "/validation",
    target_size=(256, 256),
    class_mode=None,
    batch_size=4,
    shuffle=False,
)

# %%
def combined_generator(input_gen, target_gen):
    while True:
        input_batch = next(input_gen)
        target_batch = next(target_gen)

        # Preprocess the target batch
        processed_target_batch = np.array([preprocess_target(img) for img in target_batch])
        yield input_batch, processed_target_batch

# Create the combined generator
train_generator = combined_generator(train_input_generator, train_target_generator)
val_generator = combined_generator(val_input_generator, val_target_generator)

# %%
model = unet_model_colorization()

# from tensorflow.keras.models import load_model
# model = load_model("colorization_model.keras", custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError})

# %%
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
                loss=tf.keras.losses.MeanSquaredError(), 
                metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])

# %%
checkpoint_directory = "training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

# %%
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# %%
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
model.fit(train_generator, 
          steps_per_epoch=len(train_input_generator), 
          validation_data=val_generator,
          validation_steps=len(val_input_generator),
          epochs=2, batch_size=None,
          callbacks=[tensorboard_callback])

model.save("colorization_model.keras", save_format="keras")

# %%
checkpoint.save(file_prefix=checkpoint_prefix)

# %%
"""
### Testing
"""

# %%
# from tensorflow.keras.models import load_model

# model = load_model("colorization_model.keras", custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError})

# %%
test_input_generator = input_datagen.flow_from_directory(
    file_path + "/test",
    target_size=(256, 256),
    color_mode='grayscale',
    class_mode=None,
    batch_size=4,
    shuffle=False,
)

test_target_generator = target_datagen.flow_from_directory(
    file_path + "/test",
    target_size=(256, 256),
    class_mode=None,
    batch_size=4,
    shuffle=False,
)

# %%
test_generator = combined_generator(test_input_generator, test_target_generator)
model.evaluate(test_generator, steps=len(test_input_generator))

# %%
def postprocess_lab(l_channel_with_ab):
    l_channel = l_channel_with_ab[..., 0:1]   # Extract L channel
    ab_channels = l_channel_with_ab[..., 1:3] # Extract ab channels
    lab_image = np.concatenate([l_channel * 255.0, (ab_channels + 1) * 128], axis=-1) # Convert back to LAB scale
    lab_image = lab_image.astype(np.uint8)
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB) # Function to convert LAB to RGB
    return rgb_image

# %%
input_files = sorted(os.listdir(file_path + "/test/class_0"))
target_files = sorted(os.listdir(file_path + "/test/class_0"))

num_images_to_sample = int(0.001 * len(input_files))
if num_images_to_sample == 0:
    num_images_to_sample = max(len(input_files), 10)
sample_indices = np.random.choice(len(input_files), num_images_to_sample, replace=False)

X_train = []
Y_train = []

for idx in sample_indices:
    input_image_path = os.path.join(file_path + "/test/class_0", input_files[idx])
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    processed_input = preprocess_input(input_image)
    X_train.append(processed_input)

    target_image_path = os.path.join(file_path + "/test/class_0", target_files[idx])
    target_image = cv2.imread(target_image_path)
    target_image = cv2.resize(target_image, (256, 256))

    Y_train.append(target_image)

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.uint8)

# %%
cv2.destroyAllWindows()

# %%
original_images = []
predicted_images = []
input_images = []

for idx in range(0, len(X_train)):
    # Get the grayscale image (L channel)
    l_channel = X_train[idx]
    
    # Expand dimensions to match model input shape
    l_channel_expanded = np.expand_dims(l_channel, axis=0)  # Shape (1, 256, 256, 1)
    l_channel_expanded = l_channel_expanded[..., np.newaxis]  # Shape (1, 256, 256, 1, 1)
    
    # Predict the ab channels
    predicted_ab = model.predict(l_channel_expanded)[0]
    # Post-process to get RGB image
    rgb_image = postprocess_lab(predicted_ab)  # Assuming postprocess_lab is defined

    # Store the original L channel (scaled back to [0, 255]) and predicted RGB image
    original_images.append(Y_train[idx])  # Scale back to [0, 255]
    predicted_images.append(rgb_image)
    input_images.append(l_channel)

# Display original and predicted images side by side
fig, axes = plt.subplots(num_images_to_sample, 3, figsize=(10, (num_images_to_sample + 1) * 5))

for i in range(num_images_to_sample):
    axes[i, 0].imshow(cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title('Original colored Image')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(predicted_images[i])
    axes[i, 1].set_title('Predicted Colorized Image')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(input_images[i], cmap='gray')
    axes[i, 2].set_title('Input Image')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()
cv2.destroyAllWindows()
