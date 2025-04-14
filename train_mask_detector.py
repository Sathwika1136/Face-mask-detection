# Import necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize constants
INIT_LR = 0.001  # Initial learning rate
EPOCHS = 5       # Reduced epochs for faster execution
BS = 128         # Increased batch size for efficiency
SUBSET_SIZE = 1000  # Limit dataset size for faster training

# Define the data directory and categories
DIRECTORY = r"C:\Users\chirr\Downloads\archive (1)\data"
CATEGORIES = ["with_mask", "without_mask"]

# Initialize data and labels
data, labels = [], []

print("[INFO] Loading images...")

# Loop through categories and load images
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    if os.path.exists(path):
        for i, img in enumerate(os.listdir(path)):
            if i >= SUBSET_SIZE // 2:  # Limit number of images per category
                break
            img_path = os.path.join(path, img)
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = preprocess_input(img_to_array(load_img(img_path, target_size=(224, 224))))
                    data.append(image)
                    labels.append(category)
                except Exception as e:
                    print(f"[ERROR] Failed to process {img_path}: {e}")

# Convert data and labels to numpy arrays
print(f"[INFO] Loaded {len(data)} images.")
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Convert labels to one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load MobileNetV2 with pre-trained weights
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Build the head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Place the head FC model on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] Compiling model...")
lr_schedule = ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=EPOCHS,
    decay_rate=0.9,
    staircase=True
)
opt = Adam(learning_rate=lr_schedule)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
print("[INFO] Training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# Evaluate the model
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Save the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.keras", save_format="keras")
# Plot training results
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
