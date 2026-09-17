import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, MobileNetV2

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



# Define Configuration

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 5
TRAIN_DIR = "dataset/Forect Fire/Forest Fire_Dataset/train"
TEST_DIR = "dataset/Forect Fire/Forest Fire_Dataset/test"
VAL_DIR = "dataset/Forect Fire/Forest Fire_Dataset/val"

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed = 42
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = False
)

class_names = train_ds.class_names
print("Classes:", class_names)

for images,labels in train_ds.take(1):
    print("Image shape",images.shape)
    print("Label shape",labels.shape)
    print("labels:",labels.numpy())

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)

])


vgg_base_model = VGG16(
    weights = "imagenet",
    include_top = False,
    input_shape = (224,224,3)
)

vgg_base_model.trainable = False

vgg_model = tf.keras.Sequential([
    layers.Input(shape= (224,224,3)),
    data_augmentation,
    layers.Lambda(vgg_preprocess),
    vgg_base_model,
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(4,activation="softmax")
])

vgg_model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

vgg_model.summary()

print("  Training VGG16   ")

history = vgg_model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = EPOCHS
)



vgg_test_loss, vgg_test_accuracy = vgg_model.evaluate(
    test_ds
)

print(
    f"\nVGG16 Test Accuracy: "
    f"{vgg_test_accuracy:.2%}"
)

print(
    f"VGG16 Test Loss: "
    f"{vgg_test_loss:.4f}"
)


y_true = []
y_pred = []
for images,labels in test_ds:
    predictions = vgg_model.predict(
        images,
        verbose = 0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="blues"
)
plt.show()

print("\nVGG16 Classification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

def predict_image(model, image_path):

    image = tf.keras.utils.load_img(

        image_path,

        target_size=IMG_SIZE

    )


    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )


    image_array = np.expand_dims(

        image_array,

        axis=0

    )


    predictions = model.predict(

        image_array,

        verbose=0

    )[0]


    predicted_index = np.argmax(
        predictions
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = predictions[
        predicted_index
    ]


    print("\n" + "=" * 50)

    print("FIRE & SMOKE DETECTION")

    print("=" * 50)


    print(
        f"\nPrediction: "
        f"{predicted_class}"
    )


    print(
        f"Confidence: "
        f"{confidence:.2%}"
    )


    print("\nClass probabilities:")


    for class_name, probability in zip(

        class_names,

        predictions

    ):

        print(

            f"{class_name}: "
            f"{probability:.2%}"

        )


   
    if predicted_class == "fire":

        print(
            "\n🔥 FIRE WARNING!"
        )

    elif predicted_class == "smoke":

        print(
            "\n💨 SMOKE WARNING!"
        )

    elif predicted_class == "smokefire":

        print(
            "\n🚨 FIRE + SMOKE WARNING!"
        )

    else:

        print(
            "\n✅ NO FIRE DETECTED"
        )

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.title(

        f"{predicted_class} "
        f"({confidence:.2%})"

    )

    plt.axis("off")

    plt.show()
