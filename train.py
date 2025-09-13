import os
import random
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile
import shutil

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import image_dataset_from_directory

# ------------------ Reproducibility ------------------ #
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ------------------ GPU Memory Growth ------------------ #
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU config error:", e)

# ------------------ Paths ------------------ #
DATASET_DIR = r'C:\Users\bhuva\Videos\project\dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10

# ------------------ Flatten Nested Folder Structure ------------------ #
def get_flat_dataset(data_dir):
    class_dirs = []
    for crop in os.listdir(data_dir):
        crop_path = os.path.join(data_dir, crop)
        if os.path.isdir(crop_path):
            for disease in os.listdir(crop_path):
                disease_path = os.path.join(crop_path, disease)
                if os.path.isdir(disease_path):
                    class_dirs.append(disease_path)
    return class_dirs

def create_flat_symlinked_dataset(source_dirs):
    temp_dir = tempfile.mkdtemp()
    for disease_path in source_dirs:
        class_name = os.path.basename(disease_path)
        target_class_dir = os.path.join(temp_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        for fname in os.listdir(disease_path):
            src = os.path.join(disease_path, fname)
            dst = os.path.join(target_class_dir, fname)
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy(src, dst)
    return temp_dir

# Create flattened datasets
train_dirs = get_flat_dataset(TRAIN_DIR)
flat_train_dir = create_flat_symlinked_dataset(train_dirs)

val_dirs = get_flat_dataset(VAL_DIR)
flat_val_dir = create_flat_symlinked_dataset(val_dirs)

# ------------------ Load Dataset ------------------ #
train_ds = image_dataset_from_directory(
    flat_train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=seed
)

val_ds = image_dataset_from_directory(
    flat_val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected classes ({num_classes}): {class_names}")

# Save class indices
with open('class_indices.json', 'w') as f:
    json.dump({name: i for i, name in enumerate(class_names)}, f, indent=2)

# ------------------ Augmentation & Preprocessing ------------------ #
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomWidth(0.1),
    tf.keras.layers.RandomHeight(0.1)
])

train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y)).map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# ------------------ Build DenseNet121 Model ------------------ #
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# ------------------ Callbacks ------------------ #
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_densenet_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)

# ------------------ Phase 1 Training ------------------ #
print("\n=== Phase 1: Training with frozen DenseNet base ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# ------------------ Phase 2 Fine-Tuning ------------------ #
print("\n=== Phase 2: Fine-tuning DenseNet base ===")
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=history.epoch[-1] + 1,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# Merge histories
for key in fine_history.history:
    if key in history.history:
        history.history[key].extend(fine_history.history[key])
    else:
        history.history[key] = fine_history.history[key]

# ------------------ Evaluation ------------------ #
best_model = tf.keras.models.load_model('best_densenet_model.keras')
val_loss, val_acc, val_top_k_acc = best_model.evaluate(val_ds)

val_labels = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
val_preds_raw = best_model.predict(val_ds)
val_preds = np.argmax(val_preds_raw, axis=1)
val_true = np.argmax(val_labels, axis=1)

classification_rep = classification_report(val_true, val_preds, target_names=class_names)
conf_matrix = confusion_matrix(val_true, val_preds)

# ------------------ Save Outputs ------------------ #
with open('densenet_training_history.json', 'w') as f:
    json.dump({k: [float(v) for v in val] for k, val in history.history.items()}, f, indent=2)

with open('densenet_evaluation_metrics.json', 'w') as f:
    json.dump({
        'model_type': 'DenseNet121',
        'total_parameters': int(model.count_params()),
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'val_top_k_accuracy': float(val_top_k_acc),
        'classification_report': classification_report(val_true, val_preds, target_names=class_names, output_dict=True),
        'confusion_matrix': conf_matrix.tolist(),
        'class_names': class_names,
        'training_epochs': len(history.history['loss'])
    }, f, indent=2)

best_model.save('final_densenet_model.keras')

# ------------------ Visualization ------------------ #
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('DenseNet121 Training Results', fontsize=16, fontweight='bold')

axes[0, 0].plot(history.history['accuracy'], label='Train Acc')
axes[0, 0].plot(history.history['val_accuracy'], label='Val Acc')
axes[0, 0].set_title('Accuracy')
axes[0, 0].legend()

axes[0, 1].plot(history.history['loss'], label='Train Loss')
axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
axes[0, 1].set_title('Loss')
axes[0, 1].legend()

axes[1, 0].plot(history.history['top_k_categorical_accuracy'], label='Train Top-K Acc')
axes[1, 0].plot(history.history['val_top_k_categorical_accuracy'], label='Val Top-K Acc')
axes[1, 0].set_title('Top-K Accuracy')
axes[1, 0].legend()

axes[1, 1].axis('off')  # Empty

plt.tight_layout()
plt.savefig('densenet_training_results.png', dpi=300)
plt.show()

# ------------------ Cleanup ------------------ #
shutil.rmtree(flat_train_dir)
shutil.rmtree(flat_val_dir)

print("\n🎯 Training Complete")
print(f"Validation Accuracy: {val_acc:.4f}")
print("Files Saved:")
print("✓ best_densenet_model.keras")
print("✓ final_densenet_model.keras")
print("✓ densenet_training_history.json")
print("✓ densenet_evaluation_metrics.json")
print("✓ densenet_training_results.png")
print("✓ class_indices.json")
