import tensorflow as tf
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
# Model paths
BINARY_TFLITE_PATH = "models/binary_model.tflite"
MULTI_TFLITE_PATH = "models/multi_model.tflite"

DATASET_PATH = r"D:\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset\Test"

# Image sizes
BINARY_IMG_SIZE = 224
MULTI_IMG_SIZE = 384

BATCH_SIZE = 32  # not directly used here, but kept for reference

# Random sampling: number of random images per class for each task
# For binary: we sample from both notumor (class 0) and tumor (class 1) separately
# For multi: sample from each of the 3 classes (glioma, meningioma, pituitary)
RANDOM_SAMPLES_PER_CLASS = 20     # change this to any number (e.g., 5, 20, 50)

# Multi-class class order (exactly as model was trained)
MULTI_CLASSES = ['glioma', 'meningioma', 'pituitary']  # index 0,1,2

# Binary mapping: notumor -> 0, any tumor -> 1
# ==================================================

def load_and_preprocess_image(path, target_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()

def evaluate_tflite_model(interpreter, images, true_labels, is_binary=True):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    for img in images:
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output[0])
    
    predictions = np.array(predictions)
    
    if is_binary:
        y_pred_proba = predictions.flatten()
        y_pred_class = (y_pred_proba >= 0.5).astype(int)
        y_true = true_labels
        
        acc = accuracy_score(y_true, y_pred_class)
        cm = confusion_matrix(y_true, y_pred_class)
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = float('nan')
        
        eps = 1e-7
        y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1-eps)
        loss = -np.mean(y_true * np.log(y_pred_proba_clipped) + (1 - y_true) * np.log(1 - y_pred_proba_clipped))
        
        print("\n🔹 Binary Classification Metrics (Random Sample) 🔹")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Loss (BCE): {loss:.4f}")
        print(f"AUC:       {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_true, y_pred_class, target_names=['notumor', 'tumor']))
    else:
        y_pred_proba = predictions
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        y_true = true_labels
        
        acc = accuracy_score(y_true, y_pred_class)
        cm = confusion_matrix(y_true, y_pred_class)
        
        eps = 1e-7
        y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1-eps)
        loss = -np.mean(np.log(y_pred_proba_clipped[np.arange(len(y_true)), y_true]))
        
        print("\n🔹 Multi-class Classification Metrics (Random Sample) 🔹")
        print(f"Accuracy: {acc:.4f}")
        print(f"Loss (CCE): {loss:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_class, target_names=MULTI_CLASSES))
    
    return {'accuracy': acc, 'loss': loss}

def get_random_images_and_labels(dataset_path, multi_classes, samples_per_class):
    """
    For binary: randomly sample `samples_per_class` from notumor folder and from the combined tumor folders.
    For multi: randomly sample `samples_per_class` from each of the multi_classes folders.
    Returns:
        binary_images, binary_labels, multi_images, multi_labels
    """
    binary_images = []
    binary_labels = []
    multi_images = []
    multi_labels = []
    
    # Collect all image paths per class
    tumor_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f.lower() != 'notumor']
    notumor_folder = 'notumor' if 'notumor' in [f.lower() for f in os.listdir(dataset_path)] else None
    
    # For binary sampling: get notumor images
    if notumor_folder:
        notumor_path = os.path.join(dataset_path, notumor_folder)
        notumor_images = []
        for fname in os.listdir(notumor_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                notumor_images.append(os.path.join(notumor_path, fname))
        random.shuffle(notumor_images)
        notumor_sampled = notumor_images[:samples_per_class]
        for img_path in notumor_sampled:
            img = load_and_preprocess_image(img_path, (BINARY_IMG_SIZE, BINARY_IMG_SIZE))
            binary_images.append(img)
            binary_labels.append(0)
    
    # For binary sampling: get tumor images (from all tumor folders)
    tumor_images = []
    for folder in tumor_folders:
        folder_path = os.path.join(dataset_path, folder)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                tumor_images.append(os.path.join(folder_path, fname))
    random.shuffle(tumor_images)
    tumor_sampled = tumor_images[:samples_per_class]
    for img_path in tumor_sampled:
        img = load_and_preprocess_image(img_path, (BINARY_IMG_SIZE, BINARY_IMG_SIZE))
        binary_images.append(img)
        binary_labels.append(1)
    
    # For multi sampling: sample from each multi_class folder
    for class_name in multi_classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: {class_path} not found, skipping multi-class sampling for {class_name}")
            continue
        class_images = []
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                class_images.append(os.path.join(class_path, fname))
        random.shuffle(class_images)
        sampled = class_images[:samples_per_class]
        for img_path in sampled:
            img_multi = load_and_preprocess_image(img_path, (MULTI_IMG_SIZE, MULTI_IMG_SIZE))
            multi_images.append(img_multi)
            multi_labels.append(multi_classes.index(class_name))
    
    return (np.array(binary_images), np.array(binary_labels),
            np.array(multi_images), np.array(multi_labels))

def main():
    print("Loading TFLite models...")
    binary_interpreter = tf.lite.Interpreter(model_path=BINARY_TFLITE_PATH)
    binary_interpreter.allocate_tensors()
    multi_interpreter = tf.lite.Interpreter(model_path=MULTI_TFLITE_PATH)
    multi_interpreter.allocate_tensors()
    
    print(f"Randomly sampling {RANDOM_SAMPLES_PER_CLASS} images per class...")
    binary_imgs, binary_labels, multi_imgs, multi_labels = get_random_images_and_labels(
        DATASET_PATH, MULTI_CLASSES, RANDOM_SAMPLES_PER_CLASS
    )
    
    print(f"Binary test samples (notumor: {RANDOM_SAMPLES_PER_CLASS}, tumor: {RANDOM_SAMPLES_PER_CLASS}) = {len(binary_imgs)}")
    print(f"Multi test samples (per class {RANDOM_SAMPLES_PER_CLASS} from {MULTI_CLASSES}) = {len(multi_imgs)}")
    
    if len(binary_imgs) > 0:
        print("\n" + "="*50)
        print("Evaluating Binary VGG19 (TFLite) on random images")
        print("="*50)
        evaluate_tflite_model(binary_interpreter, binary_imgs, binary_labels, is_binary=True)
    else:
        print("No binary test images sampled.")
    
    if len(multi_imgs) > 0:
        print("\n" + "="*50)
        print("Evaluating Multi-class VGG19 (TFLite) on random images")
        print("="*50)
        evaluate_tflite_model(multi_interpreter, multi_imgs, multi_labels, is_binary=False)
    else:
        print("No multi-class test images sampled.")

if __name__ == "__main__":
    main()