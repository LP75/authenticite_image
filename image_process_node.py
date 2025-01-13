import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import torch
import io
from scipy.spatial.distance import cdist
import os
import sys
import json
import warnings

warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_delf_model():
    delf_model = hub.load('https://tfhub.dev/google/delf/1')
    return delf_model

def preprocess_image_for_delf(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image, dtype=np.float32)
    image_np /= 255.0
    return image_np

def extract_delf_features(image_bytes, delf_model):
    try:
        image_np = preprocess_image_for_delf(image_bytes)
        tf_image = tf.convert_to_tensor(image_np, dtype=tf.float32)
        tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)

        delf_signature = delf_model.signatures['default']
        delf_output = delf_signature(
            image=tf_image, 
            image_scales=tf.constant([1.0]), 
            score_threshold=tf.constant(100.0),
            max_feature_num=tf.constant(1000)
        )

        locations = delf_output['locations'].numpy()
        descriptors = delf_output['descriptors'].numpy()
        return locations, descriptors
    except Exception as e:
        return None, None

def extract_resnet_features(image_bytes):
    weights = ResNet50_Weights.DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=weights).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(image_tensor).squeeze().cpu().numpy()
        return features
    except Exception as e:
        return None

def load_pkl_features(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['metadata']

def find_nearest_location(query_features, database_features, metadata, method='cosine'):
    distances = cdist([query_features], [item['resnet'] for item in database_features], metric=method)
    nearest_idx = np.argmin(distances)
    nearest_metadata = metadata[nearest_idx]
    return nearest_metadata, distances[0][nearest_idx]

def localize_image(image_path, pkl_path):
    features, metadata = load_pkl_features(pkl_path)

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    query_resnet_features = extract_resnet_features(image_bytes)
    if query_resnet_features is None:
        return None

    nearest_metadata, distance = find_nearest_location(query_resnet_features, features, metadata)
    return {'latitude': nearest_metadata['latitude'], 'longitude': nearest_metadata['longitude'], 'distance': distance}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python script.py <image_path> <pkl_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    pkl_path = sys.argv[2]

    try:
        result = localize_image(image_path, pkl_path)
        if result:
            print(json.dumps({
                "latitude": result['latitude'],
                "longitude": result['longitude'],
                "distance": result['distance']
            }))
        else:
            print(json.dumps({"error": "Failed to process the image."}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))