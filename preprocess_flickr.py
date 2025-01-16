import os
import pickle
import msgpack
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import torch
import io
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Chargement du modèle DELF
def load_delf_model():
    print("Chargement du modèle DELF...")
    delf_model = hub.load('https://tfhub.dev/google/delf/1')
    print("Modèle DELF chargé.")
    return delf_model

# Fonction pour extraire les caractéristiques via DELF
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
        print(f"Erreur avec DELF : {e}")
        return None, None

def preprocess_image_for_delf(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image, dtype=np.float32)
    image_np /= 255.0
    return image_np

# Fonction pour extraire les caractéristiques via ResNet
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
        print(f"Erreur avec ResNet: {e}")
        return None

# Charge les données du dataset d'images
def load_messagepack(filepath, limit=None):
    """
    Charge les données MessagePack depuis un fichier.
    Peut limiter le nombre d'éléments chargés.
    """
    with open(filepath, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=True)
        data = []
        for i, item in enumerate(unpacker):
            if limit is not None and i >= limit:
                break
            data.append(item)
    print(f"{len(data)} éléments chargés depuis le fichier {filepath}.")
    return data

def save_features_to_pkl(msgpack_path, pkl_path):
    print("Chargement des données...")

    data = load_messagepack(msgpack_path)
    delf_model = load_delf_model()

    feature_list = []
    metadata_list = []

    for item in tqdm(data, desc="Processing Images", unit="image"):

        image_bytes = item.get(b'image')
        if not image_bytes:
            print(f"Aucune image trouvée dans cet item : {item}")
            continue

        latitude = item.get(b'latitude')
        longitude = item.get(b'longitude')
        if latitude is None or longitude is None:
            print(f"Coordonnées GPS manquantes dans cet item : {item}")
            continue

        item_id = item.get(b'id', b'').decode('utf-8')

        # Extraire les caractéristiques ResNet
        resnet_features = extract_resnet_features(image_bytes)
        if resnet_features is None:
            continue

        # Extraire les caractéristiques DELF
        delf_features = extract_delf_features(image_bytes, delf_model)
        if delf_features is None:
            continue

        # Sauvegarde des caractéristiques
        feature_list.append({'resnet': resnet_features, 'delf': delf_features})
        metadata_list.append({'id': item_id, 'latitude': latitude, 'longitude': longitude})

    with open(pkl_path, 'wb') as f:
        pickle.dump({'features': feature_list, 'metadata': metadata_list}, f)

    print(f"Caractéristiques sauvegardées dans {pkl_path}")


# Exécution
save_features_to_pkl('flickr/shard_0.msg', 'output_features_30K.pkl')