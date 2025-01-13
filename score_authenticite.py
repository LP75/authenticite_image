import os
import cv2
import numpy as np
from PIL import Image
import piexif
from datetime import datetime
import json

# Vérifier les métadonnées EXIF et détecter l'utilisation de logiciels comme Photoshop
def check_exif(image_path):
    try:
        exif_data = piexif.load(image_path)
        
        # Vérification du logiciel
        software_used = exif_data["0th"].get(piexif.ImageIFD.Software, None)
        software_result = True
        software_message = ""
        
        if software_used:
            software_used = software_used.decode('utf-8')
            if 'Photoshop' in software_used:
                software_result = False
                software_message = f"Logiciel utilisé : {software_used}. L'image semble avoir été modifiée avec Photoshop."
            else:
                software_message = f"Logiciel utilisé : {software_used}. Le logiciel utilisé ne semble pas être suspect."
        else:
            software_message = "Les métadonnées EXIF sont absentes ou ne contiennent pas de champ 'Software'."
        
        # Vérification de la date avec gestion des différents emplacements possibles
        date_result = True
        date_message = ""
        datetime_original = None

        try:
            # Essayer différents emplacements possibles pour la date
            if "Exif" in exif_data:
                datetime_original = exif_data["Exif"].get(36867, None)  # 36867 est le code pour DateTimeOriginal
            if datetime_original is None and "0th" in exif_data:
                datetime_original = exif_data["0th"].get(306, None)  # 306 est le code pour DateTime
        
            if datetime_original:
                datetime_original = datetime.strptime(datetime_original.decode('utf-8'), '%Y:%m:%d %H:%M:%S')
                if datetime_original > datetime.now():
                    date_result = False
                    date_message = "La date de prise de vue est dans le futur. Cela pourrait suggérer une falsification."
                else:
                    date_message = f"Date de prise de vue : {datetime_original}"
            else:
                date_result = True  # On ne pénalise pas si la date n'est pas trouvée
                date_message = "La date de prise de vue n'est pas disponible dans les métadonnées EXIF."
        except Exception as date_error:
            date_result = True  # On ne pénalise pas en cas d'erreur de lecture de la date
            date_message = "Impossible de lire la date dans les métadonnées EXIF."

        # Retourner le résultat combiné
        return (software_result and date_result), f"{software_message} | {date_message}"

    except Exception as e:
        return False, f"Erreur lors de la lecture des métadonnées EXIF : {str(e)}"

# Analyser les artefacts de compression JPEG
def check_compression_artifacts(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour détecter les bords artificiels causés par la compression JPEG
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blurred)
    
    # Calculer l'histogramme des différences
    hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalisation de l'histogramme

    # Si l'histogramme montre des pics dans les petites valeurs (bords nets), il y a probablement des artefacts de compression
    if hist[10] > 0.05:  # Un seuil empirique pour les artefacts de compression
        return True, "L'image présente des artefacts de compression JPEG."
    else:
        return False, "Aucun artefact de compression détecté."

# Analyser l'histogramme des couleurs pour des anomalies
def check_histogram(image_path):
    img = Image.open(image_path)
    img_rgb = img.convert('RGB')

    # Convertir l'image en tableau NumPy
    img_array = np.array(img_rgb)

    # Calculer les histogrammes de chaque canal (R, G, B)
    hist_r = np.histogram(img_array[..., 0], bins=256, range=(0, 256))[0]
    hist_g = np.histogram(img_array[..., 1], bins=256, range=(0, 256))[0]
    hist_b = np.histogram(img_array[..., 2], bins=256, range=(0, 256))[0]

    # Vérifier les distributions des couleurs
    if np.std(hist_r) < 10 or np.std(hist_g) < 10 or np.std(hist_b) < 10:
        return False, "Les histogrammes de couleurs semblent anormaux (peu de variation). Cela pourrait suggérer une manipulation."
    else:
        return True, "Les histogrammes de couleurs semblent naturels."

# Évaluer le degré d'authenticité
def evaluate_authenticity(image_path):
    exif_check, exif_message = check_exif(image_path)
    compression_check, compression_message = check_compression_artifacts(image_path)
    histogram_check, histogram_message = check_histogram(image_path)

    # Calculer un score d'authenticité basé sur les résultats
    score = 0

    if exif_check:
        score += 1  # Présence de métadonnées EXIF et absence de Photoshop
    if not compression_check:  # Moins d'artefacts, mieux c'est
        score += 1
    if histogram_check:  # Si les histogrammes sont naturels
        score += 1

    authenticity_score = score / 3 * 100  # Le score d'authenticité est un pourcentage

    result = {
        "authenticity_score": authenticity_score,
        "exif_check": exif_message,
        "compression_check": compression_message,
        "histogram_check": histogram_message
    }

    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python score_authenticite.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    result = evaluate_authenticity(image_path)
    print(json.dumps(result))
