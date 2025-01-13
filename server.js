const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const multer = require('multer'); // Ajouter multer pour gérer les téléchargements de fichiers
const app = express();

// Chemin vers le script Python
const pythonScript = path.resolve(__dirname, 'image_process_node.py');
const pklPath = path.resolve(__dirname, 'output_features_30K.pkl');
const pythonAuthScript = path.resolve(__dirname, 'score_authenticite.py');

// Configuration de multer pour stocker les fichiers téléchargés
const upload = multer({ dest: 'uploads/' });

// Permet de servir des fichiers statiques (comme HTML, CSS)
app.use(express.static('public'));

// Route principale pour afficher la page HTML
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/index.html'));
});

// API pour exécuter le script Python avec un fichier téléchargé
app.post('/localize', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path; // Chemin de l'image téléchargée

    try {
        const result = await runPythonScript(imagePath, pklPath);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.toString() });
    }
});

// API pour évaluer l'authenticité de l'image
app.post('/authenticity', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;

    try {
        const result = await runPythonAuthScript(imagePath);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.toString() });
    }
});

// API pour exécuter les scripts Python de localisation et d'authenticité avec un fichier téléchargé
app.post('/analyze', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path; // Chemin de l'image téléchargée

    try {
        const localizationResult = await runPythonScript(imagePath, pklPath);
        const authenticityResult = await runPythonAuthScript(imagePath);
        res.json({ localization: localizationResult, authenticity: authenticityResult });
    } catch (error) {
        res.status(500).json({ error: error.toString() });
    }
});

// Fonction pour exécuter le script Python
function runPythonScript(imagePath, pklPath) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [pythonScript, imagePath, pklPath]);

        let result = '';
        let error = '';

        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            const message = data.toString();

            // Filtrer les erreurs non critiques
            const ignoredErrors = [
                'oneDNN custom operations are on', // Exemple d'erreur TensorFlow
                'slightly different numerical results', // Autre message fréquent
                'The name tf.losses.sparse_softmax_cross_entropy is deprecated', // Deprecated warning
            ];

            // Vérifiez si le message d'erreur contient une des erreurs ignorées
            const isIgnored = ignoredErrors.some((ignored) => message.includes(ignored));

            if (!isIgnored) {
                error += message; // Ajoutez uniquement les erreurs non ignorées
            }
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0 || error) {
                reject(`Erreur : ${error || `Code de sortie ${code}`}`);
            } else {
                try {
                    const parsed = JSON.parse(result);
                    resolve(parsed);
                } catch (err) {
                    reject(`Erreur de parsing JSON : ${err.message}`);
                }
            }
        });
    });
}

// Fonction pour exécuter le script Python d'authenticité
function runPythonAuthScript(imagePath) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [pythonAuthScript, imagePath]);

        let result = '';
        let error = '';

        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0 || error) {
                reject(`Erreur : ${error || `Code de sortie ${code}`}`);
            } else {
                try {
                    const parsed = JSON.parse(result);
                    resolve(parsed);
                } catch (err) {
                    reject(`Erreur de parsing JSON : ${err.message}`);
                }
            }
        });
    });
}

// Lancer le serveur
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Serveur en cours d'exécution sur http://localhost:${PORT}`);
});
