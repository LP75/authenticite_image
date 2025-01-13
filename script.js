const { spawn } = require('child_process');
const path = require('path');

// Chemins vers le script Python et les fichiers d'entrée
const imagePath = path.resolve(__dirname, 'img/tour-eiffel.jpg');
const pklPath = path.resolve(__dirname, 'output_features.pkl');
const pythonScript = path.resolve(__dirname, 'image_process_node.py');

// Fonction pour exécuter le script Python
function runPythonScript(imagePath, pklPath) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [pythonScript, imagePath, pklPath]);

        let result = '';
        let error = '';

        // Collecte des données de sortie
        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });

        // Collecte des erreurs
        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });

        // Lorsque le processus est terminé
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

// Appeler le script et afficher les résultats
(async () => {
    try {
        const output = await runPythonScript(imagePath, pklPath);
        console.log('Résultat du script Python :', output);
    } catch (error) {
        console.error('Erreur lors de l\'exécution du script Python :', error);
    }
})();

