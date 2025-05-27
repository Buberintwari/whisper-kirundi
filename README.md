# whisper-kirundi
Transcrivez automatiquement un chiffre en Kirundi à partir de votre voix

Ce projet fine-tune le modèle `openai/whisper-small` pour transcrire des fichiers audio contenant les nombres de 1 à 10 en kirundi. Il inclut :

- Rééchantillonnage des fichiers audio en 16kHz
- Création d'un dataset Hugging Face
- Fine-tuning du modèle
- Interface utilisateur avec Gradio

## Lancer le projet

```bash
pip install -r requirements.txt
python app.py

