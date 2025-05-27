# whisper-kirundi
Transcrivez automatiquement un chiffre en Kirundi (langue nationale du Burundi) à partir de votre voix

Ce projet fine-tune le modèle `openai/whisper-small` pour transcrire des fichiers audio contenant les nombres de 1 à 10 en kirundi. Il inclut :

- Rééchantillonnage des fichiers audio en 16kHz
- Création d'un dataset Hugging Face
- Fine-tuning du modèle
- Interface utilisateur avec Gradio

> ## 🚀 Tester le modèle en ligne

👉 [Accéder à l'application sur Hugging Face Spaces](https://huggingface.co/spaces/Buberintwari/whisper-kirundi)

## 🎧 Données audio

- 🔗 Audios originaux (mono) : [📁 audio/ sur Google Drive](https://drive.google.com/drive/folders/18DqujcI_po8jSruNBdL_xCLRBHVlyhmt?usp=sharing)
- 🔗 Audios rééchantillonnés (16kHz) : [📁 audio_16k/ sur Google Drive](https://drive.google.com/drive/folders/18hho5j58MGRQZoPmKcpjk8I6QGhAZewi?usp=sharing)

## 🤖 Modèle fine-tuné

- 📦 Hugging Face : [`whisper-kirundi-finetuned`](https://huggingface.co/Buberintwari/whisper-kirundi-finetuned)


## Lancer le projet

```bash
pip install -r requirements.txt
python app.py

