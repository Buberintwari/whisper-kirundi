# app.py
# 1. Importation des librairies
import os
import torch
import torchaudio
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import gradio as gr

# 2. Préparation des fichiers audio
# Rééchantillonnage de tous les fichiers audio vers 16kHz
def resample_audio_to_16k(input_path, output_path):
    waveform, original_sr = torchaudio.load(input_path)
    resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=16000)
    resampled_waveform = resampler(waveform)
    torchaudio.save(output_path, resampled_waveform, 16000)

# Appliquer à tous les fichiers dans audio/
os.makedirs("audio_16k", exist_ok=True)
for i in range(1, 11):
    in_path = f"audio/{i}.wav"
    out_path = f"audio_16k/{i}.wav"
    resample_audio_to_16k(in_path, out_path)

# 3. Création du dataset Hugging Face

kirundi_numbers = ["rimwe", "kabiri", "gatatu", "kane", "gatanu",
                   "gatandatu", "indwi", "umunani", "icenda", "icumi"]

data = [{"audio": f"audio_16k/{i}.wav", "text": kirundi}
        for i, kirundi in enumerate(kirundi_numbers, start=1)]

dataset = Dataset.from_list(data).cast_column("audio", Audio())


# 4. Prétraitement avec WhisperProcessor

model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

def make_train_features(batch):
    input_features = processor(
        batch["audio"]["array"],
        sampling_rate=batch["audio"]["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    labels = processor.tokenizer(batch["text"]).input_ids
    labels = torch.tensor(labels)

    batch["input_features"] = input_features
    batch["labels"] = labels
    return batch

prepared_dataset = dataset.map(make_train_features, remove_columns=["audio", "text"])

# 5. Configuration de l'entraînement

training_args = TrainingArguments(
    output_dir="./whisper-kirundi",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    save_steps=10,
    logging_steps=5,
    learning_rate=1e-5,
    fp16=True,
    push_to_hub=False,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_dataset,
    tokenizer=processor.feature_extractor  # ou processor.tokenizer
)

# 6. Entraînement du modèle

trainer.train()

# 7. Sauvegarde du modèle fine-tuné

model.save_pretrained("./whisper-kirundi-finetuned")
processor.save_pretrained("./whisper-kirundi-finetuned")

#  8. Transcription d'un audio avec modèle fine-tuné
# Recharger le modèle entraîné
model = WhisperForConditionalGeneration.from_pretrained("./whisper-kirundi-finetuned")
processor = WhisperProcessor.from_pretrained("./whisper-kirundi-finetuned")
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

def transcribe(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# 9. Interface Gradio pour tester
gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Téléversez un fichier audio (16kHz, mono)"),
    outputs=gr.Textbox(label="Transcription en Kirundi"),
    title="Whisper en Kirundi",
    description="Transcrivez automatiquement des audios en Kirundi (chiffres 1 à 10)."
).launch()
