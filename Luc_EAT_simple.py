# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EAT (Luc)
# Ik heb het EAT model gebruikt voor het extraheren van audio embeddings. EAT (Efficient Audio Transformer) kwam ik tegen via de DCASE challenge, de bijbehorende code staat in de lokale EAT map (gecloned van GitHub). De getrainde weights (worstchan/EAT-base_epoch30_pretrain) zijn beschikbaar via HuggingFace, zoals vermeld in de README van de DCASE EAT repo.
#
# Net als de BEATs aanpak gebruik ik een pretrained audio model om 768-dimensionale embeddings te maken van elk geluidsbestand.
#
# Verder gebruik ik een Autoencoder om anomalieën te detecteren op basis van reconstructiefout, het model leert normaal geluid reconstrueren, en geeft een hogere fout bij afwijkend geluid.

# %%
import os
import sys
import torch
import torchaudio
import numpy as np

# voeg de lokale EAT map toe zodat python hem kan vinden
sys.path.insert(0, r"C:\Users\lucth\Downloads\Sorama Internship\EAT")

import torch.nn as nn
from transformers import AutoModel, AutoProcessor

# %% [markdown]
# ## Model laden
# Hier wordt het EAT model geladen vanuit HuggingFace. De `AutoModel` klasse haalt automatisch de getrainde weights (checkpoint) op. Dit zijn de weights waarmee het model al getraind is op een grote audiodataset, wij hoeven het model dus niet zelf te trainen.
#
# Als je een (NVIDIA) GPU hebt wordt CUDA automatisch gebruikt, anders draait het op de CPU.

# %%
# Laad het EAT model en de processor vanuit HuggingFace
MODEL_NAME = "worstchan/EAT-base_epoch30_pretrain"

import transformers

# Fix voor een bug tussen EAT en de nieuwere versie van transformers.
# EAT slaat intern een lijstje op waar transformers een dict verwacht,
# waardoor het model niet laadt. Dit zet het om naar een dict voordat
# transformers er iets mee probeert te doen. Wordt maar één keer uitgevoerd.
_PATCH_FLAG = "_eat_compat_patched"

if not getattr(transformers.PreTrainedModel, _PATCH_FLAG, False):
    _orig_adjust = transformers.PreTrainedModel.__dict__[
        "_adjust_tied_keys_with_tied_pointers"
    ]

    def _eat_adjust(self, missing_keys):
        v = self.__dict__.get("all_tied_weights_keys")
        if not isinstance(v, dict):
            self.__dict__["all_tied_weights_keys"] = {}
        return _orig_adjust(self, missing_keys)

    transformers.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _eat_adjust
    setattr(transformers.PreTrainedModel, _PATCH_FLAG, True)
    print("Compatibiliteitspatch toegepast.")
else:
    print("Patch was al actief, overgeslagen.")


eat_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
eat_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
eat_model.to(device)

print(f"Model geladen op: {device}")

# %% [markdown]
# ## Data voorverwerken
# Voordat een geluidsbestand door het EAT-model kan worden gestuurd, moet het eerst worden omgezet naar het juiste formaat. EAT verwacht geen ruwe audio, maar een mel spectrogram( een soort plaatje van het geluid waarbij de horizontale as de tijd voorstelt en de verticale as de frequentie).
#
# Eerst wordt het wav-bestand ingeladen en omgezet naar mono (één kanaal in plaats van stereo) en 16.000 Hz (16kHz). Dat laatste is de samplefrequentie, het aantal meetpunten per seconde. EAT is getraind op audio van 16kHz, dus andere samplefrequenties worden automatisch omgezet.
#
# Daarna wordt het mel spectrogram berekend met de Kaldi fbank-methode. Dit verdeelt het geluid in 128 frequentiebanden (mel-kanalen) en kijkt hoe sterk elk van die frequenties aanwezig is, stap voor stap in de tijd. Het resultaat is een matrix van tijdstappen × frequentiebanden.
#
# Hieronder in afbeelding 1 zie je een voorbeeld van een mel spectogram.
#
# ![image.png](attachment:image.png)
#
# *Afbeelding 1 - Dit is een mel spectrogram (bron: mathworks.com). De horizontale as is tijd, de verticale as is frequentie op de mel-schaal (die beter aansluit bij hoe het menselijk oor geluid waarneemt), en de kleur geeft de energie aan (hoe feller, hoe sterker die frequentie aanwezig is).*
#
# Vervolgens wordt de spectrogram op een vaste lengte van 512 tijdframes gebracht: te korte opnames worden aangevuld met nullen (padding), te lange worden afgeknipt. Tot slot wordt alles genormaliseerd, de waarden worden verschoven en geschaald met vaste getallen afkomstig uit de originele EAT-training, zodat de invoer dezelfde schaal heeft als waarop het model getraind is.

# %%
TARGET_LENGTH = 512   # aantal tijdframes per clip
NORM_MEAN = -4.268    # normalisatiewaarden komen uit de originele EAT training
NORM_STD = 4.569

def wav_to_mel(wav_path, target_length=TARGET_LENGTH):
    # laad het audiobestand
    waveform, sr = torchaudio.load(wav_path)

    # naar mono als het stereo is (2 kanalen -> 1)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample naar 16khz want dat verwacht EAT
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # zet audio om naar een mel spectrogram (soort heatmap van frequenties over tijd)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10
    )  # vorm: [T, 128]

    # pad of trim zodat elke clip even lang is
    T = fbank.shape[0]
    if T < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - T))
    else:
        fbank = fbank[:target_length, :]

    # normaliseer met de waarden van de EAT training
    fbank = (fbank - NORM_MEAN) / (NORM_STD * 2)

    # voeg batch en kanaal dimensie toe zodat het model het verwachte formaat krijgt
    return fbank.unsqueeze(0).unsqueeze(0)  # [1, 1, T, 128]


def extract_eat_embedding(wav_path):
    mel = wav_to_mel(wav_path).to(device)
    with torch.no_grad():
        # stuur de mel spectrogram door het model
        out = eat_model.extract_features(mel)  # [1, T, 768]
        # pak het cls-token (positie 0), dat is de samenvatting van het hele clip
        embedding = out[:, 0, :].squeeze(0).cpu().numpy()  # [768]
    return embedding


# %% [markdown]
# ## Embeddings extraheren uit de trainingsdata
# Nu heeft het EAT model 768 features voor elk geluidsbestand gehaald. Deze embeddings worden opgeslagen als `.npy` bestanden zodat we ze later opnieuw kunnen gebruiken zonder het model elke keer opnieuw te draaien.

# %%
input_folder = r"C:\Users\lucth\Downloads\Sorama Internship\Files Development Dataset\bearing\train"
output_folder = r"C:\Users\lucth\Downloads\Sorama Internship\Luc_EAT_features\bearing"
os.makedirs(output_folder, exist_ok=True)

for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            embedding = extract_eat_embedding(file_path)

            # zelfde mappenstructuur aanhouden in de output folder
            relative_path = os.path.relpath(root, input_folder)
            out_dir = os.path.join(output_folder, relative_path)
            os.makedirs(out_dir, exist_ok=True)

            # sla de embedding op als .npy bestand
            out_path = os.path.join(out_dir, os.path.splitext(file)[0] + ".npy")
            np.save(out_path, embedding)

            print(f"verwerkt: {file} -> {out_path}")

print("alle embeddings opgeslagen")

# %% [markdown]
# ## Autoencoder trainen (anomaliedetectie)
#  We trainen de autencoder  alleen op normaal geluid. Vervolgens zal hij bij abnormaal geluid een hogere reconstructiefout hebben, dat is dan het anomaliesignaal.
#
# De architectuur is hetzelfde als bij BEAT, (encoder: 768 → 64 (comprimeert de embedding), decoder: 64 → 768 (reconstrueert de originele embedding))

# %%
from torch.utils.data import DataLoader, TensorDataset

# laad alle opgeslagen .npy embeddings terug in
all_files = []
for root, _, files in os.walk(output_folder):
    for file in files:
        if file.endswith(".npy"):
            all_files.append(os.path.join(root, file))

embeddings_list = []
file_paths = []
for f in all_files:
    emb = np.load(f)
    embeddings_list.append(emb)
    file_paths.append(f)

# stapel alles tot één grote matrix
X = np.stack(embeddings_list)  # [N, 768]
X = torch.tensor(X, dtype=torch.float32)
print("geladen embeddings vorm:", X.shape)

# autoencoder: comprimeert 768 naar 64 en reconstrueert daarna terug naar 768
# hetzelfde als bij job
class AE(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)   # comprimeer
        x_hat = self.decoder(z)  # reconstrueer
        return x_hat

# dataloader shuffelt de data per epoch zodat het model beter leert
batch_size = 32
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# initialiseer model, optimizer en loss-functie
ae = AE().to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # mean squared error: hoe ver zit de reconstructie van het origineel

# train 50 rondes (epochs)
epochs = 50
for epoch in range(epochs):
    epoch_loss = 0
    for batch in loader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        x_hat = ae(x_batch)
        loss = loss_fn(x_hat, x_batch)
        loss.backward()   # bereken gradiënten
        optimizer.step()  # update gewichten
        epoch_loss += loss.item() * x_batch.size(0)
    epoch_loss /= len(loader.dataset)
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch+1}/{epochs}  loss: {epoch_loss:.6f}")

print("autoencoder klaar met trainen")

# %% [markdown]
# ## Anomaliescores berekenen en opslaan
# Nu berekenen we voor elk geluidsbestand hoe groot de reconstructiefout is met de getrainde autoencoder. Een hoge fout = waarschijnlijk een anomalie. De resultaten worden opgeslagen in een CSV-bestand.

# %%
import csv

output_csv = r"C:\Users\lucth\Downloads\Sorama Internship\Luc_EAT_features\anomaly_scores.csv"

ae.eval()  # zet model in evaluatiemodus (geen dropout, geen gradiënten nodig)
scores = []

with torch.no_grad():
    for emb, path in zip(X, file_paths):
        emb = emb.to(device)
        recon = ae(emb)
        # l2 afstand tussen origineel en reconstructie = anomaliescore
        # hoge score betekent dat het model het geluid moeilijk kon nabootsen
        score = torch.norm(emb - recon).item()
        scores.append((path, score))

# sla alles op in een csv
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["bestand", "anomaliescore"])
    writer.writerows(scores)

print(f"anomaliescores opgeslagen in: {output_csv}")

# toon de 5 bestanden met de hoogste score (meest afwijkend)
scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
print("\ntop 5 hoogste anomaliescores:")
for path, score in scores_sorted[:5]:
    print(f"  {os.path.basename(path)}: {score:.4f}")

# %% [markdown]
# **References**
#
# Afbeelding 1: Mel spectrogram - MATLAB. (n.d.). https://nl.mathworks.com/help/audio/ref/melspectrogram.html
#

# %% [markdown]
#
