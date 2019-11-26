"""
Put mp3 file in ./data/demo/song
Ensure other folders do not exist or are empty
"""

from pydub import AudioSegment
import os.path
import os

import subprocess
from PIL import Image

import librosa
import librosa.display

import numpy as np
import pylab

from audio_to_spectrogram import *

from load_data import load_data
from model import MusicClassifier
# from train import train
import torch

from torchvision import datasets, models, transforms

from shutil import move, rmtree
import torch.nn.functional as F

# Defines
genres = ["pop","rap", "classical"]
data_dir = "./data/demo/"
raw_audio_dir = data_dir + "song/"
audio_slices_dir = data_dir + "audio_slices/"
spectrograms_dir = data_dir + "spectrograms/"

if not os.path.exists(audio_slices_dir):
    os.makedirs(audio_slices_dir)
if not os.path.exists(spectrograms_dir):
    os.makedirs(spectrograms_dir)

for file in os.listdir(raw_audio_dir):
    file_ = os.fsdecode(file)
    if file_.endswith(".mp3"):
        filename = os.path.basename(file_)    # Remove path from file
        basename, ext = filename.split(".",1)      # Filename no extension

        mp3_raw_audio_path = raw_audio_dir + filename
        spectrograms_path = spectrograms_dir
        audio_slices_path = audio_slices_dir + basename + "/"
        # Subdirectiory in audio_slices for each audio file
        os.mkdir(audio_slices_path)
        print("###############################")
        print("#       INIT DIRECTORIES      #") 
        print("###############################")
        print("")
        print("> Audio file at: " + mp3_raw_audio_path)
        print("> Spectrogram files at: " + spectrograms_path)
        print("> Audio slices at: " + audio_slices_path)
        print("")
        print("###############################")
        print("#        PROCESS DATA         #") 
        print("###############################")
        print("")
        print("> Converting " + file_ + " to mono .wav file")
        
        audio_convert(mp3_raw_audio_path, "mp3", "wav", 1)
        wav_raw_audio_path = raw_audio_dir + basename+".wav"
        print("> Slicing wav file " + basename + ".wav into 10s segments")
        audio_slice(wav_raw_audio_path, audio_slices_path, 30)
        
        print("")
        print("###############################")
        print("#    GENERATE SPECTROGRAMS    #") 
        print("###############################")
        print("")
        
        for file in os.listdir(audio_slices_path):
            file_ = os.fsdecode(file)
            if file_.endswith(".wav"):
                filename = os.path.basename(file_)          # Remove path from file

                audio_slice_path = audio_slices_path + filename
                print("> Creating spectrogram of " + file_)

                audio_to_spectrogram(audio_slice_path, spectrograms_path)

model = MusicClassifier()
model.load_state_dict(torch.load("./music_classifier.pt"))

demo_dir = "./data/demo/"
data_transform = transforms.Compose([transforms.ToTensor()])

def image_loader(loader, image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image

probs = [0,0,0]
count = 0

for file in os.listdir(demo_dir + "spectrograms/"):
    file_ = os.fsdecode(file)
    if file_.endswith(".png"):
        print(file_)
        image = image_loader(data_transform, data_dir + "spectrograms/" + file_)
        output = model(image)
        prob = F.softmax(output, dim=1)
        print("probability of classical: %f" %  (prob[0][0]*100))
        print("probability of pop: %f" % (prob[0][1]*100))
        print("probability of rap: %f" % (prob[0][2]*100))
        probs[0] += prob[0][0]
        probs[1] += prob[0][1]
        probs[2] += prob[0][2]
        count += 1
print("------")
print("OVERALL CLASSIFICATION")
classification = {"Classical":probs[0]  / count * 100, "Pop":probs[1] / count * 100, "Rap": probs[2] / count * 100}
largest = 0
for genre in classification:
    if classification[genre] > largest:
        predicted_genre = genre
        largest = classification[genre]

print(predicted_genre)
print("probability of classical: %f" %  classification["Classical"])
print("probability of pop: %f" % classification["Pop"])
print("probability of rap: %f" % classification["Rap"])