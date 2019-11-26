from pydub import AudioSegment
import os.path
import os

import subprocess
from PIL import Image

import librosa
import librosa.display

import numpy as np
import pylab

# Defines
genres = ["pop","rap", "classical"]
data_dir = "./data/"
raw_audio_dir = data_dir + "{}" + "/raw_audio/" + "{}"
audio_slices_dir = data_dir + "{}" + "/audio_slices/" + "{}" + "/"
spectrograms_dir = data_dir + "{}" + "/spectrograms/" 

def audio_slice(in_path, out_path, segment_time):
    currentPath = os.path.dirname(os.path.realpath(__file__)) 
    audio_basename, ext = os.path.basename(in_path).split(".", 1)   # Extract the mp3 name
    command = "ffmpeg -i {} -f segment -segment_time 10 -c copy {}_%03d.wav".format(in_path, out_path + audio_basename)
    p = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True, cwd=currentPath)
    output, errors = p.communicate()
    if errors:
        print (errors)

def audio_convert(audio_file, base_fmt, to_fmt, channels):
    sound = AudioSegment.from_mp3(audio_file)
    sound = sound.set_channels(channels)

    wav_file_dir = os.path.dirname(audio_file) + "/"
    wav_file_name, ext = os.path.basename(audio_file).split(".", 1)
    wav_file_path = wav_file_dir + wav_file_name + ".wav"

    # print("> Saving .wav file at: " + wav_file_path)
    sound.export(wav_file_path, format=to_fmt)

def audio_to_spectrogram(audio_file, out_name):
    y, sr = librosa.load(audio_file)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    librosa.display.specshow(spect)
    audio_basename, ext = os.path.basename(audio_file).split(".", 1)
    pylab.savefig(out_name + audio_basename + ".png", bbox_inches=None, pad_inches=0)
    pylab.close()
    # Crop whitespace out of images
    spect_img = Image.open(out_name + audio_basename + ".png")
    spect_cropped = spect_img.crop((80,58,496,370))
    spect_resized = spect_cropped.resize((224, 224), Image.NEAREST)  
    spect_resized.save(out_name + audio_basename + ".png", "png")

"""
for genre in genres:
    for file in os.listdir(data_dir + genre + "/raw_audio/"):
        file_ = os.fsdecode(file)
        if file_.endswith(".mp3"):
            filename = os.path.basename(file_)    # Remove path from file
            basename, ext = filename.split(".",1)      # Filename no extension

            mp3_raw_audio_path = raw_audio_dir.format(genre, filename)
            
            spectrograms_path = spectrograms_dir.format(genre)
            audio_slices_path = audio_slices_dir.format(genre, basename)

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
            print("#          CLEAN DATA         #") 
            print("###############################")
            print("")
            print("> Converting " + file_ + " to mono .wav file")
            
            audio_convert(mp3_raw_audio_path, "mp3", "wav", 1)
            wav_raw_audio_path = raw_audio_dir.format(genre, basename+".wav")
            # print("> Slicing wav file " + basename + ".wav into 30s segments")
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

                    audio_to_spectrogram(audio_slice_path, spectrograms_path)
            # print("\n")
"""