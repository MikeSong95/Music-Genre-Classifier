# Music-Genre-Classifier
## Dependencies
- [librosa](https://librosa.github.io/librosa/)
- [ffmpeg](https://www.ffmpeg.org/)
- [pylab](https://scipy.github.io/old-wiki/pages/PyLab)
- [numpy](https://numpy.org/)
- [pydub](https://github.com/jiaaro/pydub)
## Raw Input Data
Place your .mp3 audio files in the corresponding classification data folder, and name them `<genre>_<file number>.mp3`. For example, `/data/rap/raw_audio/rap_1.mp3`. 
## Data Processing
Uncomment the code in `audio_to_spectrogram.py` and run `python audio_to_spectrogram.py`. This will create the spectrograms needed to train the model.
## Training
Run `python train.py`.
## Running
You will need to create a folder called `song` and place the mp3 file you want to identify there. Re-comment the code in `audio_to_spectrogram.py` and run `python main.py`.
## TODO
- Create `init` file for project directory structure.
