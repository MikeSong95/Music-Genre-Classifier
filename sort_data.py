import os.path
import os

# generate random integer values
from random import seed
from random import randint

# seed random number generator
seed(1)

from shutil import copyfile

genres = ["pop","rap", "classical"]
data_dir = "./data/"

for genre in genres:
    train_count = 0
    test_count = 0
    val_count = 0
    total_count = 0
    for file in os.listdir(data_dir + genre + "/spectrograms/"):
        file_ = os.fsdecode(file)
        if file_.endswith(".png"):
            value = randint(1, 100)
            if value < 71:
                directory = data_dir + "train/" + genre
                train_count += 1
                total_count += 1
            elif value < 86:
                directory =  data_dir + "test/" + genre
                test_count += 1
                total_count += 1
            else:
                directory = data_dir + "val/" + genre
                val_count += 1
                total_count += 1
            copyfile(data_dir + genre + "/spectrograms/"+os.path.basename(file_), directory + "/" + os.path.basename(file_))
    # sanity check - data is distributed 70%/15%/15% in train/test/val 
    print("total data: %d" % total_count)
    print("train data: %d = %.2f percent" % (train_count, (train_count / total_count) * 100))
    print("test data: %d = %.2f percent" % (test_count, (test_count / total_count) * 100))
    print("val data: %d = %.2f percent" % (val_count, (val_count / total_count) * 100))