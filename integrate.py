import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import pickle
import random

path = "data/"
x_data = []
t_data = []
files = os.listdir(path)
random.shuffle(files)

for i in tqdm(files):
    x_data.append(np.max(np.asarray(Image.open(path + i)).astype(np.float32), axis=2) / 255.0)
    t_data.append(ord(i[0]) - ord("a"))

x_data = np.array(x_data, dtype=np.float32).reshape(len(x_data), 1, 28, 28)
t_data = np.array(t_data, dtype=np.int32)
print(t_data)

with open("./alphabet.pkl", "wb") as f:
    pickle.dump((x_data, t_data), f)
# np.save("./alphabet.npy", x_data)
