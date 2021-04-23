# get_ipython().getoutput("curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz")
# get_ipython().getoutput("curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz")
# get_ipython().getoutput("tar -xf images.tar.gz")
# get_ipython().getoutput("tar -xf annotations.tar.gz")


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


input_dir = "raw_images/"
target_dir = "segmented_images"


input_img_paths = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ])
target_img_paths = sorted([
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
])

print("Number of samples:", len(input_img_paths), len(target_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)



x_train = []
y_train = []


for image in tqdm(input_img_paths):
    img = Image.open(image).convert("L").resize((128,128))
    img = np.array(img)
    img = img / 255.0
    x_train.append(img)


x_train = np.array(x_train)
x_train = np.expand_dims(x_train, axis=-1)
print(x_train.shape)


for image in tqdm(target_img_paths):
    img = Image.open(image).resize((128,128))
    img = np.array(img)
    y_train.append(img)


y_train = np.array(y_train)
y_train = np.expand_dims(y_train, axis=-1)
print(y_train.shape)


np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)



