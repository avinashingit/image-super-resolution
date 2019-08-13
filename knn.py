import numpy as np
from PIL import Image
import os
import pickle
from skimage.util import view_as_blocks
from sklearn.neighbors import KDTree

train_features_dir = "../GivenData/Drive/xray_images/train_images_64x64"
train_target_dir = "../GivenData/Drive/xray_images/train_images_128x128"
test_features_dir = "../GivenData/Drive/xray_images/test_images_64x64"
patch_size = 4
nn = 1

train_features = np.zeros((16000, 64, 64))
train_targets = np.zeros((16000, 128, 128))
test_features = np.zeros((16000, 64, 64))

train_target_images = os.listdir(train_target_dir)
for i, file in enumerate(train_target_images):
    img = Image.open(os.path.join(train_target_dir, file))
    img = img.convert("L")
    image = np.asarray(img)
    train_targets[i] = image

train_input_images = os.listdir(train_features_dir)
for i, file in enumerate(train_input_images):
    img = Image.open(os.path.join(train_features_dir, file))
    img = img.convert("L")
    image = np.asarray(img)
    train_features[i] = image

test_input_images = os.listdir(test_features_dir)
for i, file in enumerate(test_input_images):
    img = Image.open(os.path.join(test_features_dir, file))
    img = img.convert("L")
    image = np.asarray(img)
    test_features[i] = image

images = np.array(train_features)
feature_blocks = view_as_blocks(images, block_shape=(1, patch_size,
                                                     patch_size)).squeeze()
target_blocks = view_as_blocks(np.asarray(train_targets), block_shape=(
    1, patch_size*2, patch_size*2)).squeeze()

test_feature_blocks = view_as_blocks(
    test_features, block_shape=(1, patch_size, patch_size)).squeeze()

number_of_blocks = int((64*64)/(patch_size*patch_size))
train_blocks = np.zeros((number_of_blocks, 16000, patch_size*patch_size))
traint_blocks = np.zeros((number_of_blocks, 16000, 4*patch_size*patch_size))

for i in range(feature_blocks.shape[0]):
    for j in range(feature_blocks.shape[1]):
        for k in range(feature_blocks.shape[2]):
            n = feature_blocks.shape[1]*j+k
            train_blocks[n][i] = feature_blocks[i][j][k].reshape(patch_size*patch_size)
            traint_blocks[n][i] = target_blocks[i][j][k].reshape(4*patch_size*patch_size)

number_of_blocks = int((64*64)/(patch_size*patch_size))
test_blocks = np.zeros((number_of_blocks, 3999, patch_size*patch_size))
for i in range(test_feature_blocks.shape[0]):
    for j in range(test_feature_blocks.shape[1]):
        for k in range(test_feature_blocks.shape[2]):
            n = test_feature_blocks.shape[1]*j+k
            test_blocks[n][i] = test_feature_blocks[i][j][k].reshape(patch_size*patch_size)

np.random.seed(1405)

kdtrees = dict()

for i in range(train_blocks.shape[0]):
    kdtrees[i] = KDTree(train_blocks[i])

nearest = dict()
for i in range(test_blocks.shape[0]):
    _, nearest[i] = kdtrees[i].query(test_blocks[i], k=nn)
with open("nearest_" + str(patch_size) + "_" + "_" + str(nn) + ".pkl", "wb") as f:
    pickle.dump(nearest, f)

output_dir = "/home/avinash/UIUC/CS446/Project/KNN/Output_images_" + \
    str(patch_size) + "_" + str(nn) + "_1405"
os.makedirs(output_dir)
for i, file in enumerate(test_input_images):
    temp1 = []
    temp2 = []
    for j in range(number_of_blocks):
        if j % (number_of_blocks**0.5) == 0 and j != 0:
            temp1.append(temp2)
            temp2 = []
        b = np.mean(traint_blocks[j][nearest[j][i]], axis=0).reshape(2*patch_size, 2*patch_size)
        temp2.append(b)
    temp1.append(temp2)
    img = np.array(temp1)
    img = img.transpose(0, 2, 1, 3).reshape(-1, img.shape[1]*img.shape[3])
    image = Image.fromarray(img).convert("L")
    image.save(os.path.join(output_dir, file), "PNG")
