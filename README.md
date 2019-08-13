Project Implementation:

Method Used: k-nearest neighbors
Hyperparameters: k: 1, patchsize: 4

Data Processing:

Loaded all the images one by one into the environment and converted them to
numpy arrays. This is done for both low resolution and high resolution images.
Each image array is now split into blocks of patchsize x patchsize. And then
each block is mapped to the corresponding 2*patchsize x 2*patchsize block in the
high resolution image array. Based on the location there will be 16000 blocks
of each patch. A k-nearest neighbor classifier is fit to each of the each 16000
blocks. For example, if the patch size is 4, we would have 256 blocks and hence
256 arrays of 16000 blocks. For a given test image array, based on the location
the train block at the same location which is nearest is found out. The
corresponding high resolution block for the train block is chosen as the high
resolution patch for the testblock. This is done for all blocks in a test image
and for all test images.

Inspiration:

As x-ray images, unlike general images, don't tend to vary a lot, a lot of
deep neural networks we tried didn't work out. This made us think simple and
exploit the dataset properties and thus we came up with k-nearest neighbors
classifier.
