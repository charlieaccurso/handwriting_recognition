import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans

digits= datasets.load_digits()
# print(digits)

# DESCRIPTION OF DATASET
# print(digits.DESCR)
# size of each image: 8x8 pixels
# dataset is from: UCI

# LOOKING AT THE DATA
# print(digits.data)
# each row represents an 8x8 pixel image of a number
# each index in the row holds the color value of a pixel (64 pixels per sample)
# color values range from 0 (white) to 16 (black)

# LOOKING AT THE TARGETS
# print(digits.target)
# digits.target contains the number each image represents

# VISUALIZE THE IMAGES USING MATPLOTLIB
plt.gray()
plt.matshow(digits.images[100]) # image at index 100
plt.show() # looks like a four (4)
plt.clf()
# check if the label matches the image:
print(digits.target[100]) # => 4

# VISUALIZING MORE THAN ONE IMAGE AT A TIME
# initialize figure (width, height)
fig= plt.figure(figsize=(6,6))
# adjust subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for loop to visualize the first 64 images (of over 1700 samples in dataset)
for i in range(64):
  # initialize subplot of 8x8 at the i+1-th pos.
  ax= fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
  # display the image at index i of digits.images
  ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
  # label image with target value
  ax.text(0, 7, str(digits.target[i]))
plt.show()
plt.clf()

# BUILDING THE MODEL
k= 10 # for ten digits 0-9
model= KMeans(n_clusters=k, random_state=42)
# FIT THE MODEL
model.fit(digits.data)

# VISUALIZING THE CENTROIDS AFTER K-MEANS
fig= plt.figure(figsize=(8,3))
fig.suptitle('Cluster Center (Centroid) Images', fontsize=14, fontweight='bold')

for i in range(10):
  # initialize subplots in a 2x5 grid, at the i+1th pos.
  ax= fig.add_subplot(2, 5, i+1)
  # display images
  ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)
plt.show()
plt.clf()

fig= plt.figure(figsize=(8,3))
indices= [5, 1, 9, 4]

for i in range(4):
  ax= fig.add_subplot(1, 4, i+1)
  ax.imshow(model.cluster_centers_[indices[i]].reshape((8,8)), cmap=plt.cm.binary)
plt.show()
plt.clf()

# TEST A NEW SAMPLE - 2059
new_samples= np.array(
[
[0.00,2.21,7.62,6.64,5.95,7.62,3.36,0.00,0.00,0.76,6.40,1.98,1.14,7.63,3.82,0.00,0.00,0.00,0.00,0.00,4.34,7.62,2.20,0.00,0.00,0.00,0.00,2.44,7.62,4.65,0.00,0.00,0.00,0.00,0.30,6.71,6.41,0.15,0.00,0.00,0.00,0.08,4.81,7.55,1.98,0.00,0.00,0.00,1.91,4.81,7.62,6.10,2.29,1.68,2.29,2.13,7.62,7.62,7.62,7.62,7.62,7.62,7.62,7.62],
[0.00,0.84,6.87,7.47,7.40,7.09,0.00,0.00,0.00,3.51,7.62,2.67,0.23,3.13,1.53,0.00,0.00,4.50,7.62,1.60,0.00,6.71,5.41,0.00,0.00,4.58,7.62,2.67,0.00,5.42,6.10,0.00,0.00,4.42,7.62,3.97,0.00,5.34,6.10,0.00,0.00,3.12,7.62,6.48,0.00,1.37,1.68,0.00,0.00,1.29,7.63,7.62,3.97,4.58,2.98,0.00,0.00,0.00,4.65,7.63,7.62,7.62,4.04,0.00],
[0.00,0.00,3.89,7.62,7.62,7.62,7.62,7.62,0.00,0.00,6.33,6.25,1.52,1.52,3.43,7.40,0.00,0.00,6.79,5.11,0.00,0.00,0.00,0.53,0.00,0.00,5.64,6.41,1.22,0.00,0.00,0.00,0.00,0.00,5.34,7.62,7.62,2.52,0.00,0.00,0.00,0.00,0.15,2.90,6.94,7.32,0.46,0.00,0.08,2.75,0.30,1.22,7.02,7.09,0.08,0.00,1.14,7.55,7.63,7.62,7.55,2.67,0.00,0.00],
[0.00,0.00,3.05,7.62,7.40,7.17,0.00,0.00,0.00,0.00,5.72,6.78,3.12,7.62,0.76,0.00,0.00,0.30,7.55,4.49,3.89,7.62,0.76,0.00,0.00,0.38,7.24,7.62,7.62,7.62,0.99,0.00,0.00,0.00,1.45,3.05,4.42,7.62,2.29,0.00,0.00,0.00,0.00,0.00,3.96,7.62,1.06,0.00,0.00,0.23,4.88,3.20,7.39,5.41,0.00,0.00,0.00,0.23,6.02,7.62,7.40,1.52,0.00,0.00]
]
)

new_labels= model.predict(new_samples)

# MAP LABELS TO DIGITS
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(3, end='')
  elif new_labels[i] == 1:
    print(0, end='')
  elif new_labels[i] == 2:
    print(8, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(9, end='')
  elif new_labels[i] == 5:
    print(2, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(7, end='')
  elif new_labels[i] == 8:
    print(6, end='')
  elif new_labels[i] == 9:
    print(5, end='')

# 2 requires most of the bottom of the matrix to be filled
# 0 requires most of the left side of the matrix to be filled
# 5 requires most of the upper right corner of the matrix to be filled
# 9 must have an open space in the bottom left section of the matrix
