import os
import cv2
import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if "train" not in os.listdir():
    os.system("cd ~/Pytorch/Exam2/")
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
    os.system("unzip train-Exam2.zip")

train_dir = [f for f in os.listdir('train') if f[-4:] == '.png']

resizehere = (400, 300)
imgs, txts, jsons =[], [], []
loading = 0
loading_end = len(train_dir)
for p in train_dir:
    imgs.append(cv2.resize(cv2.imread('train/' + p), resizehere))
    txts.append(open('train/' + p[:-4] + '.txt', 'r').read().splitlines())
    jsons.append(json.load(open('train/' + p[:-4] + '.json')))
    loading += 1
    if loading % (loading_end//10) == 0:
        print('Loading', 100*loading//loading_end, '% data.')
imgs = np.array(imgs)
txts = np.array(txts)
jsons = np.array(jsons)

# show images with different classes
labels = ["red blood cell", "difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"]
for label in labels:
    for i in range(len(txts)):
        if txts[i] == [label]:
            plt.title(str('In image # ' + str(i) + ' with label ' + str(label)))
            plt.imshow(imgs[i])
            plt.show()
            break
        if txts[i] == [labels[0], label]:
            plt.title(str('In image # ' + str(i) + ' with label ' + str(labels[0]) + ' + ' + str(label)))
            plt.imshow(imgs[i])
            plt.show()
            break

# One hot encoding multilabel target
le = MultiLabelBinarizer(classes=np.array(labels))

txts = le.fit_transform(txts)
# txts = txts.T[1:].T
print(imgs.shape, txts.shape, jsons.shape)
print(jsons[0][0]['bounding_box']['maximum'])
#
# dataset = []
# for x, y in zip(imgs, txts):
#     dataset.append([x, y])
# dataset = np.array(dataset)

#Splitting
# SEED = 42
# train, test = train_test_split(dataset, random_state=SEED, test_size=0.2)
np.save("imgs.npy", imgs)
np.save("txts.npy", txts)
# np.save("train.npy", train)
# np.save("test.npy", test)

#
# _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# blurred = cv2.GaussianBlur(img, (3,3), 0)
# x2 = thresh * img
# plt.imshow(x2, cmap='gray')
# plt.title('threshold')
# plt.show()



