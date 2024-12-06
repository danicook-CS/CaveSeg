'''
Saves images with new color palette for better visualization
'''
import cv2
import numpy as np
import os
import sys


classes=('plain', 'first layer', 'caveline', 'second layer', 'open area', 'attachment rock', 'arrow', 'reel', 'cookie', 'diver','stalactite','stalagmite','column')
palette=[[128, 0, 0],[0, 128, 0],[128, 128, 0],[0, 0, 128],[128, 0, 128],[0, 128, 128],[128, 128, 128],[64, 0, 0],[192, 0, 0],[64, 128, 0],[192,128,0],[64,0,128],[192,0,128]]
new_palette = [[64,64,64], [0, 128, 0], [192, 192, 0], [0, 128, 128], [192,192,192], [128,64,0], [128,0,0], [255,192,0], [192,0,0], [128,0,128],[0,0,192],[64,0,128],[0,64,128]]


def changeColor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pix = img[i, j, :]

            # Quantize the color channels
            pix = np.array([
                0 if pix[0] <= 32 else 64 if pix[0] <= 92 else 128 if pix[0] <= 160 else 192 if pix[0] <= 224 else 255,
                0 if pix[1] <= 32 else 64 if pix[1] <= 92 else 128 if pix[1] <= 160 else 192 if pix[1] <= 224 else 255,
                0 if pix[2] <= 32 else 64 if pix[2] <= 92 else 128 if pix[2] <= 160 else 192 if pix[2] <= 224 else 255
            ])

            # Replace with new palette color if matched
            for k in range(len(palette)):
                if np.array_equal(pix, palette[k]):
                    pix = new_palette[k]
                    break

            img[i, j, :] = pix

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


path = '/blue/mdjahiduislam/adnanabdullah/mmsegmentation/data/Test/Test_Images/outputs/output_caveseg'

if path.endswith('.jpg') or path.endswith('.png'):
    files = os.path.basename(path)
    folder = os.path.dirname(path)
else:
    files = os.listdir(path)
    folder = path

if not os.path.exists(folder+'_changed/'):
   os.makedirs(folder+'_changed/')

for filename in files:
    label = cv2.imread(folder+'/'+filename)
    label = cv2.resize(label, (480,270), interpolation = cv2.INTER_AREA)
    label = changeColor(label)
    cv2.imwrite(folder+'_changed/'+filename, label)

print('images saved with new color palette')
