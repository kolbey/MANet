import os
import glob
import numpy as np
from PIL import Image

path = 'DATASET/convert_vaihingen/train/masks_512/'

img_list = glob.glob(os.path.join(path, '*.png'))
img_list.sort()

max_id = 0
min_id = 6

for img in img_list:
    img = Image.open(img).convert('L')
    img_array = np.array(img)
    arr_id = img_array.max()
    arr_id_min = img_array.min()

    if arr_id > max_id:
        max_id = arr_id
    if arr_id_min < min_id:
        min_id = arr_id_min

print('max id : {}' .format(max_id))
print('min id : {}' .format(min_id))
