import cv2
import numpy as np
import os
import random

ROOT_DIR = os.path.abspath("../")
RAW_PATH = os.path.join(ROOT_DIR, "../data/raw/")

image_dir = os.path.join(RAW_PATH,'Football2')
target_dir = os.path.join( RAW_PATH, 'For_Photoshop/Football/val/')



file_names = next(os.walk(image_dir))[2]


first = True
avg = 0



for file_name in file_names:
   # file_name = random.choice(file_names)
    img_path = os.path.join(image_dir,file_name)

    if (file_name == ".DS_Store"):
            continue

    img = cv2.imread(img_path)

    if(first):
        first= False
        avg = np.float32(img)

    else:
        cv2.accumulateWeighted(img,avg,0.01)


res1 = cv2.convertScaleAbs(avg)


cv2.imwrite(target_dir+'avg_img.png',res1)
cv2.destroyAllWindows()
