import cv2
import pandas as pd
import os
import fnmatch
from tqdm import tqdm

def create_csv(root_dir, number_of_frames = 4, frame_jump = 1, mode="Test"):
    number_of_train_folders = len(next(os.walk(root_dir))[1])
    # print(number_of_train_folders)
    number_of_train_folders = 36
    
    train_dataset = pd.DataFrame({'frames': [], 'label': []})
    for j in range(number_of_train_folders):
      path = os.path.join(mode, f'{mode}{str(j+1).zfill(3)}')
      if not os.path.exists(path):
            os.makedirs(path)
      lst = [f"{mode}{str(j+1).zfill(3)}/{str(i+1).zfill(3)}.tif" for i in 
              range(0, len(fnmatch.filter(os.listdir(os.path.join(root_dir, f"{mode}{str(j+1).zfill(3)}")), '*.tif')), frame_jump)]
      # items = [(lst[i:i+number_of_frames], lst[i+number_of_frames]) for i in range(len(lst)-(number_of_frames+1))]
      # x = pd.DataFrame(items, columns=["frames", "label"])
      # train_dataset = pd.concat([train_dataset, x], axis=0, ignore_index=True)
      for image_path in tqdm(lst):
        image = cv2.imread(os.path.join(root_dir, image_path))
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(os.path.join(mode, f"{image_path.split('.')[0].split('/')[0]}", 
                                 f"frame_{image_path.split('.')[0].split('/')[-1]}.jpg" ), image) 
    return train_dataset

if __name__ == '__main__':
    create_csv('D:/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset/UCSDped1/Test')