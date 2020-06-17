import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mtcnn import MTCNN


def perform_face_alignment(is_training=True, show_frames=False, stop_early=False, save_output=False, output_ix=0):
    train_path = "data/train"
    val_path = "data/val"
    
    align_train_path = "data/aligned_train"
    align_val_path = "data/aligned_val"
    
    if not os.path.exists(align_train_path):
        os.makedirs(align_train_path)
    if not os.path.exists(align_val_path):
        os.makedirs(align_val_path)
    
    data_path = os.path.join(os.getcwd(), train_path) if is_training else os.path.join(os.getcwd(), val_path)
    
    detector = MTCNN()
    
    desiredLeftEye = (0.35, 0.35)
    desiredFaceWidth = 256
    desiredFaceHeight = 256
    
    count = 0
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        
        align_path = f"{align_train_path}/{folder}/" if is_training else f"{align_val_path}/{folder}/"
        if not os.path.exists(align_path):
            os.makedirs(align_path)
        
        for filename in os.listdir(folder_path):
            img = cv2.cvtColor(cv2.imread(f"{folder_path}/{filename}"), cv2.COLOR_BGR2RGB)
            
            if show_frames:
                plt.imshow(img)
                plt.show()
            
            result = detector.detect_faces(img)
            if len(result) == 1:
                keypoints = result[0]['keypoints']
                rightEyeCenter = keypoints['left_eye']
                leftEyeCenter = keypoints['right_eye']
                
                dY = rightEyeCenter[1] - leftEyeCenter[1]
                dX = rightEyeCenter[0] - leftEyeCenter[0]
                angle = np.degrees(np.arctan2(dY, dX)) - 180
                
                desiredRightEyeX = 1.0 - desiredLeftEye[0]
                
                dist = np.sqrt((dX ** 2) + (dY ** 2))
                desiredDist = (desiredRightEyeX - desiredLeftEye[0])
                desiredDist *= desiredFaceWidth
                scale = desiredDist / dist
                
                eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
                
                M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
                
                tX = desiredFaceWidth * 0.5
                tY = desiredFaceHeight * desiredLeftEye[1]
                M[0, 2] += (tX - eyesCenter[0])
                M[1, 2] += (tY - eyesCenter[1])
                
                (w, h) = (desiredFaceWidth, desiredFaceHeight)
                out_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
                
                cv2.imwrite(f"{align_path}/{filename}", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
                
                if show_frames:
                    plt.imshow(out_img)
                    plt.show()
                
                if save_output and count == output_ix:
                    cv2.imwrite(f"output/original_{folder}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 
                    cv2.imwrite(f"output/aligned_{folder}.png", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
                    
                    bounding_box = result[0]['box']
                    keypoints = result[0]['keypoints']
                    
                    cv_img = cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), (0,255,0), 3)
                    
                    cv_img = cv2.circle(img, (keypoints['left_eye']), 3, (0,255,0), 3)
                    cv_img = cv2.circle(img, (keypoints['right_eye']), 3, (0,255,0), 3)
                    cv_img = cv2.circle(img, (keypoints['nose']), 3, (0,255,0), 3)
                    cv_img = cv2.circle(img, (keypoints['mouth_left']), 3, (0,255,0), 3)
                    cv_img = cv2.circle(img, (keypoints['mouth_right']), 3, (0,255,0), 3)
                    
                    cv2.imwrite(f"output/face_bbox_{folder}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    if show_frames:
                        plt.imshow(img)
                        plt.show()
                    
                    save_output = False
                
            else:
                print(f'Unable to align, folder:{folder_path}, file:{filename}')
        
        count += 1
        if stop_early and count == 10:
            break
