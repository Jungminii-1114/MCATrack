import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


# Load with default pre-trained weights (ImageNet)
backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

# To use only as a feature extractor (freeze weights)
for param in backbone.parameters():
    param.requires_grad = False

# # Optionally remove the top classification layer
# backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

USER_PATH = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/MemLoTrack", "Anti-UAV410")
train_dir = os.path.join(DATA_DIR, "train")

sequences = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
first_seq = sequences[0] # str type

seq_path = os.path.join(train_dir, first_seq)

frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith((".jpg", ".jpeg", "png"))])

def ORB_and_Extract(seq_path, img1, img2):
    orb = cv2.ORB_create()

    img1 = cv2.imread(os.path.join(seq_path, img1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(seq_path, img2), cv2.IMREAD_GRAYSCALE)

    # Keypoints : 특징점들의 리스트 [좌표, 지름, 각도, 응답, 옥타브, 클래스 ID]
    # Descriptors : 각 특징점을 설명하기 위한 2차원 배열로 표현 : 두 특징점이 같은지 판단할 때 사용됨.
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptor is None")
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(descriptors1, descriptors2), key=lambda x : x.distance)

    good_matches = matches[:50]

    if len(good_matches) < 3:
        print(f"Affine 변환 행렬을 위한 Good Matches 수 부족 : {len(good_matches)}")
        return None

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #print(keypoints1[1])
    #print(f"matches : {matches}")

    M, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3 # 최대 허용 에러 3
    )
    residual = None

    if M is not None:
        h_img, w_img = img1.shape
        warped_img1 = cv2.warpAffine(img1, M, (w_img, h_img))
        residual = cv2.absdiff(warped_img1, img2)

        _, residual = cv2.threshold(residual, 20, 255, cv2.THRESH_BINARY)
        print(f"Residual Shape : {residual.shape}")
        print(f"Residual : {residual}")
        print(f"Affine Model [Matrix] : {M}")

    return residual
    
residual_img = ORB_and_Extract(seq_path, frame_files[1], frame_files[2])

if residual_img is not None:
    cv2.imshow("residual img", residual_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No results")