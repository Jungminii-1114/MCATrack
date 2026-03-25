import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

USER_PATH = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/MemLoTrack", "Anti-UAV410")
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")


specific_seq_path = os.path.join(test_dir, "03_2499_0962-2461")
specific_f_files = sorted([f for f in os.listdir(specific_seq_path) if f.endswith((".jpg", ".jpeg", "png"))])

#prev_frame_path = os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/Tiny Drones", "000050.jpg")
#curr_frame_path = os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/Tiny Drones", "000055.jpg")

prev_frame_path = os.path.join(specific_seq_path, "000003.jpg")
curr_frame_path = os.path.join(specific_seq_path, "000006.jpg")


prev_frame = cv2.imread(prev_frame_path)
curr_frame = cv2.imread(curr_frame_path)

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

prev_gray_tensor = torch.from_numpy(prev_frame_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
curr_gray_tensor = torch.from_numpy(curr_frame_gray).float().unsqueeze(0).unsqueeze(0) / 255.0

# cv2.imshow("prev_frame", prev_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

class Magno_Motion(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
    
    def get_affine_matrix(self, prev_img, curr_img):
        # prev_gray_tensor -> [1, 1, 512, 640]
        if len(prev_img.shape) == 3:
            prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()

        # orb.detectAndCompute(img) -> img : 8bit GrayScale 이미지
        keypoints1, descriptors1 = orb.detectAndCompute(prev_img, None)
        keypoints2, descriptors2 = orb.detectAndCompute(curr_img, None)

        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Descriptor is None")
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches=sorted(bf.match(descriptors1, descriptors2), key=lambda x : x.distance)
        good_matches = matches[:50]

        if len(good_matches) >= 10:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # Why reshape?
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1 ,1, 2)

            M, _ = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3
            )
            if M is not None:
                return M
        prev_float = np.float32(prev_img)
        curr_float = np.float32(curr_img)

        # Plan B : Phase Correlation [Fourier domain에서 shift 추정]
        shift, response = cv2.phaseCorrelate(prev_float, curr_float)
        if response > 0.05:
            tx, ty = shift
            M = np.float32([
                [1, 0, tx],
                [0, 1, ty]
            ])
            return M
        
        return np.float32([[1, 0, 0], [0, 1, 0]]) # 움직임 없음 간주 (Plan C)
    
    def warp(self, frame, M):
        # Numpy ndarray -> Torch tensor
        M_tensor = torch.tensor(M, dtype=torch.float32, device=frame.device)
        print(f"M_tensor : {M_tensor}")
        M_tensor = M_tensor.unsqueeze(0) 
        print(f"M tensor after unsqueeze(0) : {M_tensor.shape}")

        #_, _, H, W = frame_tensor.size()
        _, _, H, W = frame.size()
        M_tensor[:, 0, 2] = M_tensor[:, 0, 2] / (W / 2) # X축 이동량
        M_tensor[:, 1, 2] = M_tensor[:, 1, 2] / (H / 2) # Y축 이동량

        grid = F.affine_grid(M_tensor, frame.size(), align_corners=False)        
        output = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=False)

        return output
    
    def forward(self, x_curr, x_prev, memory_prev):
        """
        x_curr: 현재 프레임 (t)
        x_prev: 이전 프레임 (t-1)
        memory_prev: 이전 시점까지 누적된 모션 맵 (Y_{t-1})
        """

        M = self.get_affine_matrix(x_prev, x_curr)
        x_prev_aligned = self.warp(x_prev, M)
        memory_prev_aligned = self.warp(memory_prev, M)


model = Magno_Motion()

M = model.get_affine_matrix(prev_frame_gray, curr_frame_gray)
print(f"추출된 변환 행렬 M:\n{M}")
prev_gray_aligned_tensor = model.warp(prev_gray_tensor, M) # Numpy 형태에서 M을 구했지만, Tensor 형태로 GPU에서 M으로 Warping

# (1, 1, H, W) -> (H, W) 로 축소 후 0~255 복원
aligned_img_numpy = prev_gray_aligned_tensor.squeeze().cpu().numpy() * 255.0
aligned_img_numpy = aligned_img_numpy.astype(np.uint8)

cv2.imshow("Current Frame (Target)", curr_frame_gray)
cv2.imshow("Warped Previous Frame", aligned_img_numpy)

residual = cv2.absdiff(curr_frame_gray, aligned_img_numpy)
cv2.imshow("Residual (Motion Extraction)", residual)

cv2.waitKey(0)
cv2.destroyAllWindows()
