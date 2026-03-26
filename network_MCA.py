import os
import cv2
import torch
import math
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
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

images = sorted([f for f in os.listdir(seq_path) if f.endswith((".jpg", "jpeg", "png"))])

frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith((".jpg", ".jpeg", "png"))])


for frame_id in range(len(frame_files)):
    pass
'''
[Magno-Motion Module]
1. 광수용체 단계 
    - 영역 내 어두운 영역의 대비를 강화, 이후 처리를 위해 신호 균형 맞추기 

2. 수평세포 및 양극세포 (Horizontal & Bipolar Cells)
    - 수평세포는 시공간적 노이즐르 일차적으로 억제 
    - 양극세포는 처리된 신호의 Edge를 선명하게 다듬기

3. @@ 무축삭세포 (Amarcine Cells) 연산자 @@
    - 시간 및 공간적 변화를 증폭
    - 시간적 고역 통과 필터 뱅크 (Temporal high-pass Filter Bank) 역할 수행
    - Difference Equation 방식으로 구현하고 Z-transform을 활용한
        전달 함수를 통해 시간의 흐름에 따른 움직임 신호를 필터링하기

[ 작동 프로세스 ]
1. 인접 프레임 간 시공간적 정렬 (Inter-frame Registration)
    카메라 자체의 흔들림/이동으로 인한 가짜 움직임 (Motion Artifacts)를 제거하기 위한 단계
    - 인접한 프레임들을 공간적, 시간적 정렬하여 연속된 프레임 간에 동일한 배경/특징이 일관성을 갖도록 맞추기

    -> GMC 사용 X
    - ORB (Oriented FAST and Rotated BRIEF) 기반 특징점 매칭
        - 두 프레임에서 각각 특징을 뽑아 '비슷한 것 끼리 짝짓기'
        - 프레임 A, B에서 각각의 코너점(FAST)를 찾고, 이진 탐색(BRIEF)으로 매칭
        - RANSAC 돌려서 오매칭 걸러내고, homography 또는 Affine을 구하기 
        - 단점 : 텍스처가 밋밋한 배경에서는 실패 가능성 有
    
    - 위상 상관 기법 (Phase Correlation)
        - 특징점을 아예 뽑지 않고, 수학적인 주파스 변환을 이용해 이미지가 통쨰로 얼만큼 이동했는지 찾는 방법
        - 두 이미지를 고속 푸리에 변환 (FFT)하여 주파수 도메인으로 보내기 
        - 두 이미지의 위상 차이를 계산, 역 푸리에 변환, 이동한 픽셀만큼 떨어진 위치에 Peak가 생김
        - 이 Peak의 좌표가 카메라의 이동량
        장점 : cv2.phaseCorrelate 를 사용하면 속도가 미친듯이 빠름 (화면 내 작은 노이즈 무시하고 기가막히게 잘 찾음)
        단점 : 단순 상하좌우 Panning일 때는 완벽 / 크게 회전, 줌 아웃 줌 인 -> Log-Polar 변환과 같은 추가 처리 없으면 성능 저하 
    
    - ECC (Enhanced Correlation Coefficient) 극대화
        - 단순 픽셀 차이 상위 호환 버전 
        - 두 이미지의 픽셀값 상관관계를 수학적으로 극대화 하는 변환 행렬을 반복적으로 찾기 
        - cv2.findTransformECC
        - 빛 밝기가 변하는 상황에서 강력한 내성 
        - 반복 최적화 연산을 수행해서 가장 무거움. -> Downscaling 피라미드 이미지에 적용하는 방법이 필수적
'''
def crop_resize(img, center_y, center_x, crop_size, output_size, avg_color):
    """
    img : 원본 프레임 (H, W, C)
    center_y, center_x : Taret의 중심 좌표 
    crop_sze : 잘라낼 정사각형의 실제 픽셀 길이 (s_z or s_x)
    output_size : 모델에 넣을 최종 크기 (Template : 127, SearchREgion  :255)
    avg_color : 화면 밖을 벗어났을 때 채울 배경색
    """
    img_h, img_w = img.shape[:2]
    half = (crop_size -1) / 2
    ymin = int(np.round(center_y - half))
    ymax = int(np.round(center_y + half))
    xmin = int(np.round(center_x - half))
    xmax = int(np.round(center_x + half))

    pad_top = max(0, -ymin)
    pad_bottom = max(0, ymax - img_h + 1)
    pad_left = max(0, -xmin)
    pad_right = max(0, xmax - img_w + 1)

    # 실제 원본 이미지에서 자를 수 있는 유효 좌표
    valid_ymin = max(0, ymin)
    valid_ymax = min(img_h - 1, ymax)
    valid_xmin = max(0, xmin)
    valid_xmax = min(img_w -1, xmax)

    cropped_img = img[valid_ymin:valid_ymax + 1, valid_xmin:valid_xmax+1, :]
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        canvas = np.empty((int(crop_size), int(crop_size), img.shape[2]), dtype=img.dtype)
        canvas[:, :] = avg_color

        canvas[pad_top : pad_top + cropped_img.shape[0], 
               pad_left : pad_left + cropped_img.shape[1], :] = cropped_img
        
        cropped_img = canvas
    if crop_size != output_size:
        cropped_img = cv2.resize(cropped_img, (output_size, output_size))
        
    return cropped_img



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

        _, _, H, W = frame.size()
        M_tensor[:, 0, 2] = M_tensor[:, 0, 2] / (W / 2) # X축 이동량
        M_tensor[:, 1, 2] = M_tensor[:, 1, 2] / (H / 2) # Y축 이동량

        grid = F.affine_grid(M_tensor, frame.size(), align_corners=False)        
        output = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=False)

        return output

    def forward(self, x_curr, x_prev, memory_prev):
        """
        x_curr: 현재 프레임 (X_t)
        x_prev: 이전 프레임 (X_{t-1})
        memory_prev: 이전 시점까지 누적된 모션 맵 (Y_{t-1})
        """

        prev_np = (x_prev.squeeze().cpu().numpy() * 255).astype(np.uint8)
        curr_np = (x_curr.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        M = self.get_affine_matrix(prev_np, curr_np)
        
        '''
        현재 프레임 & 과거 프레임 이용 -> Affine Matrix
        이전 정렬된 프레임 : memory_prev, M
        '''
        x_prev_aligned = self.warp(x_prev, M)
        memory_prev_aligned = self.warp(memory_prev, M)
        current_motion = torch.abs(x_curr - x_prev_aligned)
        
        # 2-2. 수식 적용: 새로운 모션 맵(Y_t) 생성 = α * 현재움직임 + α * 과거기억
        # New Motion Map (Y_t) = ⍺ * curr_motion + ⍺ * (prev memory)
        y_curr = self.alpha * current_motion + self.alpha * memory_prev_aligned

        # Feature Enhancement
        enhanced_feature = torch.cat([x_curr, y_curr], dim=1)

        # Return Value : [1. 2-channel data for Backbone] [2. y_curr for next frame calculation]
        return enhanced_feature, y_curr

class AdjustLayer(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()

        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        x = self.downsampling(x)
        return x
    
class MCA_Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        #self.backbone = backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT, replace_stride_with_dilation=[False, True, True])
        for param in self.backbone.parameters():
            param.requires_grad=False 
        
        old_conv = self.backbone.conv1
        old_weight = old_conv.weight # [64, 3, 7, 7] 필터 수, 채널, H, W
        new_conv = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            compressed_weight = torch.mean(old_weight, dim=1, keepdim=True) #[64, 1, 7, 7] : RGB 가중치 평균
            new_conv.weight[:, 0:1, :, :] = compressed_weight # 채널0
            new_conv.weight[:, 1:2, :, :] = compressed_weight # 채널 1

            # 기존 3채널 가중치 중에서 앞의 2개 채널(R, G) 지식만 복사해서 새 레이어에 씌우기 
            #new_conv.weight[:, :2, :, :] = old_conv.weight[:, :2, :, :]
        self.backbone.conv1 = new_conv
        #self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_features=2)

        # 추적(Tracking)을 위한 꼬리 자르기 -> nn.Linear로 하면 분류가 되어버림 (우리는 Cross-Correlation 해야해서 지도 형태 2차원 텐서맵 필요)
        # fc 레이어와 avgpool 레이어(맨 뒤 2개)를 아예 제거하여 '특징 맵(Feature Map)' 자체를 출력하게 만듭니다.
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

    def forward(self, x):
        x = self.backbone(x)

        return x
    
class Dynamic_Target_Cross_Guidance(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        '''
        초기 특징 z : Query 생성
        동적 특징 d  :K, V 생성

        융합 특징 F 계산 : Softmax(...)

        잔차 연결을 통한 특징 업데이트 
        F를 바탕으로 초기 타겟 특징을 업데이트 -> z_r (향상된 타겟 특징) 생성
        z_r = \gamma F + z (gamma : Learnable Scalar Parameter)

        z_r과 x를 Cross Correlation 연산하기 
        - Depth-wise Convolution 사용
        '''

        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z, d, x):
        #Q = self.conv_q.view(B, d, H*W).permute(0, 2, 1)
        B, C, H, W = z.shape
        N = H*W
        Q = self.conv_q(z).view(B, C, N).permte(0, 2, 1)
        K = self.conv_k(d).view(B, C, N)
        V = self.conv_v(d).view(B, C, N).permute(0, 2, 1)

        attn = torch.matmul(Q, K) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        F_out = torch.matmul(attn, V)

        z_r = self.gamma * F_out + z
        
        # Depth-wise Cross Correlation 
        # groups=C : Depth-wise 
        response_map = F.conv2d(x, z_r, groups=C)
        return response_map, z_r
        



    
class MCATracker:
    def __init__(self):
        self.backbone = MCA_Backbone().cuda()
        self.neck = AdjustLayer(in_channels=2048, out_channels=256).cuda()

        self.magno_motion = Magno_Motion(alpha=0.8).cuda()
        self.DTCG = Dynamic_Target_Cross_Guidance()
        
        self.crop_and_resize = crop_resize()

        self.template_feat = None # Z_0
        self.memory_prev = None
        self.prev_frame = None
        
    def init(self, frame_1, bbox_1):
        z_0_tensor = ...

        self.template_feat = self.backbone(z_0_tensor)
        feat_z0 = backbone(z0)
        feat_zt = backbone(z_t)
        feat_x = backbone(xt)

    def track(self, frame_t):
        # [현재 프레임 ] Search Region (X_t) Cropping + Motion Map 합치기 
        x_t_tensor = ...

        search_feat = self.backbone(x_t_tensor)

         # Cross Attentoin (DTCG)