import os
import cv2
import torch
import math
import json
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align


# Load with default pre-trained weights (ImageNet)
backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

# To use only as a feature extractor (freeze weights)
for param in backbone.parameters():
    param.requires_grad = False

# # Optionally remove the top classification layer
# backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))


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
# def crop_resize(img, center_y, center_x, crop_size, output_size, avg_color):
#     """
#     img : 원본 프레임 (H, W, C)
#     center_y, center_x : Taret의 중심 좌표 
#     crop_sze : 잘라낼 정사각형의 실제 픽셀 길이 (s_z or s_x)
#     output_size : 모델에 넣을 최종 크기 (Template : 127, SearchREgion  :255)
#     avg_color : 화면 밖을 벗어났을 때 채울 배경색
#     """
#     img_h, img_w = img.shape[:2]
#     half = (crop_size -1) / 2
#     ymin = int(np.round(center_y - half))
#     ymax = int(np.round(center_y + half))
#     xmin = int(np.round(center_x - half))
#     xmax = int(np.round(center_x + half))

#     pad_top = max(0, -ymin)
#     pad_bottom = max(0, ymax - img_h + 1)
#     pad_left = max(0, -xmin)
#     pad_right = max(0, xmax - img_w + 1)

#     # 실제 원본 이미지에서 자를 수 있는 유효 좌표
#     valid_ymin = max(0, ymin)
#     valid_ymax = min(img_h - 1, ymax)
#     valid_xmin = max(0, xmin)
#     valid_xmax = min(img_w -1, xmax)

#     cropped_img = img[valid_ymin:valid_ymax + 1, valid_xmin:valid_xmax+1, :]
#     if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
#         canvas_h = pad_top + cropped_img.shape[0] + pad_bottom
#         canvas_w = pad_left + cropped_img.shape[1] + pad_right
#         canvas = np.full((canvas_h, canvas_w, img.shape[2]), avg_color, dtype=img.dtype)

#         target_h = (pad_top + cropped_img.shape[0]) - pad_top
#         target_w = (pad_left + cropped_img.shape[1]) - pad_left

#         if cropped_img.shape[0] != target_h or cropped_img.shape[1] != target_w:
#             cropped_img = cv2.resize(cropped_img, (target_w, target_h))

#         canvas[pad_top : pad_top + cropped_img.shape[0], 
#                 pad_left : pad_left + cropped_img.shape[1], :] = cropped_img
        
#         cropped_img = canvas
#     if cropped_img.shape[0] != output_size or cropped_img.shape[1] != output_size:
#         cropped_img = cv2.resize(cropped_img, (output_size, output_size))
        
#     return cropped_img


def crop_resize(img, center_y, center_x, crop_size, output_size, avg_color):
    # 1. 자를 영역 계산
    x = int(center_x - crop_size / 2)
    y = int(center_y - crop_size / 2)
    w = int(crop_size)
    h = int(crop_size)

    img_h, img_w, _ = img.shape
    valid_xmin = max(0, x)
    valid_ymin = max(0, y)
    valid_xmax = min(img_w, x + w)
    valid_ymax = min(img_h, y + h)

    pad_top = max(0, -y)
    pad_left = max(0, -x)
    
    if valid_ymax > valid_ymin and valid_xmax > valid_xmin:
        cropped_img = img[valid_ymin:valid_ymax, valid_xmin:valid_xmax, :]
    else:
        cropped_img = np.empty((0, 0, 3), dtype=img.dtype)

    canvas_h = int(h)
    canvas_w = int(w)
    canvas = np.full((canvas_h, canvas_w, 3), avg_color, dtype=img.dtype)

    if cropped_img.size > 0:
        copy_h = min(cropped_img.shape[0], canvas_h - pad_top)
        copy_w = min(cropped_img.shape[1], canvas_w - pad_left)
        
        if copy_h > 0 and copy_w > 0:
            patch = cv2.resize(cropped_img, (copy_w, copy_h))
            canvas[pad_top : pad_top + copy_h, pad_left : pad_left + copy_w, :] = patch

    final_img = cv2.resize(canvas, (output_size, output_size))
    return final_img

'''
Z_0 : [init_temp_256]      (initial Frame)
x_t : [c_t_f]              (Current Search Region (GRAY))
y_t : [current_memory]     (Current Memory Frame - Motion Map)
X_t : [search_region]      (Search Region - 2 Channel)
F_X : [search_feat_final]  (Search Region Feature Map) - X_t's After AdjustLayer
Z_t : [z_t_final]          (Dynamic Frame)
F_Z : [z_t_feat_final]     (Dynamic Frame Feature Map) - Z_t's After AdjustLayer


'''
class Magno_Motion(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        #self.device = device  # 이거 삭제하기 
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

        if descriptors1 is not None and descriptors2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches=sorted(bf.match(descriptors1, descriptors2), key=lambda x : x.distance)
            good_matches = matches[:50]

            if len(good_matches) >= 10:
                 # Why reshape?
                 # Ans : 
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
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
            
        return np.float32([[1, 0, 0], [0, 1, 0]]) # Plan C : 움직임 없음
    
    def warp(self, frame, M):
        # Numpy ndarray -> Torch tensor
        M_tensor = torch.tensor(M, dtype=torch.float32, device=frame.device)
        #print(f"M_tensor : {M_tensor}")
        M_tensor = M_tensor.unsqueeze(0) 
        #print(f"M tensor after unsqueeze(0) : {M_tensor.shape}")

        _, _, H, W = frame.size()
        M_tensor[:, 0, 2] = M_tensor[:, 0, 2] / (W / 2) # X축 이동량
        M_tensor[:, 1, 2] = M_tensor[:, 1, 2] / (H / 2) # Y축 이동량

        grid = F.affine_grid(M_tensor, frame.size(), align_corners=False)        
        #output = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=False)
        output = F.grid_sample(frame, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return output

    def forward(self, x_curr, x_prev, memory_prev):
        """
        x_curr: 현재 프레임 (X_t)
        x_prev: 이전 프레임 (X_{t-1})
        memory_prev: 이전 시점까지 누적된 모션 맵 (Y_{t-1})
        """

        prev_np = (x_prev.squeeze().cpu().numpy() * 255).astype(np.uint8)
        curr_np = (x_curr.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # prev_np = (x_prev.squeeze().to(self.device).numpy() * 255).astype(np.uint8)
        # curr_np = (x_curr.squeeze().to(self.device).numpy() * 255).astype(np.uint8)
        
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
        
        # 과거 메모리가 너무 강하게 누적되어 수식 변경
        #y_curr = self.alpha * current_motion + (1-self.alpha) * memory_prev_aligned


        # Feature Enhancement
        enhanced_feature = torch.cat([x_curr, y_curr], dim=1)
        print("current_motion min/max/mean:", current_motion.min().item(), current_motion.max().item(), current_motion.mean().item())
        print("y_curr min/max/mean:", y_curr.min().item(), y_curr.max().item(), y_curr.mean().item())


        # Return Value : [1. 2-channel data for Backbone] [2. y_curr for next frame calculation]
        return enhanced_feature, y_curr

class AdjustLayer(nn.Module):
    # 백본에 넣기 위해 
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

        


# 이제 이 roi_features [num_boxes, 256, 7, 7] 들을 DTCG 모듈에 넣어서 z_r과 비교하면 됨

    
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
        B, C, H, W = z.shape                                # [1, 256, 16, 16]
        N = H*W                                             # N : 256
        Q = self.conv_q(z).view(B, C, N).permute(0, 2, 1)   # [1, 256, 256]  B N C
        K = self.conv_k(d).view(B, C, N)                    # [1, 256, 256]  B C N
        V = self.conv_v(d).view(B, C, N).permute(0, 2, 1)   # [1, 256, 256]  B N C

        attn = torch.matmul(Q, K) / math.sqrt(C)           # [1, N, C] x [1, C, N]  -> [1, 256, 256] : B N N
        attn = F.softmax(attn, dim=-1)
        F_out = torch.matmul(attn, V)  # (1 N N) x (1 N C) -> B, N, C (1, 256, 256)

        # F_out의 모양을 다시 [B, C, H, W]로 되돌리기 
        # permute 로 순서 뒤집기 -> view로 N을 H. W 로 펼치기 
        
        #F_out = F_out.permute(0, 2, 1).view(1, 256, 16, 16) # [1, 256, 16, 16]
        F_out = F_out.permute(0, 2, 1).view(B, C, H, W)

        # z_r : [out_channels, in_channels / groups, H, W]
        # --> [1, 256, 16, 16]이 아닌, [256, 1, 16, 16]
        z_r = self.gamma * F_out + z

        z_r_kernel = z_r.view(C, 1, H, W)
        
        # Depth-wise Cross Correlation 
        # groups=C : Depth-wise 

        # response_map : [1, 256, 16, 16]
        response_map = F.conv2d(x, z_r_kernel, groups=C)
        
        return response_map, z_r
    
class RPN(nn.Module):
    '''
    RPN의 생성 Anchor Box 공식
    Total Anchor = (Feature Map W) x (Feature Map H) x k
    - k : Anchor Number per pixel
        ㄴ 서로 다른 Scale과 Aspect Ratio를 가진 앵커 박스의 개수 : 보통 9개 사용 (3 scale x 3 ratio)

    출력 크기 : 입력 크기 - 커널크기 + 1 (Stride= 1, padding=0)
    '''

    def __init__(self, in_channels = 256, out_channels = 256):
        super().__init__()
        #self.feature_size = 17
        self.feature_size=26
        self.ratio = [0.5, 1, 2]
        #self.scale = [4, 8]
        self.scale = [8]
        self.k = len(self.scale) * len(self.ratio)

        # register_buffer -> model.to(device) 할 때 디바이스 따라감
        self.register_buffer('anchors', self._generate_anchors())

        #self.anchor_boxes = np.zeros(((self.feature_size * self.feature_size * 9), 4))
        self.intermediate_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.reg_layer = nn.Conv2d(out_channels, self.k * 4, kernel_size=1, stride=1, padding=0)
        self.cls_layer = nn.Conv2d(out_channels, self.k * 2, kernel_size=1, stride=1, padding=0) # For classification whether 0 or 1

    def _generate_anchors(self):
        size = self.feature_size 
        stride=8
        base_size=8

        # cx, cy Grid 생성하기: stride 곱해서 원본 픽셀 스케일의 중심좌표 생성
        shifts=(torch.arange(0, size, dtype=torch.float32) + 0.5) * stride

        shift_y, shift_x = torch.meshgrid(shifts, shifts, indexing="ij")

        shift_x = shift_x.reshape(-1) 
        shift_y = shift_y.reshape(-1)

        centers = torch.stack((shift_x, shift_y), dim=1)

        w_list = []
        h_list = []
        for scale in self.scale:
            for ratio in self.ratio:
                w = base_size * scale * math.sqrt(ratio)
                h = base_size * scale * math.sqrt(1. /ratio)
                w_list.append(w)
                h_list.append(h)

        whs = torch.tensor([w_list, h_list], dtype=torch.float32).T

        centers = centers.unsqueeze(1).expand(size * size, self.k, 2).reshape(-1, 2)
        whs = whs.unsqueeze(0).expand(size * size, self.k, 2).reshape(-1, 2)

        anchors = torch.cat((centers, whs), dim=1)

        return anchors

    def forward(self, x):
        x = self.intermediate_layer(x)
        pred_anchor_locs = self.reg_layer(x)
        pred_class_score = self.cls_layer(x)

        return pred_anchor_locs, pred_class_score


class EFCG(nn.Module):
    '''
    Procedure
    1. Response_map을 RPN에 입력하여 Proposal 생성 (z_r은 안넣음)
    2. RPN이 찾은 수많은 후보 영역들 각각에 대하여 Feature Map을 추출하기  -> x_i \in R ^ kxkxc
    3. EFCG 모듈은 DTCG에서 만든 z_r을 가져옴. 
    3-2. z_r과 x_i를 1:1로 꼼꼼하게 비교함 -> Depth-wise Conv가 아니라 아다마르 곱 사용하기
                -> 아다마르 곱으로 동일한 위치의 값들끼리 직접 곱하는 방식으로 정밀한 상관관계 특징을 계산함
    
    4. RCNN을 통해 Classification과 Regression 수행
        - Classification : 이 후보 상자 안의 객체가 진짜인지 아닌지 판별하여 Confidence Score 산출 (BCE Loss 사용)
        - Regression : 후보 상자의 테두리를 타겟의 크기와 위치에 완벽히 들어맞ㄱ 미세 조절 (Smooth L1 Loss)
    
    5. RCNN의 검증을 마쳐, 모든 후보 상자에 점수가 생기면, 그 중 가장 높은거 선택
    '''
    def __init__(self, in_chan = 256, out_chan = 256):
        super().__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.RPN = RPN(self.in_channels, self.out_channels)

        self.rcnn = RCNN(in_channels=256)
        

        # 1. RPN에 Response_map 넣어서 Proposal 생성

    def forward(self, res_map, z_r, search_feature):
        anchor_loc, class_score = self.RPN.forward(res_map)
        anchor_loc = anchor_loc.permute(0, 2, 3, 1)
        anchor_loc = anchor_loc.reshape(-1, 4)    # [2028, 4]

        class_score = class_score.permute(0, 2, 3, 1) 
        class_score = class_score.reshape(-1, 2)  # [2028, 2]

        #print(f"💡Anchor_loc : shape [After reshaping] : {anchor_loc.shape}")     # [2028, 4]
        #print(f"💡class_score : shape [After reshaping] : {class_score.shape}")   # [2028, 2]

        '''
        1차 : 점수가 높은 상자들을 추려내기 
           ㄴ 추려진 상자의 위치 정보 (anchor locs)
        2차 : Anchor locs를 가지고 백본에서 추출된 넓은 Search region 위로 -> ROI Align, 7x7로 잘라내기
        3차 : RPN이 알려준 좌표를 바탕으로 특징 맵에서 잘려 나온 7x7xc 크기의 텐서가 x_i
        4차 : 각 후보마다 7x7로 규격화된 특징이 준비 됨. -> EFCG가 z_r이랑 x_i를 아다마르 곱     
        '''

        
        drone_scores = class_score[:, 1]
        max_val, idx = drone_scores.max(dim=0)
        max_score_pos = int(idx.item())
        #confidence_score = float(max_val.item())
        
        scores = F.softmax(class_score, dim=1)[:, 1]
        confidence_score = float(scores[max_score_pos].item())


        #print(f"‼️Best confidence score in [class_score] is {max_val}")
        #print(f"‼️Best confidence index in [class_score] is {idx}") 

        #print(f"⭐️Best confidence score  pos : {max_score_pos}")
        #print(f"⭐️Confidence score at msp : {confidence_score}")

        dx, dy, dw, dh= anchor_loc[max_score_pos]

        cx_a, cy_a, w_a, h_a = self.RPN.anchors[max_score_pos]

        # FAST R-CNN 공식
        cx_pred = (dx * w_a) + cx_a
        cy_pred = (dy * h_a) + cy_a
        w_pred = torch.exp(dw) * w_a
        h_pred = torch.exp(dh) * h_a

        x1_pred = cx_pred - (w_pred / 2.0)
        y1_pred = cy_pred - (h_pred / 2.0)
        x2_pred = cx_pred + (w_pred / 2.0)
        y2_pred = cy_pred + (h_pred / 2.0)

        # [device Error Occured] - Solution : dx.device로 이미 존재하는 텐서가 속한 디바이스를 따라가도록 설계
        pred_pos = torch.tensor([[0.0, x1_pred, y1_pred, x2_pred, y2_pred]], device=dx.device)
        
        #print("🤖")
        #print(f"pred_pos shape : {pred_pos.shape}") # [1, 5]
        #breakpoint()

        self.x_i_roi = roi_align(
            input = search_feature,
            boxes=pred_pos,
            output_size=(7,7),
            spatial_scale=32.0/255.0,
            aligned=True
        )

        # Hadamard Product
        #print(f"🤖🤖🤖x_i shape : {self.x_i_roi.shape}  |  z_r shape : {z_r.shape}🤖🤖🤖")
        #breakpoint()

        Fused_feature = self.x_i_roi * z_r
        #print(f"🤖🤖🤖 Fused Feature Shape : {Fused_feature.shape}") # [1, 256, 7, 7]
        #breakpoint()
        '''
        1. Max Score Position (Anchor Box)를 가지고 pred_anchor_locs에서 해당 좌표로 박스 정보 4개 뽑기
        2. serach_feat_final로 가서 roi_align으로 7x7 잘라서 따로 저장 -> 이게 바로 x_i
        3. 
        '''

        final_cls_score, final_reg_score = self.rcnn(Fused_feature)
        #print("🙌 R-CNN 통과 완료!")
        #print(f"최종 점수 : {final_cls_score}")
        #print(f"최종 델타 : {final_reg_score}")

        print("max_score_pos:", max_score_pos)
        print("anchor:", self.RPN.anchors[max_score_pos])
        print("delta:", anchor_loc[max_score_pos]) # 비정상 나와버렸음... [수정 필요]
        print("decoded xyxy:", x1_pred.item(), y1_pred.item(), x2_pred.item(), y2_pred.item())


        return final_cls_score, final_reg_score, pred_pos, confidence_score
    
    def train_forward(self, res_map):
        anchor_loc, class_score = self.RPN(res_map)
        
        anchor_loc = anchor_loc.permute(0, 2, 3, 1).reshape(-1, 4)    # [2028, 4]
        class_score = class_score.permute(0, 2, 3, 1).reshape(-1, 2)  # [2028, 2]
    
        base_anchors = self.RPN.anchors # [2028, 4]
        
        return class_score, anchor_loc, base_anchors
    
class RCNN(nn.Module):
#     4. RCNN을 통해 Classification과 Regression 수행
#         - Classification : 이 후보 상자 안의 객체가 진짜인지 아닌지 판별하여 Confidence Score 산출 (BCE Loss 사용)
#         - Regression : 후보 상자의 테두리를 타겟의 크기와 위치에 완벽히 들어맞ㄱ 미세 조절 (Smooth L1 Loss)
    def __init__(self, in_channels=256):
        super().__init__()
        self.custom_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cls_head = nn.Sequential(
            nn.Linear(in_channels, 128),
            #nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        self.reg_head = nn.Sequential(
            nn.Linear(in_channels, 128),
            #nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.custom_conv(x)
        x = self.pool(x)          # [1, 256, 1, 1]
        x = x.view(x.size(0), -1) # [1, 256]
        cls_score = self.cls_head(x)
        reg_score = self.reg_head(x)

        return cls_score, reg_score

    
class MCATracker(nn.Module):#
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.backbone = MCA_Backbone().to(self.device)
    
        self.neck = AdjustLayer(in_channels=2048, out_channels=256).to(self.device)
        self.magno_motion = Magno_Motion(alpha=0.8).to(self.device)
        self.DTCG = Dynamic_Target_Cross_Guidance(in_channels=256, out_channels=256).to(self.device)
        self.EFCG = EFCG(in_chan = 256, out_chan = 256).to(self.device)

        self.template_feat = None # Z_0 : 이건 time이 흘러도 변하지 않는, 고정적인 Initial Frame (Z_0)
        self.prev_memory = None
        self.prev_frame = None
        
    def init(self, frame_1, bbox_1):
        # 첫 프레임인 Init 단계에서는 위 클래스들을 사용하기 어려움...
        #def crop_resize(img, center_y, center_x, crop_size, output_size, avg_color):
        """
        img : 원본 프레임 (H, W, C)
        center_y, center_x : Taret의 중심 좌표 
        crop_sze : 잘라낼 정사각형의 실제 픽셀 길이 (s_z or s_x)
        output_size : 모델에 넣을 최종 크기 (Template : 127, SearchREgion  :255)
        avg_color : 화면 밖을 벗어났을 때 채울 배경색
        """
        x, y, w, h = bbox_1
        cx = x + w / 2.0
        cy = y + h / 2.0

        context = 0.5
        p = (w+h) / 2.0 * context
        val = max(0.0, (w+p) * (h+p))
        s_z = float(np.sqrt(val))

        #bbox_1 기준으로 crop_resize 127로 하기 
        # z_crop : (127, 127, 3) -> (W, H, C)
        z_crop = crop_resize(frame_1, cy, cx, crop_size=s_z, output_size=127, avg_color=np.mean(frame_1, axis=(0, 1)))
        

        z_crop = cv2.resize(z_crop, (127, 127))  # 사이즈 오류 발생하여 해당 코드 생성
        
        
        if len(z_crop.shape) == 3:
            z_crop = cv2.cvtColor(z_crop, cv2.COLOR_BGR2GRAY)
            # (127, 127)
        
        #print(f"z_crop.shape : {z_crop.shape}")
        #breakpoint()
        
        # 채널 추가 -> (1, 127, 127)
        z_tensor = torch.Tensor(z_crop)
        z_tensor = z_tensor.unsqueeze(0)
        #print(f"z_tensor.shape : {z_tensor.shape}")  # (1, 127, 127)
        #breakpoint()

        
        '''
        여기 중요!! 
        prev_frame이 만들어지는 과정 : 127x127 흑백 사진을 저장해놓음
        현재시점의 frame을 prev에 저장해놔서 다음 iter 돌 때 과거로 되도록

        memory_frame은 127x127 짜리 검정색으로 저장해놓기 
        -> 이것도 iter에 따라 업데이트 해줘야 함.
        '''
        # 2번 프레임(track)과의 크기(255x255)를 맞추기 위해 1번 프레임도 넓게 잘라서 저장
        sx = s_z * 2.0
        #search_crop_1 = crop_resize(frame_1, cy, cx, crop_size=sx, output_size=255, avg_color=np.mean(frame_1, axis=(0, 1)))
        search_crop_1 = crop_resize(frame_1, cy, cx, crop_size=s_z, output_size=255, avg_color=np.mean(frame_1, axis=(0, 1)))
        if len(search_crop_1.shape) == 3:
            search_crop_1 = cv2.cvtColor(search_crop_1, cv2.COLOR_BGR2GRAY)
            
        search_tensor_1 = torch.Tensor(search_crop_1).unsqueeze(0).unsqueeze(0) # [1, 1, 255, 255]
        search_tensor_1 = search_tensor_1.to(torch.float) / 255.0
        search_tensor_1 = search_tensor_1.to(self.device)

        self.prev_frame = search_tensor_1
        self.prev_memory = torch.zeros_like(search_tensor_1)


        #z_clone = z_tensor.clone().detach()
        #z_final = torch.cat((z_clone, z_tensor), dim=0)
        # 만약 같은 프레임을 뒤에 concat 해버리면 너무 선명해짐 -> 배경까지 움직이는 잔상으로 착각
        z_final = torch.cat((z_tensor, torch.zeros_like(z_tensor)), dim=0)
        #print(f"z_final shape : {z_final.shape}")  # [2, 127, 127]
        #breakpoint()

        z_final = z_final.unsqueeze(0)
        #print(f"Added Batch Size : {z_final.shape}")   #[1, 2, 127, 127]
        #breakpoint()

        #z_final = z_final.astype('float32') / 255.0   # 습관적으로 Numpy 변환 주의하기
        z_final = z_final.to(torch.float) / 255.0
        z_final = z_final.to(self.device)
        #print(f"After Normalized : {z_final.shape}")
        #print(f"{z_final}")
        #breakpoint()

        self.template_feat = self.backbone(z_final) 
        #print(f"template_feat.shape : {self.template_feat.shape}") # [1, 2048, 16, 16]
        #print(f"template_feat.size : {self.template_feat.size}")
        #breakpoint()

        # 1. 127x127 이미지 내에서의 축소/확대 비율
        scale_ratio = 127.0 / s_z
        
        # 2. 127x127 이미지 내에서의 새로운 드론 크기
        w_rel = w * scale_ratio
        h_rel = h * scale_ratio
        
        # 3. 127x127 이미지 내에서의 새로운 중심점
        cx_rel = 127.0 / 2.0
        cy_rel = 127.0 / 2.0
        
        # 4. 최종 RoI Align용 바운딩 박스 좌표 (x1, y1, x2, y2)
        x1_rel = cx_rel - (w_rel / 2.0)
        y1_rel = cy_rel - (h_rel / 2.0)
        x2_rel = cx_rel + (w_rel / 2.0)
        y2_rel = cy_rel + (h_rel / 2.0)
        
        # 파이토치 규격 [Batch, x1, y1, x2, y2] 에 맞게 텐서로 포장
        z_box = torch.tensor([[0.0, x1_rel, y1_rel, x2_rel, y2_rel]], device=self.device)
        
        # 확인용 출력
        #print(f"변환된 RoI 좌표: {z_box}")
        '''
        z_final : initial template
        아마도 2048 --> 256 압축하는건 init에서 하면 안될듯...?
        사실 backbone 들어가는것도 아직 하면 안될것 같기도 한데..
        # 아닌가? 초기에는 Memory, Dynamic 없는게 당연하니까 backbone 이후까지의 초기 과정은 다 완료된걸수도?

        Search Region 이외에 전부 ROI Align 시켜버리기
        '''
        ## Adjust Layer를 통해 ResNet으로 나온 2048 Channel 을 256으로 압축하기 
        self.init_temp_256 =  self.neck(self.template_feat)   # [1, 256, 16, 16]
        #print(f"압축 이후 : {self.init_temp_256.shape}")
        #print(f"압축 이후 size : {self.init_temp_256.size}") 
        #breakpoint()


        '''
        Initial Frame을 ROI_Align 해버리고 dynamic이 clone하면 연산량이 주는거 아닌가?
        < 알아보기 >
        '''
        z_box = torch.tensor([[0, x1_rel, y1_rel, x2_rel, y2_rel]], device=self.device)
        self.z_0_roi = roi_align(
            input = self.init_temp_256,
            boxes = z_box,
            output_size=(7,7),
            #spatial_scale = 1.0 / 16.0,
            spatial_scale = 16.0 / 127.0,
            aligned=True
        )
        #print(f"🔥Initial Frame Shape After ROI_Aligned : {self.z_0_roi.shape}") # [1, 256, 7, 7]

        self.dynamic_feat_final = self.z_0_roi.clone()

        #print(f"🔥Dynamic Frame Shape After ROI_Aligned : {self.dynamic_feat_final.shape}")  # [1, 256, 7, 7]
        self.dynamic_roi = roi_align(
            input = self.dynamic_feat_final,
            boxes = z_box,
            output_size=(7,7),
            spatial_scale = 16.0 / 127.0,
            aligned=True
        )
        
        return self.z_0_roi

    def track(self, frame_t, bbox):
        '''
        [ 지금까지 진행 상황 정리 ]
        MCATracker.init() : Magno_motion initial Template까지 뽑아서 ResNet 통과 완료
        ( z_final : Frame_init   |    z(k * k * c) : ROI Feature of Init Temp -> 생성해보기)

        [여기서 해야할 것]
        Z_t : 동적 프레임 : 실시간ㅇ로 만들어야 함 (흑백 + Magno가 찾은 Motino Map) -> 백본행
        X_t : 현재 탐색 프레임 : 실시간으로 만들어야 함 (흑백 + Magno가 찾은 Motion Map) -> 백본행
         + init에서 만든 초기 프레임은 (흑백 + 0으로 채운 모션) -> 백본행
        '''
        
        # [Search Region Template 생성]
        #print(f"MCATrack에 들어온 frame의 shape : {frame_t.shape}")  # [512, 640, 3]   
        # 255x255로 만들어야 함. 
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        context = 0.5
        p = (w+h) / 2.0 * context
        val = max(0.0, (w+p) * (h+p))
        sz = float(np.sqrt(val))

        cropped_frame = crop_resize(frame_t, cy, cx, crop_size=sz, output_size=255, avg_color=np.mean(frame_t, axis=(0, 1)))
        if len(cropped_frame.shape) == 3:
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        cropped_frame = cv2.resize(cropped_frame, (255, 255))

        #print(f"cropped_frame.shape : {cropped_frame.shape}")  # [255, 255]
        #breakpoint()

        cropped_tensor = torch.Tensor(cropped_frame)
        cropped_tensor = cropped_tensor.unsqueeze(0)
        #print(f"cropped_tesnor.shape : {cropped_tensor.shape}")   # [1, 255, 255]
        #breakpoint()

        cropped_tensor_final = cropped_tensor.unsqueeze(0)
        c_t_f = cropped_tensor_final.to(torch.float) / 255.0
        c_t_f = c_t_f.to(self.device)
        #print(f"After Normalized : {c_t_f.shape}")  # [1, 1, 255, 255]
        #breakpoint()

        #print("\n\n\n\n\n\n\n자 !!! 이제 Magno-Motion Module 드갑니다 !!!!!!! ")
        
        '''
        실험 1 : Magno_motion Abalation 테스트
        search_region, current_memory = self.magno_motion.forward(x_curr=c_t_f, x_prev = self.prev_frame, memory_prev = self.prev_memory)
        에서 

        search_region = torch.cat([c_t_f, torch.zeros_like(c_t_f)], dim=1)
        current_memory = torch.zeors_like(c_t_f)

        motion branch를 잠깐 끄고 appearance만 넣어보자 
        -> 그래도 bbox 터지면 Magno_motion이 원인은 아님
        -> 갑자기 안정되면 Magno_Motion이 원인임
        '''
        #search_region, current_memory = self.magno_motion.forward(x_curr=c_t_f, x_prev = self.prev_frame, memory_prev = self.prev_memory)
        search_region = torch.cat([c_t_f, torch.zeros_like(c_t_f)], dim=1)
        current_memory = torch.zeros_like(c_t_f)

        #print(f"\n\nsearch region's shape : {search_region.shape}") # [1, 2, 255, 255]
        #print(f"current memory's shape : {current_memory.shape}") # [1, 1, 255, 255]
        #breakpoint()

        # 다음 턴(Next Frame)을 위해 '현재'를 '과거'로 덮어쓰기
        self.prev_frame = c_t_f
        self.prev_memory = current_memory

        search_feat = self.backbone(search_region)
        #print(f"search_region after backbone : {search_feat.shape}") # [1, 2048, 32, 32]
        self.search_feat_final = self.neck(search_feat)
        #print(f"Search Feat Final : {self.search_feat_final.shape}") # [1, 256, 32, 32]
        

        ## Z_t : 동적 프레임 만들기 (X_t와 동일 사이즈만 127)
        # 동적 프레임은 DTCG의 Confidence Score와 Displacement가 필요함.
        # Z_t -> F_Z(After Neck)
        

         # Cross Attentoin (DTCG)
         # init, dynamic, search region

        #self.dynamic_feat_final = self.init_temp_256.clone()
        # 14:00 수정 -> Dynamic frame을 항상 초기 프레임으로 초기화해버림 (안됨)
        #self.dynamic_feat_final = self.z_0_roi.clone()
        
        response_map, z_r = self.DTCG.forward(z = self.z_0_roi, d = self.dynamic_feat_final, x = self.search_feat_final)
        with torch.no_grad():
            beta = 0.9
            self.dynamic_feat_final = (beta * self.dynamic_feat_final + (1.0 - beta) * z_r.detach())

        # Response Map : [1, 256, 17 , 17]
        print(f"==== Respoinse Map의 크기 : {response_map.shape} ====")
        print(f"z_r shape : {z_r.shape}")
        print(f"search_feat_final shape : {self.search_feat_final.shape}")
        #breakpoint()


        final_cls_score, final_reg_score, pred_pos, confidence_score = self.EFCG.forward(
            response_map, z_r, self.search_feat_final
        )
        print("confidence_score:", confidence_score)
        print("pred_pos:", pred_pos)
        print("final_reg_score:", final_reg_score)
        print("final_cls_score:", final_cls_score)
        # RCNN의 최종 분류 confidence
        rcnn_probs = F.softmax(final_cls_score, dim=1)
        rcnn_confidence = float(rcnn_probs[0, 1].item())

        print("rcnn_confidence:", rcnn_confidence)  

        # pred_pos: [batch_idx, x1, y1, x2, y2] in search-crop coordinates
        # _, x1, y1, x2, y2 = pred_pos[0]

        # proposal_w = x2 - x1
        # proposal_h = y2 - y1
        # proposal_cx = x1 + proposal_w / 2.0
        # proposal_cy = y1 + proposal_h / 2.0

        # # RCNN refinement delta 적용
        # dx, dy, dw, dh = final_reg_score[0]

        # refined_cx = dx * proposal_w + proposal_cx
        # refined_cy = dy * proposal_h + proposal_cy
        # refined_w = torch.exp(dw) * proposal_w
        # refined_h = torch.exp(dh) * proposal_h

        # refined_x1 = refined_cx - refined_w / 2.0
        # refined_y1 = refined_cy - refined_h / 2.0
        '''
        실험 2 : RCNN Refinement 끄고 RPN Proposal만 원본 좌표로 복원하기
        _, x1, y1, x2, y2 = pred_pos[0]

        refined_x1 = x1
        refined_y1 = y1
        refined_w = x2 - x1
        refined_h = y2 - y1
        만 적용
        '''     
        _, x1, y1, x2, y2 = pred_pos[0]

        refined_x1 = x1
        refined_y1 = y1
        refined_w = x2 - x1
        refined_h = y2 - y1

        # search crop(255x255) -> original frame 좌표계 복원
        scale = sz / 255.0
        crop_x1 = cx - (sz / 2.0)
        crop_y1 = cy - (sz / 2.0)

        # pred_x = crop_x1 + float(refined_x1.item()) * scale
        # pred_y = crop_y1 + float(refined_y1.item()) * scale
        # pred_w = float(refined_w.item()) * scale
        # pred_h = float(refined_h.item()) * scale

        pred_x1_global = crop_x1 + float(refined_x1.item()) * scale
        pred_y1_global = crop_y1 + float(refined_y1.item()) * scale
        pred_w_global = float(refined_w.item()) * scale
        pred_h_global = float(refined_h.item()) * scale

        pred_cx_global = pred_x1_global + pred_w_global / 2.0
        pred_cy_global = pred_y1_global + pred_h_global / 2.0

        prev_cx = x + w / 2.0
        prev_cy = y + h / 2.0

        dx = pred_cx_global - prev_cx
        dy = pred_cy_global - prev_cy

        max_shift = 12.0
        dx = max(-max_shift, min(max_shift, dx))
        dy = max(-max_shift, min(max_shift, dy))

        candidate_cx = prev_cx + dx
        candidate_cy = prev_cy + dy

        momentum = 0.7
        smooth_cx = momentum * prev_cx + (1.0 - momentum) * candidate_cx
        smooth_cy = momentum * prev_cy + (1.0 - momentum) * candidate_cy

        pred_w = w
        pred_h = h

        pred_x = smooth_cx - pred_w / 2.0
        pred_y = smooth_cy - pred_h / 2.0


        # 이미지 경계 보정
        img_h, img_w = frame_t.shape[:2]
        pred_x = max(0.0, min(pred_x, img_w - 1.0))
        pred_y = max(0.0, min(pred_y, img_h - 1.0))
        pred_w = max(1.0, min(pred_w, img_w - pred_x))
        pred_h = max(1.0, min(pred_h, img_h - pred_y))

        pred_bbox = [pred_x, pred_y, pred_w, pred_h]

        #print("proposal_w/h:", proposal_w.item(), proposal_h.item())
        print("refined_w/h:", refined_w.item(), refined_h.item())

        print(f"pred_bbox before fallback: {pred_bbox}")
        # confidence가 너무 낮으면 이전 bbox 유지
        center_dist = math.sqrt((pred_cx_global - prev_cx) ** 2 + (pred_cy_global - prev_cy) ** 2)

        if confidence_score < 0.5 or rcnn_confidence < 0.3 or center_dist > 40.0:
            print("Fall Back to previous bbox", bbox)
            pred_bbox = bbox



        return pred_bbox



    def train_track(self, frame_t, bbox):
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        context = 0.5
        p = (w+h) / 2.0 * context
        sz = float(np.sqrt(max(0.0, (w+p) * (h+p))))

        cropped_frame = crop_resize(frame_t, cy, cx, crop_size=sz, output_size=255, avg_color=np.mean(frame_t, axis=(0, 1)))
        if len(cropped_frame.shape) == 3:
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        cropped_frame = cv2.resize(cropped_frame, (255, 255))

        c_t_f = torch.Tensor(cropped_frame).unsqueeze(0).unsqueeze(0).to(torch.float) / 255.0
        c_t_f = c_t_f.to(self.device)


        '''
        실험 1 : Magno_Motion Abalation 테스트
        motion branch를 잠깐 끄고 appearance만 넣어보자 
        -> 그래도 bbox 터지면 Magno_motion이 원인은 아님
        -> 갑자기 안정되면 Magno_Motion이 원인임
        '''
        #search_region, current_memory = self.magno_motion(c_t_f, self.prev_frame, self.prev_memory)
        search_region = torch.cat([c_t_f, torch.zeros_like(c_t_f)], dim=1)
        current_memory = torch.zeros_like(c_t_f)
        
        self.prev_frame = c_t_f
        self.prev_memory = current_memory

        search_feat = self.backbone(search_region)
        self.search_feat_final = self.neck(search_feat)
        
        #self.dynamic_feat_final = self.z_0_roi.clone()
        response_map, z_r = self.DTCG(self.z_0_roi, self.dynamic_feat_final, self.search_feat_final)

        with torch.no_grad():
            beta = 0.9
            self.dynamic_feat_final = (beta * self.dynamic_feat_final + (1.0 - beta) * z_r.detach())

        class_score, anchor_loc, base_anchors = self.EFCG.train_forward(response_map)
        
        return class_score, anchor_loc, base_anchors

        

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    #print(f"현재 사용중인 device : {device}")

    model = MCATracker(device = device)
#    print(model)

    # 테스트용 더미 데이터
    # frame = cv2.imread(os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/MCATrack", "000077.jpg"))
    # bbox = [100, 100, 50, 60]

    # init_frame = model.init(frame, bbox) # init 내부에서 prev_frame, prev_memory 만들어짐

    # print(f"init_frame은? : {init_frame.shape}")       # [1, 256, 16, 16]
    # print(f"prev_memory : {model.prev_memory.shape}")  # [1, 127, 127]
    # print(f"prev_frame : {model.prev_frame.shape}")    # [1, 127, 127]

    # print("\n\n이제 model.track을 수행합니다 . . . ")
    # search_region = model.track(frame, bbox)


if __name__ == "__main__":
    main()
