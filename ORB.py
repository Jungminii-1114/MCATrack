import os
import cv2

USER_PATH = os.path.expanduser("~")
DATA_DIR = os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/MemLoTrack", "Anti-UAV410")
train_dir = os.path.join(DATA_DIR, "train")

sequences = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
first_seq = sequences[0]
print(f"First SEq : {first_seq}") # str

seq_path = os.path.join(train_dir, first_seq)
images = [f for f in os.listdir(seq_path) if f.endswith((".jpg", "jpeg", "png"))]

images.sort()
frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith((".jpg", ".jpeg", "png"))])

for frame_id in range(len(frame_files)):
    frame_file = frame_files[frame_id]
    frame_path = os.path.join(seq_path, frame_file)
    curr_frame = cv2.imread(frame_path)


image_path = os.path.join(seq_path, images[0])
sec_image_path = os.path.join(seq_path, images[1])


img = cv2.imread(image_path)
img2 = cv2.imread(sec_image_path)

orb = cv2.ORB_create(
    nfeatures=40000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K = 2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20
)

kps, desc = orb.detectAndCompute(img, None)
kps2, desc2 = orb.detectAndCompute(img2, None)

if desc is None or desc2 is None:
    raise ValueError("Descriptor is None")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc, desc2)
matches = sorted(matches, key=lambda x:x.distance)

matched = cv2.drawMatches(img, kps, img2, kps2, matches[:50], None, flags=2)
cv2.imwrite("orb_matched.jpg", matched)

