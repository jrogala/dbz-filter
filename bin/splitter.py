import cv2 as cv
import argparse
import os

parser = argparse.ArgumentParser(description='Splitter for mkv videos')
parser.add_argument('-i', type=str, required=True, help='The path of the video file')
parser.add_argument('-o', type=str, required=True, help='The path of the folder with pics')
parser.add_argument('-k', type=int, required=True, help='Number of pics wanted, if k=0 get all pics')

args = parser.parse_args()

cap = cv.VideoCapture(args.i)

if not os.path.exists(args.o):
    os.makedirs(args.o)

k = args.k
amount_of_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
if k:
    frame_to_skip = amount_of_frame // k
    for i in range(k):
        _, raw_img = cap.read()
        if not (raw_img is None):
            cv.imwrite(f"{args.o}/{i}.jpeg", raw_img)
            for _ in range(frame_to_skip):
                cap.read()
else:
    frame_to_skip = 0
    for i in range(amount_of_frame):
        _, raw_img = cap.read()
        if not (raw_img is None):
            cv.imwrite(f"{args.o}/{i}.jpeg", raw_img)
