import numpy as np
import cv2
import time
import sys

cap = cv2.VideoCapture(0)

#sample = cv2.imread('/tmp/card1/card1/positive/card10.png',0)
sample = cv2.imread('/tmp/crop.png',0)
orb_sample = cv2.ORB()
kp_sample, des_sample = orb_sample.detectAndCompute(sample, None)

def drawMatches(img1, kp1, img2, kp2, matches):

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    x = 0
    y = 0

    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

	x += kp1[img1_idx].pt[0]
        y += kp1[img1_idx].pt[1]

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    cv2.circle(out, (int(x/10),int(y/10)), 4, (255, 255, 0), 1)
    return out

while True:

	ret, frame = cap.read()

	if ret is not True:
		continue
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	try:
        	# Initiate STAR detector
		orb_obs = cv2.ORB()
	
        	# Find the keypoints with ORB
		kp_obs = orb_obs.detect(img, None)
	
        	# Compute the descriptors with ORB
		kp_obs, des_obs = orb_obs.detectAndCompute(img, None)

		# Create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# Match descriptors.
		matches = bf.match(des_sample, des_obs)
	except Exception, e:
		pass

	# Sort them in the order of their distance.
	try:
		matches = sorted(matches, key = lambda x:x.distance)
		print matches[1].distance
	except Exception, e:
		pass

	# Draw first 10 matches.
	try:
		result = drawMatches(img, kp_obs, sample, kp_sample, matches[:10])
	        cv2.imshow(':: ORBPy (Quit: press q) ::', result)
	except Exception, e:
		pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(1)
