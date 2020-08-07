import cv2
import numpy as np 
import os

class matchers:

		def detectAndDescribe(self, image):
		# convert the image to grayscale
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# check to see if we are using OpenCV 3.X
				# detect and extract features from the image
			orb = cv2.ORB_create(nfeatures=5000)
			kp = orb.detect(image,None)
			(kps, features) = orb.compute(image, kp)

			kps = np.float32([kp.pt for kp in kps])

			# return a tuple of keypoints and features
			return (kps, features)


		def match(self, im1,im2, direction=None, method='homography'):
			# compute the raw matches and initialize the list of actual
			# matches
			print("Direction : ", direction)
			kpsA, featuresA = self.detectAndDescribe(im1)
			kpsB, featuresB = self.detectAndDescribe(im2)
			matcher = cv2.DescriptorMatcher_create("BruteForce")
			rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
			matches = []

			# loop over the raw matches
			for m in rawMatches:
				# ensure the distance is within a certain ratio of each
				# other (i.e. Lowe's ratio test)
				if len(m) == 2 and m[0].distance < m[1].distance * 0.65:
					matches.append((m[0].trainIdx, m[0].queryIdx))

			print("matches selected: ",len(matches))
			# computing a homography requires at least 4 matches
			if len(matches) > 4:
				# construct the two sets of points
				ptsA = np.float32([kpsA[i] for (_, i) in matches])
				ptsB = np.float32([kpsB[i] for (i, _) in matches])

				# compute the homography between the two sets of points

				if method == 'affine':
					partial_homo, ma = cv2.estimateAffine2D(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=5.0)
					H = np.array([[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 1]])
					H[0:partial_homo.shape[0],0:partial_homo.shape[1]] = partial_homo
					print(H)
					print(H.shape)
        			#M = np.append(M, [[0,0,1]], axis=0)

				if method == 'homography':
					H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,5.0)

				# return the matches along with the homograpy matrix
				# and status of each matched point
			return H

			# otherwise, no homograpy could be computed
			return None


# for file1 in os.listdir('/Users/sharutgupta/Dropbox/COL780/Ass2/Images/1'):
# 	if file1=='.DS_Store':
# 		continue
# 	max_match = 0
# 	for file2 in os.listdir('/Users/sharutgupta/Dropbox/COL780/Ass2/Images/1'):
# 		if file2=='.DS_Store':
# 			continue
# 		print(file1+", "+file2)
# 		if(file1==file2):
# 			continue

# 		img1 = cv2.resize(cv2.imread('./Images/1/'+file1),(480, 320))
# 		img2 = cv2.resize(cv2.imread('./Images/1/'+file2), (480, 320))

# 		count_img1 = file1.split('.')[0]
# 		count_img2 = file2.split('.')[0]
# 		my_match= matchers();
# 		my_match.match(img1,img2,None)

