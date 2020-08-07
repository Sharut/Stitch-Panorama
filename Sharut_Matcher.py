import cv2
import numpy as np 
import os

Match_dict={}

class matchers:

		def detectAndDescribe(self, image):
		# convert the image to grayscale
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# check to see if we are using OpenCV 3.X
				# detect and extract features from the image
			orb = cv2.ORB_create(nfeatures=20000)
			kp = orb.detect(image,None)
			(kps, features) = orb.compute(image, kp)

			kps = np.float32([kp.pt for kp in kps])

			# return a tuple of keypoints and features
			return (kps, features)


		def match(self, im1,im2, direction=None):
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
				if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
					matches.append((m[0].trainIdx, m[0].queryIdx))

			print("matches selected: ",len(matches))
			# computing a homography requires at least 4 matches
			if len(matches) > 4:
				# construct the two sets of points
				ptsA = np.float32([kpsA[i] for (_, i) in matches])
				ptsB = np.float32([kpsB[i] for (i, _) in matches])

				# compute the homography between the two sets of points
				(H, mask) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
					5.0)

				# return the matches along with the homograpy matrix
				# and status of each matched point
				return H, len(matches)

			# otherwise, no homograpy could be computed
			return None


for file1 in os.listdir('./Photos/1'):
	if file1=='.DS_Store':
		continue
	max_match = 0
	smallDict = {}
	for file2 in os.listdir('./Photos/1'):

		if file2=='.DS_Store':
			continue
		print(file1+", "+file2)
		if(file1==file2):
			continue

		name1 = file1.split('.')[0]
		name2 = file2.split('.')[0]
		print(file1)

		img1 = cv2.resize(cv2.imread('./Photos/1/'+file1),(480, 320))
		img2 = cv2.resize(cv2.imread('./Photos/1/'+file2), (480, 320))

		my_match= matchers();
		A, b = my_match.match(img1,img2,None)
		smallDict[name2]=b

	Match_dict[name1]=(smallDict)

for key,val in Match_dict.items():
	print(key,sorted(val))

min_match = 1000000

for i,j in Match_dict.items():
	for k,l in j.items():
		if(l<min_match):
			start=i
			end = k
			min_match=l

print(start,end,min_match)







