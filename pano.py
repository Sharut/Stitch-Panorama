import numpy as np
import cv2
import sys
from matchers import matchers
import time

class Stitch:
	def __init__(self, args):
		self.path = args
		fp = open(self.path, 'r')
		filenames = [each.rstrip('\r\n') for each in  fp.readlines()]
		print( filenames)
		# self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
		self.images = [cv2.imread(each) for each in filenames]
		self.count = len(self.images)
		self.left_list, self.right_list, self.center_im = [], [],None
		self.matcher_obj = matchers()
		self.prepare_lists()

	def warpTwoImages(self, img2, img1, H):
		'''warp img2 to img1 with homograph H'''
		h1,w1 = img1.shape[:2]
		h2,w2 = img2.shape[:2]
		pts1 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]], dtype= 'f').reshape(-1,1,2)
		pts2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]], dtype= 'f').reshape(-1,1,2)
		pts2_ = cv2.perspectiveTransform(pts2, H)
		pts = np.concatenate((pts1, pts2_), axis=0)
		[xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5,dtype='int')
		[xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5,dtype='int')
		t = [-xmin,-ymin]
		Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

		result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
		


		for i in range(h1):
			for j in range(w1):
				if(img1[i,j].all()!=0): # all 3 coordinates are non-0
					result[t[1]+i,t[0]+j]=img1[i,j]
		return result

	def prepare_lists(self):
		print( "Number of images : %d"%self.count)
		self.centerIdx = self.count/2 
		print( "Center index image : %d"%self.centerIdx)
		self.center_im = self.images[int(self.centerIdx)]
		for i in range(self.count):
			if(i<=self.centerIdx):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])
		print( "Image lists prepared")

	def leftshift(self):
		# self.left_list = reversed(self.left_list)
		b = self.left_list[-1]
		H = np.identity(3)
		prev=b
		for a in reversed(self.left_list[0:-1]):
			H = np.matmul(H,self.matcher_obj.match(a,prev,'left'))
			print( "Homography is : ", H)
			xh = np.linalg.inv(H)
			print( "Inverse Homography :", xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
			ds = ds/ds[-1]
			print( "final ds=>", ds)
			f1 = np.dot(xh, np.array([0,0,1]))
			f1 = f1/f1[-1]
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
			print( "image dsize =>", dsize)
			#tmp = cv2.warpPerspective(a, xh, dsize)
			# cv2.imshow("warped", tmp)
			# cv2.waitKey()
			print(b.shape)
			#tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
			#tmp[0:b.shape[0], 0:b.shape[1]] = b
			tmp = self.warpTwoImages(a,b,H)
			b = tmp
			prev = a

		self.leftImage = b

		
	def rightshift(self):
		a = self.leftImage
		prev = self.left_list[-1]
		H=np.identity(3)
		for b in self.right_list:
			H = np.matmul(H,self.matcher_obj.match(b, a, 'right'))
			print( "Homography :", H)
			txyz = np.dot(H, np.array([b.shape[1], b.shape[0], 1]))
			txyz = txyz/txyz[-1]
			dsize = (int(txyz[0])+a.shape[1], int(txyz[1])+a.shape[0])
			#tmp = cv2.warpPerspective(b, H, dsize)
			# cv2.imshow("tp", tmp)
			# cv2.waitKey()
			# tmp[:a.shape[0], :a.shape[1]]=a
			#tmp = self.mix_and_match(a, tmp)
			# print "tmp shape",tmp.shape
			# print "a shape=", a.shape
			tmp = self.warpTwoImages(b,a,H)
			prev=b
			a = tmp

		self.leftImage = a
		# self.showImage('left')



	def mix_and_match(self, leftImage, warpedImage):
		i1y, i1x = leftImage.shape[:2]
		i2y, i2x = warpedImage.shape[:2]
		print( leftImage[-1,-1])

		t = time.time()
		black_l = np.where(leftImage == np.array([0,0,0]))
		black_wi = np.where(warpedImage == np.array([0,0,0]))
		print( time.time() - t)
		print( black_l[-1])

		for i in range(0, i1x):
			for j in range(0, i1y):
				try:
					if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						# print( "BLACK")
						# instead of just putting it with black, 
						# take average of all nearby values and avg it.
						warpedImage[j,i] = [0, 0, 0]
					else:
						if(np.array_equal(warpedImage[j,i],[0,0,0])):
							# print "PIXEL"
							warpedImage[j,i] = leftImage[j,i]
						else:
							if not np.array_equal(leftImage[j,i], [0,0,0]):
								bw, gw, rw = warpedImage[j,i]
								bl,gl,rl = leftImage[j,i]
								# b = (bl+bw)/2
								# g = (gl+gw)/2
								# r = (rl+rw)/2
								warpedImage[j, i] = [bl,gl,rl]
				except:
					pass
		# cv2.imshow("waRPED mix", warpedImage)
		# cv2.waitKey()
		return warpedImage



	def trim_left(self):
		pass

	def showImage(self, string=None):
		if string == 'left':
			cv2.imwrite("./hey/leftImage.jpg", self.leftImage)
			# cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
		elif string == "right":
			cv2.imwrite("./hey/rightImage.jpg", self.rightImage)


if __name__ == '__main__':
	try:
		args = sys.argv[1]
	except:
		args = "./txtlists/files1.txt"
	finally:
		print( "Parameters : ", args)
	s = Stitch(args)
	s.leftshift()
	# s.showImage('left')
	s.rightshift()

	print(("done"))
	cv2.imwrite("oldcode_sharut.jpg", s.leftImage)
	print( "image written")
	cv2.destroyAllWindows()
	
