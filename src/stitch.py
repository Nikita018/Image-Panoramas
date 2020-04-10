import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys




# Find matching keypoints in both images 
# Technique used : ratio of sum of squares of distances (SSD)
# input : 2 numpy arrays
# output : sum of squares of distance

def find_ssd(a,b):
  return np.sum(np.square(a-b))
  
  
  
# Function : keypoints_matcher
# Inputs   : Descriptors for the 2 images from SIFT
# Outputs  : The best matches keypoints in the 2 images
# Strategy : Find SSD between each descriptor and find SSD ratio to find the best matched points
# Function : keypoints_matcher
# Inputs   : Descriptors for the 2 images from SIFT
# Outputs  : The best matches keypoints in the 2 images
# Strategy : Find SSD between each descriptor and find SSD ratio to find the best matched points
def keypoints_matcher(descriptors_L,descriptors_R):
  SSD_ratios, match=[], []
  for i in range(len(descriptors_R)):
    box=[]
    for j in range(len(descriptors_L)):
      box=box+[find_ssd(descriptors_R[i],descriptors_L[j])]
    box=np.array(box)
    idx=box.argsort()[0:2]
    idx1,idx2=idx[0],idx[1]
    ratio=box[idx1]/box[idx2]
    SSD_ratios=SSD_ratios+[ratio]
    match=match+[idx1]
  print("length SSD ratios : ",len(SSD_ratios))
  print("length match : ",len(match))
  top_indices = np.array(SSD_ratios).argsort()[0:10]
  indices_filtered = []
  for x in top_indices:
    if(SSD_ratios[x]<0.2):
      indices_filtered = indices_filtered + [x]
  if(len(indices_filtered)<4):
    return None
  else:
    best_match=[]
    for i in indices_filtered:
      best_match=best_match+[[i,match[i]]]
    return(best_match)
  
  
  
  
  
# Function : homography_finder
# Inputs   : Best matched coordinates from keypoints_matcher
# Outputs  : Homography matrix
# Strategy : Finding the 4 best points and finding homography matrix using the formula
def homography_finder(coordinates, keypoints_L,keypoints_R):
  #find keypoint coordinates
  kp_left,kp_right = [],[]
  for i in range(10):
    kp_right = kp_right + [keypoints_R[coordinates[i][0]].pt]
    kp_left = kp_left + [keypoints_L[coordinates[i][1]].pt]

  #find best 4 points for homography
  R,L,R_final,L_final=[],[],[],[]
  R= kp_right
  L= kp_left
  for i in range(len(R)):
    if len(R_final)<4:
      if R[i] not in R_final and L[i] not in L_final:
        R_final.append(R[i])
        L_final.append(L[i])
  L=L_final
  R=R_final

  # Formula to compute Homography
  # Reference : https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
  PH=np.array([[-L[0][0],-L[0][1],-1,0,0,0,(L[0][0]*R[0][0]),(L[0][1]*R[0][0]),(R[0][0])], [0,0,0,-L[0][0],-L[0][1],-1,(L[0][0]*R[0][1]),(L[0][1]*R[0][1]),R[0][1]],[-L[1][0],-L[1][1],-1,0,0,0,(L[1][0]*R[1][0]),(L[1][1]*R[1][0]),R[1][0]],[0,0,0,-L[1][0],-L[1][1],-1,(L[1][0]*R[1][1]),(L[1][1]*R[1][1]),R[1][1]],
              [-L[2][0],-L[2][1],-1,0,0,0,(L[2][0]*R[2][0]),(L[2][1]*R[2][0]),R[2][0]],[0,0,0,-L[2][0],-L[2][1],-1,(L[2][0]*R[2][1]),(L[2][1]*R[2][1]),R[2][1]],[-L[3][0],-L[3][1],-1,0,0,0,(L[3][0]*R[3][0]),(L[3][1]*R[3][0]),R[3][0]],[0,0,0,-L[3][0],-L[3][1],-1,(L[3][0]*R[3][1]),(L[3][1]*R[3][1]),R[3][1]],[0,0,0,0,0,0,0,0,1]])
  Y=np.array([[0,0,0,0,0,0,0,0,1]])
  PH_inv = np.linalg.pinv(PH)
  H=np.matmul(np.linalg.pinv(PH),np.transpose(Y))
  Homograph=np.array([[H[0][0],H[1][0],H[2][0]],[H[3][0],H[4][0],H[5][0]],[H[6][0],H[7][0],H[8][0]]])
  H_=np.linalg.inv(Homograph)
  return H_, L, R
  
  
  
# Padding an image
def pad(img):
  ht, wd, cc= img.shape

  # create new image of desired size and color for padding
  ww = int(2*wd)
  hh = int(2*ht)
  color = (0)
  result = np.full((hh,ww,cc), color, dtype=np.uint8)

  # compute center offset
  xx = (ww - wd) // 2
  yy = (hh - ht) // 2

  # copy img image into center of result image
  result[yy:yy+ht, xx:xx+wd] = img

  return result

  
  
  
# Function : image_stitching
# Inputs   : 2 images : image1 (warped right image), image2
# Outputs  : Resultant Stiched Image
# Strategy : Make the dimensions of the 2 images(warped right image and left image) same by adding zeros and then find the maximum of each pizel and keep. Finally normalize the image.
def image_stitching(image1,image2):
  dim1,dim2,dim=image1.shape,image2.shape,[]
  dim=[max(dim1[0],dim2[0]),max(dim1[0],dim2[0]),3]
  
  if dim2[0]>dim1[0]:
    image1=np.vstack((image1,np.zeros([dim2[0]-dim1[0],dim1[1],3])))
  elif dim2[0]<dim1[0]:
    image2=np.vstack((image2,np.zeros([dim1[0]-dim2[0],dim2[1],3])))
  dim1,dim2=image1.shape,image2.shape
  if dim2[1]>dim1[1]:
    image1=np.hstack((image1,np.zeros([dim1[0],dim2[1]-dim1[1],3])))
  elif dim2[1]<dim1[1]:
    image2=np.hstack((image2,np.zeros([dim2[0],dim1[1]-dim2[1],3])))

  result=np.maximum(image1, image2)
  result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
  return(result)
  
# Function : create_panaroma
# Inputs   : 2 raw images : leftImg, leftImg
# Outputs  : Resultant stiched Image
# Strategy : Pass raw images through SIFT to find matched keypoints and descriptors in the two images. Find the 4 best matched points and compute homography to warp one image. 
#            Finally combine the warped right image and the raw left image
def create_panaroma(leftImg,rightImg):
  #sift
  print("Detecting Keypoints using SIFT....")
  sift = cv2.xfeatures2d.SIFT_create()

  keypoints_L, descriptors_L = sift.detectAndCompute(leftImg,None)
  keypoints_R, descriptors_R = sift.detectAndCompute(rightImg,None)

  #feature_matching
  print("Finding Homography....")
  coordinates=keypoints_matcher(descriptors_L,descriptors_R)

  if(coordinates == None):
   print("The images cannot be combined...Returning the Left Image back")
   return leftImg
  else:
   homography, L, R = homography_finder(coordinates, keypoints_L, keypoints_R)


   #Image Warping
   l,b,h=rightImg.shape
   warped_img = cv2.warpPerspective(rightImg,homography, (b,l))


   #stitch image
   panaroma=image_stitching(warped_img,leftImg)

   return panaroma



  
path = sys.argv[1]
print("path is : ", path)
files = os.listdir(path)
idx = 0
print("Left Image:",files[idx])
leftImg = cv2.imread(path + files[idx])[:,:,::-1]
leftImg = pad(leftImg)
for i in range(1,len(files)):
  rightImg = cv2.imread(path + files[i])[:,:,::-1]
  rightImg = pad(rightImg)
  leftImg = create_panaroma(leftImg,rightImg)


cv2.imwrite(path + "/panorama.jpg",cv2.cvtColor(leftImg, cv2.COLOR_RGB2BGR))

