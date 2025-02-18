import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import scipy
import scipy.ndimage

#image_folder = "Exercise1/data/toyProblem_F22/"
image_folder = "data/toyProblem_F22/"


arr = np.zeros((256, 256, 63))

dx = np.zeros((256, 255, 63))
dy = np.zeros((255, 256, 63))
dz = np.zeros((256, 256, 62))

dx_list = []
dy_list = []
dp_list = []

# Load all JPG images into the initilized array
for img_number in range(1,64):
  if(img_number < 10):
    img_path = image_folder + "frame_0" + str(img_number) + ".png"
  else:
    img_path = image_folder + "frame_" + str(img_number) + ".png"

  #convert image to gray and float representation.
  arr[:,:,img_number - 1] = cv.imread(img_path, cv.IMREAD_GRAYSCALE)/255
  dx_list += [arr[:,1:,img_number - 1]  - arr[:,0:-1,img_number - 1]]
  dy_list += [arr[1:,:,img_number - 1]  - arr[0:-1,:,img_number - 1]]
  dp_list += [(dx_list[-1][1:,:] + dy_list[-1][:,1:])/2]
  dx[:,:,img_number - 1] = arr[:,1:,img_number - 1]  - arr[:,0:-1,img_number - 1]
  dy[:,:,img_number - 1] = arr[1:,:,img_number - 1]  - arr[0:-1,:,img_number - 1]
  if(img_number < 63):
    dz[:,:,img_number - 1] = arr[:,:,img_number]  -  arr[:,:,img_number - 1]

print(dx.shape)

#temp = arr[:, 0:100, 0]
temp = dp_list[0]
plt.imshow(temp, cmap='gray')
plt.axis('off')  # Remove axes
plt.show()

#Create an animation for the images
fig, ax = plt.subplots()


ims = []

for img in range(arr.shape[2]):
    im = ax.imshow(arr[:,:,img], cmap='gray', animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
plt.show()


#Creates an animation for the gradient map of the images
fig, ax = plt.subplots()
ims_p = []

for img in range(len(dp_list)):
    im = ax.imshow(dp_list[img][:,:], cmap='gray', animated=True)
    ims_p.append([im])

ani = animation.ArtistAnimation(fig, ims_p, interval=50, blit=True, repeat_delay=1000)
plt.show()

#Problem 2.2

#Gauss filter over x and y axis where z-axis is the different images.
gauss_list = scipy.ndimage.gaussian_filter(arr, sigma = 1, radius=1, axes=(0,1))


#plt.imshow(ims_gauss[:,:,1], cmap='gray')
#plt.show()

#Creates an animation for the gradient map of the images
fig, ax = plt.subplots()
ims_gauss = []

for img in range(gauss_list.shape[2]):
    im = ax.imshow(gauss_list[:,:, img], cmap='gray', animated=True)
    ims_gauss.append([im])

ani = animation.ArtistAnimation(fig, ims_gauss, interval=50, blit=True, repeat_delay=1000)
plt.show()


#prewitt filter over x, y and z axis where z-axis is the different images.
prewitt_list_x = scipy.ndimage.prewitt(arr, axis=0)
prewitt_list_y = scipy.ndimage.prewitt(arr, axis=1)
prewitt_list_z = scipy.ndimage.prewitt(arr, axis=2)

fig, axs = plt.subplots(1, 3)
ims_prewitt = []

for img in range(prewitt_list_x.shape[2]):
    im1 = axs[0].imshow(prewitt_list_x[:,:, img], cmap='gray', animated=True)
    im2 = axs[1].imshow(prewitt_list_y[:,:, img], cmap='gray', animated=True)
    im3 = axs[2].imshow(prewitt_list_z[:,:, img], cmap='gray', animated=True)

    ims_prewitt.append([im1,im2,im3])


ani = animation.ArtistAnimation(fig, ims_prewitt, interval=50, blit=True, repeat_delay=1000)
plt.show()

#plots premade function vs estimates picture "id"
id = 10
fig, axs = plt.subplots(2, 3)
axs[0,0].imshow(prewitt_list_x[:,:, id], cmap='gray', animated=True)
axs[0,0].set_title("Prewitt x-axis")
axs[0,1].imshow(prewitt_list_y[:,:, id], cmap='gray', animated=True)
axs[0,1].set_title("Prewitt y-axis")
axs[0,2].imshow(prewitt_list_z[:,:, id], cmap='gray', animated=True)
axs[0,2].set_title("Prewitt z-axis")

axs[1,0].imshow(dx[:,:, id], cmap='gray', animated=True)
axs[1,0].set_title("Estimate x-axis")
axs[1,1].imshow(dy[:,:, id], cmap='gray', animated=True)
axs[1,1].set_title("Estimate y-axis")
axs[1,2].imshow(dz[:,:, id], cmap='gray', animated=True)
axs[1,2].set_title("Estimate z-axis")

plt.show()