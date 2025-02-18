import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import scipy
import scipy.ndimage


#image_folder = "Exercise1/data/toyProblem_F22/"

image_folder = "data/toyProblem_F22/"


arr = np.zeros((256, 256, 63))

# Load all JPG images into the initilized array
for img_number in range(1,64):
    if(img_number < 10):
        img_path = image_folder + "frame_0" + str(img_number) + ".png"
    else:
        img_path = image_folder + "frame_" + str(img_number) + ".png"

    #convert image to gray and float representation.
    arr[:,:,img_number - 1] = cv.imread(img_path, cv.IMREAD_GRAYSCALE)/255

#prewitt filter over x, y and z axis where z-axis is the different images.
prewitt_list_x = scipy.ndimage.prewitt(arr, axis=0)
prewitt_list_y = scipy.ndimage.prewitt(arr, axis=1)
prewitt_list_z = scipy.ndimage.prewitt(arr, axis=2)


fig, ax = plt.subplots()
ax.imshow(arr[:,:,0], cmap='gray')
# animation function.  This is called sequentially
def animate(t):
    count = 0
    coords = np.zeros((289,3),dtype=np.int32)

    ax.clear()
    ax.imshow(arr[:,:,t], cmap='gray')
    u = []
    radius = 2
    for i in range(4,arr.shape[0] - 5,15):
        for j in range(4,arr.shape[1] - 5,15):
            coords[count] = [i, j, t]
            coord = coords[count]
            count += 1
            Vx = prewitt_list_x[coord[0]-radius-1:coord[0]+radius, coord[1]-radius-1:coord[1]+radius, coord[2]].flatten()
            Vy = prewitt_list_y[coord[0]-radius-1:coord[0]+radius, coord[1]-radius-1:coord[1]+radius, coord[2]].flatten()

            A = np.array([Vx,Vy]).T
            b = -prewitt_list_z[coord[0]-radius-1:coord[0]+radius, coord[1]-radius-1:coord[1]+radius, coord[2]].flatten()
            u += [np.linalg.lstsq(A,b)[0]]
            
    us = np.array(u)
    plt.quiver(coords[:,1],coords[:,0], us[:,1], us[:,0], color=['r'], scale=100)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=arr.shape[2], interval=100)

plt.show()



'''
fig, ax = plt.subplots()
ax.imshow(clean_imgs[:,:,0], cmap='gray')
# animation function.  This is called sequentially
def animate(t):
    count = 0
    coords = np.zeros((576,3),dtype=np.int32)

    ax.clear()
    ax.imshow(clean_imgs[:,:,t], cmap='gray')
    u = []
    radius = 2
    for i in range(4,arr.shape[0] - (2*radius + 1),20):
        for j in range(4,arr.shape[1] - (2*radius + 1),20):
            coords[count] = [i, j, t]
            coord = coords[count]
            count += 1
            Vx = prewitt_list_x[int(coord[0])-radius-1:int(coord[0])+radius, int(coord[1])-radius-1:int(coord[1])+radius, int(coord[2])].flatten()
            Vy = prewitt_list_y[coord[0]-radius-1:coord[0]+radius, coord[1]-radius-1:coord[1]+radius, coord[2]].flatten()

            A = np.array([Vx,Vy]).T
            b = -prewitt_list_z[coord[0]-radius-1:coord[0]+radius, coord[1]-radius-1:coord[1]+radius, coord[2]].flatten()
            u += [np.linalg.lstsq(A,b)[0]]
            
    us = np.array(u)
    plt.quiver(coords[:,1],coords[:,0], us[:,1], us[:,0], color=['r'], scale=700)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=arr.shape[2], interval=200)

plt.show()
'''