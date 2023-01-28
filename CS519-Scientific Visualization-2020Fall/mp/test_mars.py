import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

mars_gray = mpimg.imread('/Users/zongfan/Downloads/mars.png')
Msrgb = np.array([[0.412391, 0.357584, 0.180481],
                  [0.212639, 0.715169, 0.072192],
                  [0.019331, 0.119195, 0.950532]])
mars_xyz = mars_gray.copy()
for i in range(mars_gray.shape[0]):
    for j in range(mars_gray.shape[1]):
        mars_xyz[i][j] = np.matmul(Msrgb, mars_gray[i][j])

mars_brightness = mars_xyz[:,:,1]

c_dict = {
           'red': [[0, 0, 0], 
                  [0.2, 0, 0.14],
                  [0.75, 1., 1.],
                  [1., 0.7, 0.7]],
            'green': [[0, 0, 0.14],
                      [0.24, 0.14, 0.24],
                      [0.3, 0.7, 0.7],
                      [0.4, 0.3, 0.],
                      [1., 0, 0]], 
            'blue': [[0, 1., 1.],
                     [0.28, 0, 0],
                     [1., 0, 0]]
         }
mars_colormap = LinearSegmentedColormap("mars_cmp", segmentdata=c_dict, N=256)
mars_red = mars_colormap(mars_brightness)
# plt.imshow(mars_cmp)
# plt.show()