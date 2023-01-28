import numpy as np
import matplotlib.pyplot as plt

# EX1 
# xlim = 50
# expand = 20
# x = np.linspace(-xlim, xlim, expand*xlim+1)
# print(len(x))
# x_inter = 2*xlim/(expand*xlim+1)
# print("x interval: ", x_inter)
# # f
# f = np.cos(2*x) * np.sin(x/4)
# # rect kernel
# rect = np.zeros_like(x)
# rect_half_w = 1/np.pi
# start_rect_idx = int((xlim-rect_half_w) // x_inter) 
# end_rect_idx = int(len(x) - start_rect_idx)
# rect[start_rect_idx:end_rect_idx] = 1

# # convolve 
# image = np.convolve(f, rect, mode="same") / np.sum(rect)
# # image /= expand 

# # display value in range [0, 8pi]
# disp_start, disp_end = int(len(x)//2), int(len(x)//2 + 8*np.pi/x_inter)
# fig = plt.figure()
# # object 
# plt.plot(x[disp_start:disp_end], f[disp_start:disp_end], label="object f(x)")
# # imagE
# plt.step(x[disp_start:disp_end], image[disp_start:disp_end], label="image Hf(x)")
# plt.step((x[disp_start], x[disp_end]), (0, 0), "k--")

# # H*g
# hg = np.convolve(image, rect, mode='same')  / np.sum(rect)
# # hg /= expand
# # check <Hf, g> = <f, H*g>
# inner1 = np.sum(image*image)
# inner2 = np.sum(hg*f)
# # assert inner1 == inner2

# plt.step(x[disp_start:disp_end], hg[disp_start:disp_end], label="H*g")

# plt.legend()
# plt.xlabel("x")
# plt.xlim(x[disp_start], x[disp_end])
# plt.show()

# EX3
# from skimage.transform import radon, iradon
# from scipy.signal import peak_widths

# def width(profile, point):
#     return peak_widths(profile, point)

# R = 64
# C = 64
# image = np.zeros((R, C))
# image[R//2, C//2] = 64
# image[R//4, C//4] = 128

# # number of views
# Ns = [256, 128, 64, 32, 16, 8]
# # Ns = [256]
# fig, axs = plt.subplots(2, 6)
# fig.set_size_inches(12, 4)
# diff_list = []
# for i, N in enumerate(Ns):
#     theta = np.linspace(0, 180, N)
#     sinogram = radon(image, theta=theta)
# #     axs[i].imshow(sinogram, aspect='auto')
# #     axs[i].set_xlim(0, N)
# # plt.legend()
# # plt.show()


#     recon_image = iradon(sinogram, theta=theta, filter="ramp")
#     # get width
#     profile1 = recon_image[:, 32]
#     profile2 = recon_image[:, 16]
#     width1 = peak_widths(profile1, np.array([32])) 
#     width2 = peak_widths(profile2, np.array([16]))
#     print(width1, width2)
#     # fwhm
#     p1_halfmax = recon_image[R//2, C//2] / 2
#     p2_halfmax = recon_image[R//4, C//4] / 2
#     # print(p1_halfmax, p2_halfmax)
#     # print(recon_image[R//2-10:R//2+10, C//2-10:C//2+10] > p1_halfmax)
#     # print(recon_image[R//4-10:R//4+10, C//4-10:C//4+10] > p1_halfmax) 
#     # estimate none-zero area
#     nonzero = np.zeros_like(recon_image)
#     nonzero[recon_image > p1_halfmax] = 1
#     # recon value error 
#     pt1_err = recon_image[R//2, C//2] - image[R//2, C//2]
#     pt2_err = recon_image[R//4, C//4] - image[R//4, C//4]
#     # max error of other non-zero region 
#     diff = np.abs(recon_image - image)
#     diff[R//2, C//2] = 0
#     diff[R//4, C//4] = 0
#     print("Error between original object and reconstructed object: ", pt1_err, pt2_err)
#     print("Other non-zero region error: ", np.max(diff))
#     print("Mean error: ", np.mean(diff))
#     diff_list.append(np.ravel(diff))
#     im = axs[0,i].imshow(image)
#     im = axs[1,i].imshow(recon_image)
#     # im = axs[2,i].imshow(diff)
# # plt.boxplot(np.array(diff_list).transpose())
# # labels = ["N={}".format(x) for x in Ns]
# # plt.xticks(ticks=list(range(1, len(Ns)+1)), labels=labels)
# fig.colorbar(im, ax=axs.ravel().tolist())
# plt.show()

# EX 5
# import cv2
# from scipy.signal import convolve2d

# image = "hw01_ex08_HeLa-cells-from-imagej.png"
# img = cv2.imread(image)
# img = img / 255
# f = img[:,:, 0]
# h, w = f.shape
# c = np.zeros((h, w))
# c[0::4, 0::4] = 1
# # a. diminished f
# fd = f*c
# # fd = convolve2d(f, c, mode="same")
# # b. fft g
# g = np.fft.fft2(fd)
# # g = np.abs(g)
# # c. fft c
# fc = np.fft.fft2(c)
# # print(fc)
# # fc = fc.astype(np.float)
# # d. gd
# # gd = convolve2d(g, fc, mode="same")
# gd = fc * g
# # gd = np.real(gd)
# # e. igd
# igd = np.fft.ifft2(gd)
# # fgd = fgd.astype(np.float)
# igd = np.abs(igd)
# print(igd)

# # fd = fd * c
# plt.imshow(igd)
# plt.colorbar()
# plt.show()

# # EX6
aper_rs = [1e-3,2e-3,4e-3]
plane_size = 1024 # each pixel is 0.1mm
# set aper 
R = 0.001
pixel_size = 32e-6

def aperture(plane_size, pixel_size, aper_r):
    win_size = plane_size * pixel_size
    dk = np.arange(-win_size/2, win_size/2, pixel_size)
    print(dk)
    xx, yy = np.meshgrid(dk, dk)
    plane = np.sqrt(xx**2+yy**2)
    plane = np.where(plane>aper_r, 0, 1)
    return plane

# for idx, aper_r in enumerate(aper_rs):
#     plane = aperture(plane_size, pixel_size, aper_r)

#     # apply fft
#     image = np.fft.fftshift(np.fft.fft2(plane))

#     axs[0, idx].imshow(np.abs(plane), cmap="gray")
#     axs[1, idx].imshow(np.abs(image), cmap="gray")
#     axs[1,idx].set_xlabel('r={}'.format(aper_r))
# axs[0, 0].set_ylabel("circ")
# axs[1, 0].set_ylabel("F{circ}")

lam = 5e-7
Z = 1
detector_plane_size = 1
detector_size = 32e-6
num_detectors = 1024

u = np.linspace(-detector_plane_size/2/detector_size, detector_plane_size/2/detector_size, num_detectors)
print(u)
U,V = np.meshgrid(u, u)
# fresnel approx
H = np.exp(2j*np.pi/lam*Z)*np.exp(-1j*np.pi*lam*Z*(U**2+V**2)) 

plane = aperture(plane_size, pixel_size, R)

# plane_size = 1024
# ###Define the distance of observation z
# Z=0.1
# #Define a scale factor for the coordinates
# # h=10
# ##Define the wavelenght. 
# r = 1e-3
# l_ambda = 1e-2
# pixel_size = 32e-6
# ##Define the angular spectrum coordinates
# u = np.arange(-plane_size/2, plane_size/2, 1)
# v = np.arange(-plane_size/2, plane_size/2, 1)
# U,V = np.meshgrid(u,v)
# plane = np.zeros((plane_size, plane_size))
# center = (plane_size//2, plane_size//2)
# for i in range(plane_size):
#     for j in range(plane_size):
#         y0 = (i-center[0])*pixel_size
#         x0 = (j-center[1])*pixel_size
#         if x0**2+y0**2 <= r**2:
#             plane[i,j] = 1
# #Define the propagation matrix
# # propagator=np.exp(2*np.pi*1j*(Z/h)*np.sqrt((1/l_ambda)**2-(U/10)**2-(V/10)**2))
# H = np.exp(2j*np.pi/l_ambda*Z)*np.exp(-1j*np.pi*l_ambda*Z*(U**2+V**2)) 
# #### Compute the Fast Fourier Transform of the image
O=np.fft.fft2(plane)
# ##Correct the low and high frequencies
Oshift=np.fft.fftshift(O)
##multiply both matrices: Fourier transform and propagator matrices.
image=Oshift*H
##Calculate the inverse Fourier transform
image=np.fft.ifft2(np.fft.ifftshift(image))

plt.imshow(np.abs(Oshift), cmap="gray")
plt.show()
