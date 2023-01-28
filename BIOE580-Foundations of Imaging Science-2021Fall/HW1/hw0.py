import numpy as np
import os 
import glob
import cv2
import math

# exercise 1
# root_dir = "BIOE580_hw01_data"
# image_dir = "hw01_ex01_fastMRI-T1-brain-slices"
# images = glob.glob(os.path.join(root_dir, image_dir+"/*.png"))

# print(",".join(images)+"\n")
# for img_a in images:
#     frame_a = cv2.imread(img_a)
#     # normalize image 
#     frame_a = frame_a / 255
#     print("{}:".format(img_a), end="")
#     vector_a = np.ravel(frame_a)
#     for img_b in images:
#         frame_b = cv2.imread(img_b)
#         frame_b = frame_b / 255
#         vector_b = np.ravel(frame_b)
#         cos_sim = np.inner(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
#         angle = np.arccos(cos_sim) / math.pi * 180
#         print("{}/{},".format(cos_sim, angle), end="")
#     print("\n")


# ex2
# import gzip
# H = np.load(gzip.GzipFile(root_dir + "/hw01_ex02_H-operator.npy.gz", "r"))
# print(np.linalg.det(H))

# ex 3
# B1 = np.array([[3,4],[4,-3]])
# B2 = np.array([[1, 2, 2], [6, 3, 6], [6,6, 3]])

# T = np.array([[1, 1j],[1, -1j],[2, 0]])

# T2 = np.matmul(np.linalg.inv(B2), T)
# T2 = np.matmul(T2, B1)

# print(T2)
# u=np.array([[1],[1j]])
# v=np.matmul(T2, u)
# ue=np.matmul(B1, u)
# ve=np.matmul(B2, v)
# print(np.matmul(T, ue))

# ex 7
import matplotlib.pyplot as plt
def toeplitz(r, c, sig):
    return np.exp(-(r-c)**2/sig)

vector = list(range(1, 65))
c_matrix = np.repeat([vector], 64, axis=0)
r_matrix = np.repeat(np.expand_dims(vector, -1), 64, axis=1)

# H = toeplitz(r_matrix, c_matrix, 8)
# # print(H)
# # plt.imshow(H)
# # plt.show()
# # print rank 
# sigs = [2**i for i in range(3,9)]
# nullity = []
# for sig in sigs:
#     H = toeplitz(r_matrix, c_matrix, sig)
#     nullity.append(64-np.linalg.matrix_rank(H))
#     # print(np.linalg.det(H))
# print(np.linalg.matrix_rank(H))
# plt.plot(sigs, nullity, "o-")
# plt.xlabel("Sigma")
# plt.ylabel("Nullity")
# plt.show()

# ex 8
from scipy.linalg import orth
H32 = toeplitz(r_matrix, c_matrix, 32)
A = orth(H32)
# check orthogonal
# res = np.matmul(A, np.transpose(A))

P = np.matmul(A, np.linalg.inv(np.matmul(A.transpose(), A)))
P = np.matmul(P, np.transpose(A))
# print(np.sum(P))
# print(np.linalg.matrix_rank(P))
P_null = np.identity(64) - P

img = cv2.imread("BIOE580_hw01_data/hw01_ex08_HeLa-cells-from-imagej.png")
img = img[:,:, 0]
frame1 = np.matmul(P, img)
frame2 = np.matmul(P_null, img)
frame = np.concatenate([frame1, frame2], axis=1)
# plt.imshow(frame)
# plt.show()
vector_a = np.ravel(frame1)
vector_b = np.ravel(frame2)
cos_sim = np.inner(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
angle = np.arccos(cos_sim) / math.pi * 180
print(cos_sim, angle)