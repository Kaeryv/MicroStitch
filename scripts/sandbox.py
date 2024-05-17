from skimage.color import rgb2xyz, xyz2lab
from skimage import img_as_float
from skimage._shared.filters import gaussian
import numpy as np
N=500
img = (np.random.rand(N,N,3) * 256).astype(np.uint8)
img = xyz2lab(rgb2xyz(img))
img = img_as_float(img)
print(img[...,0].min(), img[...,0].max())
print(img[...,1].min(), img[...,1].max())
print(img[...,2].min(), img[...,2].max())
img = gaussian(img, 0, mode='reflect')
print(img[...,0].min(), img[...,0].max())
print(img[...,1].min(), img[...,1].max())
print(img[...,2].min(), img[...,2].max())

