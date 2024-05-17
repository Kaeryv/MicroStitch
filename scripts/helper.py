from skimage import measure, morphology
from scipy import ndimage
import numpy as np
import cv2
from functools import cache
from skimage import graph
from math import ceil
from itertools import cycle

cmap = cycle([[round(255*np.random.rand()) for l in range(3)] for i in range(1000)])

def hysteresis_thresholding(im, v_low, v_high):
    mask_low = im > v_low
    mask_high = im > v_high
    # Connected components of mask_low
    labels_low = measure.label(mask_low, background=0) + 1
    count = labels_low.max()
    # Check if connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(count + 1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums[1:] > 0
    output_mask = good_label[labels_low]
    return output_mask   



@cache
def dark_tile(N, num_channels=3):
    return np.zeros((N,N,num_channels), dtype=np.uint8)



    
def unsplit(patches):
    return cv2.hconcat([cv2.vconcat(p) for p in patches])
def unsplit_np(patches):
    return np.hstack([np.vstack(p) for p in patches])

def split(X, w=512):
    patches_y, patches_x = [ceil(s/w) for s in X.shape[0:2]]
    splitted = [ [ None for j in range(patches_y) ] for i in range(patches_x) ]
    for nx in range(patches_x):
        for ny in range(patches_y):
            splitted[nx][ny] = np.asarray(X)[ny*w:(ny+1)*w,nx*w:(nx+1)*w].astype(X.dtype)
    return splitted

def contains(r1, r2):
   return r1[0] < r2[0] < r2[1] < r1[1] and r1[2] < r2[2]< r2[3] < r1[3]

def ensure_contains(r1, r2):
    xm = max(r1[0], r2[0])
    xM = min(r2[1], r1[1])
    ym = max(r1[2], r2[2])
    yM = min(r2[3], r1[3])
    return (xm, xM, ym, yM), (xm-r2[0], r2[1]-xM, ym-r2[2], r2[3]-yM)

def color_labels(t):
    sd= np.zeros((t.shape[0], t.shape[1],3),dtype=np.uint8)
    icmap = iter(cmap)
    for k in range(int(t.max())):
        sd[t==k] = next(icmap)
    return sd

def smallestbox(a):
    r = a.any(1)
    if r.any():
        m,n = a.shape
        c = a.any(0)
        out = a[r.argmax():m-r[::-1].argmax(), c.argmax():n-c[::-1].argmax()]
        return out, (r.argmax(), c.argmax(), m-r[::-1].argmax(), n-c[::-1].argmax())
    else:
        out = np.empty((0,0),dtype=bool)
        return out, (0,0,0,0)
from types import SimpleNamespace
def regionprops(labels):
    return SegmentPropertiesMap(labels)

class SegmentPropertiesMap(object):
    def __init__(self, labels) -> None:
        self.entries = {}
        self.identifiers = np.unique(labels.flatten())
        self.labels = labels.astype(np.uint64).copy()
    
    def compute_single(self, identifier):
        image, bbox = smallestbox(self.labels == identifier)
        area = np.count_nonzero(image)
        return SimpleNamespace(**{"area": area, "image": image, "bbox": bbox})
    
    def compute_all(self):
        for id in self.labels:
            self.entries[id] = self.compute_single(id)

    def __getitem__(self, identifier):
        if identifier in self.entries.keys():
            return self.entries[identifier]
        else:
            self.entries[identifier] = self.compute_single(identifier)
            return self.entries[identifier]
    def keys(self):
        return self.entries.keys()
