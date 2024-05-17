import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.segmentation import mark_boundaries
from skimage import img_as_float
from helper import unsplit_np
from invert_cmap import img_inv_cmap
import matplotlib.pyplot as plt
from scipy import stats as st
path = "save.npz"
data = np.load(path)
segment = unsplit_np(data["manual_rag"])
phimaps = np.array(Image.open("./22fep01_s0.phi.png"))
phimaps = img_as_float(phimaps)
old = phimaps.copy()
mask = np.logical_and(phimaps[:,:,0]==1, np.logical_and(phimaps[:,:,1]==1, phimaps[:,:,2]==1))
phimaps = rgb2hsv(phimaps[:,:,:3])
phi_raw = phimaps[:,:,0]
phi_raw[mask] = np.nan
from tqdm import tqdm
from scipy import stats
import os
if os.path.isfile("phimap.npz"):
    data = np.load("phimap.npz")
    cailloux = data["c"]
    new_phimap = data["phi"]
    new_pmap = data["pval"]
else:
    cailloux = list()
    new_phimap = np.empty_like(phi_raw)
    new_pmap = np.empty_like(phi_raw)
    for i in tqdm(np.unique(segment)):
        mask = segment == i
        phis = phi_raw[mask]
        nanmask = ~np.isnan(phis)
        fillfactor = (np.count_nonzero(nanmask) / phis.size)
        if fillfactor > 0.1:
            phis = phis[nanmask]
            phis = np.round(phis*180)
            phis = phis.flatten()
            if phis.size < 8:
                phis = np.ones((8)) * np.mean(phis)
            k2, new_pmap[mask] = stats.normaltest(phis)
            mode = st.mode(phis).mode
            mode = mode[0] if len(mode) > 0 else np.nan
        else:
            mode = np.nan
        mode = mode / 180
        cailloux.append((phis.size, mode))
        #mean = np.nanmean(phis)
        new_phimap[mask] = mode
    np.savez("phimap.npz", phi=new_phimap, c=cailloux, pval=new_pmap)
    
cailloux = np.asarray(cailloux)
cailloux = cailloux[cailloux[:, 0]> 500, :]
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2)

ax1.hist(-90+new_phimap.flatten()*180, bins=50)
ax2.matshow(phi_raw, cmap="hsv", vmin=0.0, vmax=1.0)
# ax3.imshow(mark_boundaries(new_phimap, segment))
#ax3.scatter(np.sqrt(cailloux[:, 0]), cailloux[:,1]*180)
ax3.matshow(new_pmap, cmap="hot")
ax4.matshow(new_phimap, cmap="hsv", vmin=0.0, vmax=1.0)
plt.show()

