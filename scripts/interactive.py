from math import ceil
import json
from json import JSONEncoder
from tqdm import tqdm
from functools import partial
from copy import deepcopy as copy

import numpy as np
import matplotlib.pyplot as plt

from skimage import graph
from helper import regionprops

import cv2

from scipy import ndimage

from skimage import morphology, segmentation, feature, img_as_float
from skimage.color import rgb2gray

from itertools import cycle

import os

from helper import ensure_contains

disp_bg = cycle(["polarized", "phi_raw"])
show_segment = False
cur_disp_bg = next(disp_bg)
cur_disp_img = "base"

variables = list()
img = cv2.imread("img3.png")
sizey, sizex = img.shape[:2]
from helper import unsplit, unsplit_np, split, color_labels
mapping = {}
w = 512
img= cv2.copyMakeBorder(img.copy(),0,w - sizey%w,0,w-sizex%w,cv2.BORDER_CONSTANT,value=(0,0,0))
smallWindowName = 'inset'
filteredWindow = 'filtered'
slidersWindow = "params"
patches_y, patches_x = [ceil(s/w) for s in np.asarray(img).shape[0:2]]
nx, ny = 0, 0

phi_img = cv2.imread('22fep01_s0.phi.png')
print(phi_img.shape)
print(img.shape)

images = { key: [ [ None for j in range(patches_y) ] for i in range(patches_x) ]  for key in ["base", "denoised", "boundaries", "quickshift", "global_rag", "manual_rag", "phi", "phi_raw"]}
for nx in range(patches_x):
    for ny in range(patches_y):
        base_patch = np.asarray(img)[ny*w:(ny+1)*w,nx*w:(nx+1)*w]
        images["base"][nx][ny]       = base_patch
        images["denoised"][nx][ny]   = base_patch
        images["boundaries"][nx][ny] = base_patch
        # Segmentations
        labels_shape = base_patch.shape[0:2]
        images["quickshift"][nx][ny] = np.zeros(labels_shape, dtype=np.int64)
        images["global_rag"][nx][ny] = np.zeros(labels_shape, dtype=np.int64)
        images["manual_rag"][nx][ny] = np.zeros(labels_shape, dtype=np.int64)
        # Phi map
        images["phi"][nx][ny] = np.zeros_like(base_patch)
        images["phi_raw"][nx][ny] = np.asarray(phi_img)[ny*w:(ny+1)*w,nx*w:(nx+1)*w]
nx, ny = 0, 0


default_params = {
    "strength": 10,
    "pattern_size":7,
    "outer_search":21,
    "coloring_threshold": 5,
    "QS_balance": 70,
    "QS_sigma": 10,
    "QS_kernel": 9,
    "QS_max_size": 20,
}

params = [ [ copy(default_params) for j in range(patches_y) ] for i in range(patches_x) ]


if os.path.isfile("save.json"):
    with open("save.json", "r") as f:
        params = json.load(f)

if os.path.isfile("save.npz"):
    images = {}
    for key, val in np.load("save.npz").items():
        images[key] = val


from helper import dark_tile
def draw_inset():
    global nx, ny, images, cur_disp_bg, dark, show_segment, cur_disp_img
    if cur_disp_bg == "polarized":
        if show_segment:
            cur_disp_img = "boundaries"
        else:
            cur_disp_img = "base"
    elif cur_disp_bg == "phi_raw":
        if show_segment:
            cur_disp_img = "phi"
        else:
            cur_disp_img = "phi_raw"

    imgs = images[cur_disp_img]

    dark = dark_tile(w)
    tiles = [
            [dark, dark, dark],
            [dark, dark, dark],
            [dark, dark, dark]
        ]
    tiles[1][1] = imgs[nx][ny]
    if nx > 0:
        tiles[1][0] = imgs[nx-1][ny+0]
    if ny > 0:
        tiles[0][1] = imgs[nx][ ny-1]
    if ny > 0 and nx > 0:
        tiles[0][0] = imgs[nx-1][ ny-1]
    if ny < patches_y-1:
        tiles[2][1] = imgs[nx][ny+1]
    if nx < patches_x-1:
        tiles[1][2] = imgs[nx+1][ny]
    if nx < patches_x-1 and ny < patches_y-1:
        tiles[2][2] = imgs[nx+1][ny+1]
    if ny < patches_y-1 and nx > 0:
        tiles[2][0] = imgs[nx-1][ny+1]
    if ny > 0 and nx < patches_x-1:
        tiles[0][2] = imgs[nx+1][ny-1]
    return cv2.vconcat([cv2.hconcat(tiles[i]) for i in range(3)])



sel_labels={}
def global2inset(c, r, nx, ny, w=512):
    return c - (nx-1) * w, r - (ny-1)*w

props = None
minimap = None
def update_inset(X, progress=None, message="Operation in progress"):
    global cur_disp_img, sel_labels, current_inset, images, props, minimap, mapping
    font = cv2.FONT_HERSHEY_SIMPLEX
    UI = cv2.resize(X.copy(),(w*3,w*3))
    if progress:
        UI = cv2.rectangle(UI, (0,0), (3*w,10*w//100), (255,255,255,255), 4)
        UI = cv2.rectangle(UI, (0,0), (3*round(progress*w),10*w//100), (255,255,255,255), -1)
        UI = cv2.putText(UI, message, (6*w//5, 30), font, 1, (200,100,255), 2)
    #UI = cv2.putText(UI.copy(), f"nx:{nx} ny:{ny}", (0, 400), 1, 5, (255,0,0), 10)
    cv2.setWindowTitle(smallWindowName, cur_disp_img)
    for i, (sel_label, label_description) in enumerate(sel_labels.items()):
        label_integer = int(sel_label)
        if not props:
            segmentation = unsplit(images["manual_rag"]).astype(np.int64)
            props = regionprops(segmentation)
        prop = props[label_integer]
        UI = cv2.rectangle(UI, (0, 29+i*25), (500, 29+(i-1)*25), (255, 255, 255), -1)
        UI = cv2.putText(UI, f"Segment: {int(sel_label)} Area: {int(prop.area)}", (0, 25+i*25), font, 0.5, (0,0,255), 2)

        if cur_disp_img == "boundaries":
            minr, minc, maxr, maxc = prop.bbox
            
            gminc, gminr = global2inset(minc, minr, nx, ny)
            gmaxc, gmaxr = global2inset(maxc, maxr, nx, ny)
            plop, red = ensure_contains((0, 3*w, 0, 3*w), (gminc, gmaxc, gminr, gmaxr))
            gminc, gmaxc, gminr, gmaxr=plop
            mask = prop.image==1
            if red[1] > 0:
                mask = mask[:,:-red[1]]
            if red[0] > 0:
                mask = mask[:,red[0]:]
            if red[2] > 0:
                mask = mask[red[2]:,:]
            if red[3] > 0:
                mask = mask[:-red[3],:]
            #UI = cv2.rectangle(UI, (gminc, gminr), (gmaxc, gmaxr), (255,255,255,255), 4)

            #cv2.imshow("test", cv2.cvtColor((255*prop.image).astype(np.uint8), cv2.COLOR_GRAY2RGB))
            UI[gminr:gmaxr, gminc:gmaxc, 0][mask] //= 2
            UI[gminr:gmaxr, gminc:gmaxc, 1][mask] //= 2
            UI[gminr:gmaxr, gminc:gmaxc, 2][mask] = 255
            UI = cv2.putText(UI, f"{int(sel_label)}", ((gminc+gmaxc)//2, (gminr+gmaxr)//2), font, 0.5, (255,255,255), 2)
    
    UI = cv2.rectangle(UI, (512,512), (1024,1024), (255,200,255), 2)
    UI[-256:,-256:] = minimap
    UI = cv2.rectangle(UI, (3*w-256,3*w-256), (3*w,3*w), (255,200,255), 2)
    cv2.imshow(smallWindowName, UI)
    cv2.waitKey(1)

def update_sliders():
    for title,name in variables:
        cv2.setTrackbarPos(title, slidersWindow, round(params[nx][ny][name]))

def onchange(value, variable="x"):
    global nx, ny, inset_img,minimap
    inset_img = draw_inset()
    minimap = cv2.rectangle(img.copy(), (nx*w,ny*w), ((nx+1)*w,(ny+1)*w), (255,200,255,255), 10)
    minimap = cv2.resize(minimap,(256,256))
    update_inset(inset_img)
    update_sliders()
    update_result_window()

launched = False
def treatment_params_cb(value, variable):
    global params, launched
    if launched:
        params[nx][ny][variable] = value


cv2.namedWindow(slidersWindow, cv2.WINDOW_GUI_EXPANDED)#, #cv2.WINDOW_NORMAL)
cv2.resizeWindow(slidersWindow, 400, 400)
def register_variable(title, name, min=0, max=100):
    global params,variables
    cv2.createTrackbar(title, slidersWindow, min, max, partial(treatment_params_cb, variable=name))
    cv2.setTrackbarPos(title, slidersWindow, round(params[nx][ny][name]))
    variables.append((title,name))


register_variable('AV_STR', "strength")
register_variable('AV_PS', "pattern_size")
register_variable('AV_OS', "outer_search")
register_variable('RA_CT', 'coloring_threshold', min=0, max=round(np.sqrt(3)*1000))
register_variable("QS_BL", "QS_balance")
register_variable("QS_SG", "QS_sigma")
register_variable("QS_KL", "QS_kernel")
register_variable("QS_MR", "QS_max_size")
def circle(N):
    x = np.linspace(-0.5, 0.5, N)
    X,Y = np.meshgrid(x,x)
    return 0.5**2 > X**2+Y**2 
n_real = 400
from skimage import measure, morphology

cv2.namedWindow(filteredWindow)
from skimage.transform import rescale, resize, downscale_local_mean

def np2cv(X):
    return (255 * X).astype(np.uint8)
def update_result_window():
    global images
    disp_quickshift   = np2cv(segmentation.mark_boundaries(images["base"][nx][ny], images["quickshift"][nx][ny]))
    disp_global_rag   = np2cv(segmentation.mark_boundaries(images["base"][nx][ny],images["global_rag"][nx][ny]))
    main = cv2.hconcat([
            images["denoised"][nx][ny], 
            disp_quickshift,
            disp_global_rag
        ])
    final = cv2.resize(main,(w*3//2,w//2))
    cv2.imshow(filteredWindow, final)
    cv2.waitKey(1)

update_result_window()
launched = True

#cv2.moveWindow(smallWindowName, 512,0)
cv2.moveWindow(filteredWindow , 0,25+512)
cv2.moveWindow(slidersWindow, 25+2*512,512)


def get_label_from_inset_position(x, y):
    # Get global position
    gx = nx * w + x - w
    gy = ny * w + y - w
    return int(unsplit(images["manual_rag"])[gy, gx]), int(gx), int(gy)

inset_img = None
def mouse_callback(event, x, y, flags, params):
    global sel_labels, inset_img, images
    if event == 2:
        #x = min(x, inset_img.shape[1]-1)
        #y = min(y, inset_img.shape[1]-1)
        lab, gx, gy = get_label_from_inset_position(x, y)
        if not lab in sel_labels.keys():
            sel_labels[lab] = { "gx": gx, "gy": gy, "x": x, "y": y}
        else:
            del sel_labels[lab]
        
        update_inset(inset_img)
    #if event == 1:
    #    lab, gx, gy = get_label_from_inset_position(x, y)
    #    labels = unsplit_np(images["manual_rag"]).astype(np.uint64)
    #    bs = 50
    #    labels[gy-bs:gy+bs,gx-bs:gx+bs] = labels.max()+1
    #    images["manual_rag"] = split(labels)

def op_denoise(X, p):
    return cv2.fastNlMeansDenoisingColored(
        X.copy(), None, p["strength"], 10, p["pattern_size"], p["outer_search"]
    )
def op_quickshift(X, p):
    labels = segmentation.quickshift(X, p["QS_balance"]/100.0, p["QS_kernel"], convert2lab=True, sigma=p["QS_sigma"], max_dist=p["QS_max_size"])
    return labels
from skimage.color import rgb2lab
def op_rag(denoised, segmap, p={}):
    denoised = rgb2lab(denoised)
    print(np.max(denoised[:,:,1]), np.max(denoised[:,:,0]))
    import adjacency_matrix
    num_components = segmap.max()+1
    adjmat = np.asfortranarray(np.zeros((segmap.max()+1, segmap.max()+1), dtype=np.int64))
    distmat = np.asfortranarray(np.zeros((segmap.max()+1, segmap.max()+1), dtype=np.float32))
    denoised = np.asfortranarray(np.ascontiguousarray(denoised))
    adjacency_matrix.adjacency_matrix(segmap, adjmat, num_components)
    mean_colors = np.asfortranarray(np.zeros((num_components,4), dtype=np.float32))
    adjacency_matrix.color_distance_matrix(denoised, segmap, adjmat, distmat, mean_colors)
    # import matplotlib.pyplot as plt
    # fig, (ax1,ax2) = plt.subplots(1,2)
    # ax1.matshow(distmat)
    # ax2.matshow(adjmat>0)
    # plt.show()
    # print(num_components)
    mapping = np.asfortranarray(np.arange(num_components, dtype=np.int64))
    #print(np.unique(mapping))
    adjacency_matrix.rag_merge(adjmat, distmat, mapping, p["coloring_threshold"]/100., mean_colors)
    #mapping_dsts = np.unique(mapping)
    #segmap_values = np.unique(mapping[segmap])
    # print(np.unique(segmap))
    #segmap = mapping[segmap]
    # print(np.unique(segmap_values))
    # print(mapping_dsts)
    
    return mapping[segmap]


def apply_all(operation, pic_in, pic_out):
    global images
    inset = draw_inset()
    for xx in range(patches_x):
        for yy in range(patches_y):
            p = params[xx][yy]
            images[pic_out][xx][yy] = operation(images[pic_in][xx][yy], p)
            update_inset(inset, progress=(yy+xx*patches_y)/patches_x/patches_y)

onchange(0, "x")
cv2.setMouseCallback(smallWindowName, mouse_callback)


def update_segmentation_display():
    global images, inset_img, props
    segm = unsplit(images["manual_rag"]).astype(np.uint64)
    images["boundaries"] = split((segmentation.mark_boundaries(unsplit(images["base"]), segm)*255).astype(np.uint8))
    images["phi"] = split((segmentation.mark_boundaries(unsplit(images["phi_raw"]), segm)*255).astype(np.uint8))
    props = regionprops(segm)
    update_result_window()
    inset_img = draw_inset()
    update_inset(inset_img)

while True:
    key = cv2.waitKey(50)
    p = params[nx][ny]
    if key == ord('q'):
        break
    elif key == ord('v'):
        cur_disp_bg = next(disp_bg)
        inset_img = draw_inset()
        update_inset(inset_img)
    elif key == ord('V'):
        show_segment = not show_segment
        inset_img = draw_inset()
        update_inset(inset_img)
    elif key == ord('d'):
        params[nx][ny] = copy(default_params)
        update_sliders()
    elif key == ord("t"):
        im_work = images["base"][nx][ny].copy()
        images["denoised"][nx][ny] = cv2.rectangle(im_work.copy(), (w//4,w//4), (3*w//4,3*w//4), (255,0,0,255), 6)
        update_result_window()
        images["denoised"][nx][ny] = op_denoise(im_work, p)
        update_result_window()
        if cur_disp_img == "denoised":
            inset_img = draw_inset()
        update_inset(inset_img)
    elif key == ord("T"):
        images["denoised"] = split(op_denoise(unsplit(images["base"]), p=default_params))
        #apply_all(op_denoise, "base", "denoised")
        if cur_disp_img == "denoised":
            inset_img = draw_inset()
        update_inset(inset_img)


    elif key == ord("y"):
        # Override the quickshift in one cell.
        # We ensure that all labels are new labels.
        update_inset(inset_img, progress=0.0, message="Quickshift in progress")
        cur_max_labels = [ 
            round(unsplit(images["quickshift"]).max()),
            round(unsplit(images["global_rag"]).max()),
            round(unsplit(images["manual_rag"]).max())
        ]
        cur_max_label = np.max(cur_max_labels)
        images["quickshift"][nx][ny]  = 1 + cur_max_label + op_quickshift(images["denoised"][nx][ny], p)
        update_inset(inset_img, progress=1.0, message="Quickshift Done")
        update_result_window()
    elif key == ord("Y"):
        update_inset(inset_img, progress=0.0, message="Quickshift in progress")
        images["quickshift"] = split(op_quickshift(unsplit(images["denoised"]), default_params))
        update_inset(inset_img, progress=1.0, message="Quickshift Done")

    elif key == ord("u"):
        images["global_rag"][nx][ny] = op_rag(images["denoised"][nx][ny].astype(float) / 255, images["quickshift"][nx][ny], p=params[nx][ny])
        update_result_window()
        update_inset(draw_inset())
    elif key == ord("U"):
        segmap = unsplit_np(images["quickshift"]).astype(np.int64)
        denoised = unsplit_np(images["denoised"]).astype(float) / 255.0
        images["global_rag"] = split(op_rag(denoised, segmap, p=default_params))
        update_result_window()
        update_inset(draw_inset())

    elif key == ord("b"):
        update_segmentation_display()
    elif key == ord("i"):
        images["manual_rag"][nx][ny] = images["global_rag"][nx][ny]
        update_segmentation_display()
    elif key == ord("I"):
        apply_all(lambda x,p: x, "global_rag", "manual_rag")
        update_segmentation_display()
    elif key == ord("o"):
        images["global_rag"][nx][ny] = images["quickshift"][nx][ny]
        update_result_window()
    elif key == ord("O"):
        apply_all(lambda x,p: x, "quickshift", "global_rag")
        update_result_window()
    elif key == 0:
        ny = ny - 1 if ny > 0 else ny
        update_result_window()
        onchange(nx, "x")
    elif key == 1:
        ny = ny + 1 if ny < patches_y-1 else ny
        update_result_window()
        onchange(nx, "x")
    elif key == 2:
        nx = nx - 1 if nx > 0 else nx
        update_result_window()
        onchange(nx, "x")
    elif key == 3:
        nx = nx + 1 if nx < patches_x-1 else nx
        update_result_window()
        onchange(nx, "x")
    elif key == ord("s"):
        update_inset(inset_img, progress=0)
        with open("save.json", "w") as f:
            json.dump(params, f)
        update_inset(inset_img, progress=0.5, message="Saving...")
        np.savez_compressed("save.npz", **images)
        update_inset(inset_img, progress=1.0, message="Saving done")

    elif key == ord("D"):
        default_params = copy(params[nx][ny])
  
    elif key == ord("e"):
        update_inset(inset_img, progress=0)
        params = [ [ copy(default_params) for j in range(patches_y) ] for i in range(patches_x) ]
        update_inset(inset_img, progress=1)
    
    elif key == ord("J"):
        # Join current selected labels
        labels = list(sel_labels.keys())
        segmap = unsplit(images["manual_rag"]).astype(np.int64)
        if len(labels) > 1:
            dst = labels[0]
            for l in labels[1:]:
                segmap[segmap==l] = dst
                del sel_labels[l]
            sel_labels.clear()
            segmap,_,__ = segmentation.relabel_sequential(segmap)
            images["manual_rag"] = split(segmap)
            images["boundaries"] = split((segmentation.mark_boundaries(unsplit(images["base"]), segmap)*255).astype(np.uint8))
            props = regionprops(segmap)
            inset_img = draw_inset()
            update_inset(inset_img)

    elif key == ord("p"):
        segmap = unsplit(images["manual_rag"]).astype(np.int64)
        phimap = unsplit(images["phi"])

        for i, label in enumerate(tqdm(np.unique(segmap))):
            mask = segmap == label
            pixels = phi_img[mask,:]
            pixels = pixels[np.logical_and(pixels[:,2]!=255, np.logical_and(pixels[:,0]!=255, pixels[:,1] !=255))]
            phimap[mask, :] = np.mean(pixels, axis=0)
        images["phi"] = split(phimap)
    
    elif key == ord("x"):
        rgba = cv2.cvtColor(unsplit(images["base"]), cv2.COLOR_RGB2RGBA)
        cv2.imwrite("background.png", rgba)




cv2.destroyAllWindows()
