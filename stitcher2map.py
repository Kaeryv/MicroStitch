from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--json", type=str, required=True)
parser.add_argument("--background", type=str, required=True)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

import numpy as np
from PIL import Image
from tqdm import tqdm
import json


with open(args.json) as f:
    data = json.load(f)

patches = data["patches"]
background = data["background"]

x0, y0, w0, h0 = [ background[e] for e in ["x", "y", "w", "h"]]
phi_map = Image.open(background["file"])

if not args.debug:
    phi_map = Image.fromarray(np.ones_like(phi_map) * 255)

name = args.json.replace(".json", ".phi.png")
angle = data["global"]["angle"]
scale = data["global"]["scale"]
#folder = data["global"]["folder"]
folder = "pngs_22fep01_s0"

for i, patch in enumerate(tqdm(patches)):
    x, y, w, h, fn,r = [ patch[e] for e in ["x", "y", "w", "h", "fn", "r"]]
    x = (x-x0)/w0
    y = (y-y0)/h0
    w = w / w0 * scale * 2
    h = h / h0 * scale * 2
    try:
        img = Image.open(f"{folder}/{i}.png")
        x = round(x * phi_map.size[0])
        y = round(y * phi_map.size[1])
        w = round(w * phi_map.size[0])
        h = round(h * phi_map.size[1])
        img = img.resize((w,h))
        img = img.rotate(180+r)
        phi_map.paste(img, (x, y, x+w, y+h))
    except:
        print(f"Image {i} has a problem.")


phi_map.save(name)
exit()

'''
    print(extent)
    save["patches"].append(deepcopy({"x": x, "y": y, "w":w, "h":h, "dpi": dpi, "rotation": rot}))
    x = x - x0 
    y = y - y0 

    w = w / w0 * img0.shape[1]
    h = h / h0 * img0.shape[0]

    x = x / w0 * img0.shape[1]
    y = y / h0 * img0.shape[0]-w - 0.05*w
    #y = img0.shape[0] - y / h0 * img0.shape[0]-w - 0.05*w
    x = x + h * 3 / 180 * 3.14
    resize_factor = dpi[0] / dpi0[0]
    #x, y, w, h = [ int(e * ) for e in extent]
    pimg = Image.fromarray(np.fliplr(np.rot90(img)))
    pimg.save(f"{i}.png")
    #pimg.putalpha(128)

    #pimg = pimg.rotate(-3, center=(0,0), expand=True,fillcolor=(0,0,0,0))
    #pimg = pimg.resize([ round(e*resize_factor) for e in pimg.size])
    #pil_img.paste(pimg, (round(x), round(y)), pimg) 
    #pil_img.putalpha(255)

'''
