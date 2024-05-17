from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
args = parser.parse_args()

import numpy as np
from numpy.testing import assert_allclose
npz_data = np.load(args.i)
keys = ['quickshift', 'global_rag', 'manual_rag']

def unsplit_np(patches):
    return np.hstack([np.vstack(p) for p in patches])
from math import prod
original_shape = unsplit_np(npz_data[keys[0]]).shape
buffer_size =  3 * prod(original_shape) * 8 * 1
buffer_size += 5 * prod(original_shape) * 1 * 4
buffer_size += 1 *                    1 * 8 * 4
header = np.asarray([buffer_size, original_shape[1], original_shape[0], 7]).astype(np.uint64)
# Write the header
buffer = bytearray(header.tobytes())
# Dummy images
buffer.extend(np.zeros((5 * prod(original_shape) * 1 * 4), dtype=np.uint8))
# The segmentations
for k in keys:
    buffer.extend(unsplit_np(npz_data[k]).astype(np.uint64).tobytes())

with open(args.o, "wb") as f:
    f.write(bytes(buffer))
