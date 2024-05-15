import numpy as np
from models.sam.utils.amg import build_point_grid

nps = 64
pb = 256
point_grids = build_point_grid(nps)
ps = point_grids*1024
point_available = np.ones((nps*nps,), dtype=int)
while np.sum(point_available) != 0:
    indices, = np.nonzero(point_available)
    if len(indices) >= pb:
        selected_idx = np.random.choice(indices, size=pb)
        points_new = ps[selected_idx]
        point_available[selected_idx] = 0
        print()