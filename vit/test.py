import numpy as np
from compute_flops import compute_flops


ratio = (np.random.uniform(0.0, 0.0, size=(1, 25))[0]).tolist()
flops = compute_flops(384, 4, 197, 6, ratio[:12], ratio[12:24], ratio[-1])

print(flops)

