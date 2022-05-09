# import psutil
# import os
# import time
# import sys
# import math
# from contextlib import contextmanager

# @contextmanager
# def trace(title=''):
#     t0 = time.time()
#     p = psutil.Process(os.getpid())
#     m0 = p.memory_info()[0] / 2. ** 30
#     yield
#     m1 = p.memory_info()[0] / 2. ** 30
#     delta = m1 - m0
#     sign = '+' if delta >= 0 else '-'
#     delta = math.fabs(delta)
#     print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)