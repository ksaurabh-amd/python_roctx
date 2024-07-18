import os, sys
import numpy as np
from hip_tools import HIP_tools

use_torch = False
if use_torch:  import torch


# Get the path to the hip code library
work_dir = os.getcwd()
hip_lib_path = f'{work_dir}/libHIPcode.so'

# Initialize the roctracer tools
hip_tools = HIP_tools( hip_lib_path )

if use_torch:
  if not torch.cuda.is_available():
    print('Warning GPU not found')
    device = torch.device('cpu')
  else:  
    print('Setting torch device cuda')
    device = torch.device('cuda')
else:
  # Set the device: Needed since nothing else initialize the device 
  print('Setting hip device 0')
  hip_tools.set_device(0)  


# Do some fun stuff
id_init = hip_tools.start_marker('init')

nx, ny = 1024, 1024
A = np.random.rand( nx, ny )
B = np.random.rand( nx, ny )

hip_tools.stop_marker(id_init)

id_main = hip_tools.start_marker('main')

n_iterations = 20
for i in range(n_iterations):

  id_iter = hip_tools.start_marker(f'iter_{i}')
  print( f'iteration: {i}')
  C = np.matmul(A, B)

  hip_tools.stop_marker(id_iter)


hip_tools.stop_marker(id_main)