
encode = lambda s : s.encode('utf-8')

class HIP_tools:    
  def __init__(self,  hip_lib_path):


    from ctypes import cdll, c_int, c_char_p        
    print(f'Loading library: {hip_lib_path}')
    libhip = cdll.LoadLibrary( hip_lib_path )
    
    self._set_device = libhip.set_device
    self._set_device.argtypes = [ c_int ]
    self._set_device.resypes = c_int

    self._start_roctracer = libhip.start_roctracer
    self._stop_roctracer = libhip.stop_roctracer
    self._roctxr_push = libhip.roctxr_push
    self._roctxr_pop = libhip.roctxr_pop    
    self._roctxr_push.argtypes = [c_char_p]
    self._roctxr_start = libhip.roctxr_start
    self._roctxr_stop  = libhip.roctxr_stop
    self._roctxr_start.argtypes = [ c_char_p ]
    self._roctxr_start.resypes = c_int 
    self._roctxr_stop.argtypes = [ c_int ]
  

  def set_device(self, device_id ):
    self._set_device(device_id)

  def start_roctracer(self):
    self._start_roctracer()

  def stop_roctracer(self):
    self._stop_roctracer()   

  def start_marker( self, marker_name ):
    marker_id = self._roctxr_start( encode(marker_name) )
    return marker_id
  
  def stop_marker( self, marker_id, sync_device=False, device=None ):
    # if sync_device: torch.cuda.synchronize(device=device)
    self._roctxr_stop( marker_id )
