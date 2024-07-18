#include <stdio.h>
#include <iostream>
#include <roctx.h>
#include <roctracer_ext.h>
#include <hip/hip_runtime.h>

#define CHECK(command) {   \
  hipError_t status = command; \
  if (status!=hipSuccess) {    \
    std::cout << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
    std::abort(); }}


extern "C" {


int set_device( int proc_id ) {

  int n_devices; 
	CHECK( hipGetDeviceCount(&n_devices) );

	if (n_devices == 0){
		std::cout << "MPI rank= " << proc_id << " NO DEVICES FOUND!" << std::endl;
		return 0;
	}

	int device = proc_id % n_devices;
	CHECK( hipSetDevice(device) ); 
	std::cout << "MPI rank= " << proc_id << " will use GPU ID " << device << " / " << n_devices << std::endl;
	return device;

}

void start_roctracer(){
  roctracer_start();
}

void stop_roctracer(){
  roctracer_stop();
}

int roctxr_start( char *c){
  // std::cout << "Starting roctx marker: " << c << std::endl; 
  int id = roctxRangeStart(c);
  return id;
}

void roctxr_stop( int id){
  roctxRangeStop(id);
}

void roctxr_push( char *c){
  roctxRangePush(c);
}

void roctxr_pop(){
  roctxRangePop();
}


}