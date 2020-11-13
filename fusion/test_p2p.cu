#include <cuda_runtime.h>
#include <stddef.h>
#include <string>
#include <vector>
#include <iostream>

#include <unistd.h>


int main(){
    int count = 8;
    std::vector<int> devices={0,1,2,3,4,5,6,7};

    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j) {
            if (devices[i] == devices[j]) continue;
            int can_acess = -1;
            int ret = cudaDeviceCanAccessPeer(&can_acess, devices[i], devices[j]);
            if (can_acess != 1) {
                std::cout << "Cannot enable P2P access from " << devices[i] << " to " << devices[j];
            } else {
                cudaSetDevice(devices[i]);
                cudaDeviceEnablePeerAccess(devices[j], 0);
            }
        }
    }

    
   usleep(1000 * 1000 * 1000);
   return 0;
}
