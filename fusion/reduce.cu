#include "fusion_api.h"

__global__ void fuse_elementwise_reduce_dot_elementwise(float* mul_1, const float* X, const float* Y) {
}

__global__ void fuse_elementwise_reduce_broacast_elementwise(float* mul_1, const float* X, const float* Y) {
 
   int warp_id = threadIdx.x / 32;
   int lane_id = threadIdx.x % 32;
   int num_warps = blockDim.x / 32;
 
   #pragma unroll
   for (int bid = blockIdx.x; bid < 256; bid += gridDim.x) {
     #pragma unroll
     for (int wid = warp_id; wid < 16; wid += num_warps) {
       float reduce_res = 0;
 
       #pragma unroll
       for (int tid = lane_id; tid < 64; tid += 32) {
         // pre elementwise function here
         reduce_res += X[bid*16*64 + wid*64 + tid];
       } /* thread loop */
 
       /* intra warp reducetion */
       reduce_res += __shfl_xor_sync(0xffffffff, reduce_res, 16, 32);
       reduce_res += __shfl_xor_sync(0xffffffff, reduce_res,  8, 32);
       reduce_res += __shfl_xor_sync(0xffffffff, reduce_res,  4, 32);
       reduce_res += __shfl_xor_sync(0xffffffff, reduce_res,  2, 32);
       reduce_res += __shfl_xor_sync(0xffffffff, reduce_res,  1, 32);
 
       #pragma unroll
       for (int tid = lane_id; tid < 64; tid += 32) {
         int index = bid*16*64 + wid*64 + tid;
         // post elementwise function here
         mul_1[index] = reduce_res * Y[index];
       } /* thread loop */
     } /* warp loop*/
   } /* block loop */
 }

__global__ 	void fuse_elementwise_reduce_stop(int *g_idata, int *g_odata) {  
    extern 	shared 	int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  

    // pre elementwise function here
    sdata[tid] = g_idata[i];

    syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        syncthreads();
    }

    // write result for this block to global mem  
    if (tid == 0) 
        // post some function here
        g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_one_dim(int *g_idata, g_odata){
    return 0;
}

int main(){
    return 0;
}

