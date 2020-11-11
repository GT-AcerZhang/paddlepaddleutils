#include "fusion_api.h"

__global__ void fusion32(float* mul_1, const float* X, const float* Y) {
 
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
         mul_1[index] = reduce_res + Y[index];
       } /* thread loop */
     } /* warp loop*/
   } /* block loop */
 }


