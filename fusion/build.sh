rm *.o

NVCC=/usr/local/cuda/bin/nvcc
gen_code="-gencode arch=compute_75,code=sm_75"
for file in *.cu ; do
    echo ${file}
    ${NVCC} -ccbin g++  -m64    --std=c++11  ${gen_code} -o ${file}.o -c ${file}
done
    
${NVCC} -ccbin g++  -m64    --std=c++11  ${gen_code} -o fusion_test *.o
