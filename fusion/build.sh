rm -f *.o

NVCC=/usr/local/cuda/bin/nvcc
gen_code="-gencode arch=compute_75,code=sm_75"
files=("test_p2p.cu")
for file in ${files[@]}; do
    file_name=$(basename "${file}")
    name="${file_name%.*}"
    echo $file_name $name
    ${NVCC} -ccbin g++  -m64    --std=c++11  ${gen_code} -o ${name} ${file}
done
    
#${NVCC} -ccbin g++  -m64    --std=c++11  ${gen_code} -o fusion_test *.o
