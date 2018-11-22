#!bin/bash
set -eu

source ./shell/env.sh
source ./shell/utils.sh

source ./model_conf

#init
core_num=100
data_dir=data
tmp_dir=tmp
file_list=filelist

HADOOP="hadoop fs -D hadoop.job.ugi=${ugi} -D fs.default.name=${hdfs_path}"

if [[ ${slurm_train_files_dir:-""} == "" ]];then
    mkdir -p $data_dir
    mkdir -p $tmp_dir
    #download train data
    $HADOOP -get ${train_files_dir} ./$data_dir/
    train_tar=`basename ${train_files_dir}`
    cd ./$data_dir; tar -xvf ${train_tar}; rm -rf ${train_tar}; cd -
fi


