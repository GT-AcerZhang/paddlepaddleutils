# paddle v2
################################## User Define Configuration ###########################
################################## Data Configuration ##################################
#type of storage cluster
storage_type="hdfs"
#attention: files for training should be put on hdfs
##the list contains all file locations should be specified here
fs_name="hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310"
##If force_reuse_output_path is True ,paddle will remove output_path without check output_path exist
force_reuse_output_path="True"
##ugi of hdfs
#fs_ugi=idl-dl,idl-dl@admin
fs_ugi="bml,bml#user"

#the initial model path on hdfs used to init parameters
#init_model_path=

#the initial model path for pservers
#pserver_model_dir=

#which pass
#pserver_model_pass=
FLAGS_rpc_deadline=3000000
#GLOG_v=1
#GLOG_logtostderr=1

##train data path on hdfs
#train_data_path="/app/inf/mpi/bml-guest/paddlepaddle/public/transformer/en-fr-train"
train_data_path="/app/inf/mpi/bml-guest/paddlepaddle/public/transformer/en-fr-train-test-data/train" 

##test data path on hdfs, can be null or not setted
#test_data_path="/app/inf/mpi/bml-guest/paddlepaddle/public/transformer/wmt14_enfr_test"

#add datareader to thirdparty
#thirdparty_path = "/app/inf/mpi/bml-guest/paddlepaddle/public/transformer/en-fr-thirdparty"
thirdparty_path = "/app/inf/mpi/bml-guest/paddlepaddle/public/transformer/en-fr-train-test-data/thirdparty"
#thirdparty_path="/app/inf/mpi/bml-guest/paddlepaddle/public/transformer/wmt14_enfr_test"

#the output directory on hdfs
output_path="/app/inf/mpi/bml-guest/paddlepaddle/gongweibao/transformer/en-fr/output"
