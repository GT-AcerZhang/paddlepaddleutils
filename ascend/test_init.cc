#include <string>
#include <map>
#include <iostream>

#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "graph/types.h"

std::shared_ptr<ge::Session> session_;

std::map<ge::AscendString, ge::AscendString> _GetDefaultInitOptions() {
    ge::AscendString prof_opts=R"({"output":"/tmp/profiling","training_trace":"on","fp_point":"resnet_model/conv2d/Conv2Dresnet_model/batch_normalization/FusedBatchNormV3_Reduce","bp_point":"gradients/AddN_70"})";

	std::map<ge::AscendString, ge::AscendString> ge_options ;
    ge_options["ge.exec.deviceId"] = "0";
    ge_options["ge.graphRunMode"] = "1";
    ge_options["ge.exec.profilingMode"] = "1";
    ge_options["ge.exec.profilingOptions"] = prof_opts;

	return ge_options;
}

std::map<ge::AscendString, ge::AscendString> _GetDefaultInitSessionOptions() {
	std::map<ge::AscendString, ge::AscendString> init_options;
	return init_options;
}

void InitGlobalResouces() {
	std::cout << "Begin InitGlobalResouces";
	ge::Status status = ge::GEInitialize(_GetDefaultInitOptions());

    session_.reset(new ge::Session(_GetDefaultInitSessionOptions()));
	std::cout << "End InitGlobalResouces" << status << session_;
}

void DestroyGlobalResources() {
    std::cout << "Begin ascend DestroyGlobalResouces";
    session_ = nullptr;
    ge::GEFinalize();
    std::cout << "Begin ascend DestroyGlobalResouces";
}


int main(){
	InitGlobalResouces();	
    DestroyGlobalResources();
	return 0;
}
