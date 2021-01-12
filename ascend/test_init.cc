#include <string>
#include <map>
#include <iostream>

#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "graph/types.h"

std::map<std::string, std::string> GetDefaultInitOptions() {
	std::map<std::string, std::string> init_options;
	init_options["ge.exec.deviceId"] = std::to_string(0);
	init_options["ge.graphRunMode"] = std::to_string(1);
	return init_options;
}

std::map<std::string, std::string> GetDefaultInitSessionOptions() {
	std::map<std::string, std::string> init_options;
	init_options["a"] = "b";
	init_options["ge.trainFlag"] = "1";
	return init_options;
}

void InitGlobalResouces() {
	std::cout << "Begin InitGlobalResouces";
	ge::Status status = ge::GEInitialize(GetDefaultInitOptions());

	auto ss = new ge::Session(GetDefaultInitSessionOptions());
	std::cout << "End InitGlobalResouces" << status << &ss;
}

int main(){
	InitGlobalResouces();	
	return 0;
}
