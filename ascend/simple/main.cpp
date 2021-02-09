#include <ge/ge_api.h>
#include <graph/graph.h>
#include <graph/attr_value.h>
#include <graph/operator_factory.h>
#include <all_ops.h>
#include <vector>
#include <iostream>

#define GECHECK(cmd) do {                         \
  Status e = cmd;                              \
  if( e != SUCCESS) {                          \
    printf("Failed: GE error %s:%d '%d'\n",             \
        __FILE__,__LINE__,e);   \
    exit(FAILED);                             \
  }                                                 \
} while(0)

using namespace ge;
using namespace std;

int main(int argc, char* argv[]) {
    auto shape_data = vector<int64_t>({8});
    TensorDesc desc_data(Shape(shape_data), FORMAT_ND, DT_INT8);
    
    vector<Operator> inputs;
    vector<Operator> outputs;
    for (auto i = 0; i < 100; i ++) {
	auto data = op::Data("data_" + std::to_string(i));
	data.update_input_desc_x(desc_data);
	data.update_output_desc_y(desc_data);

        auto allreduce = op::HcomAllReduce("all_reduce_" + std::to_string(i));
        allreduce.set_input_x(data).set_attr_reduction("sum").set_attr_group("hccl_world_group");
	inputs.push_back(data);
        outputs.push_back(allreduce);
    }

    auto graph = Graph("TestAllreduce");
    graph.SetInputs(inputs).SetOutputs(outputs);

    AscendString rank {getenv("RANK_ID")};
    AscendString rankTableFile {getenv("RANK_TABLE_FILE")};

    ge::AscendString prof_opts=R"({"output":"/home/gongwb/go/prof.log","training_trace":"on"})";

    std::map<ge::AscendString, ge::AscendString> ge_options ;
    ge_options["ge.exec.deviceId"] = "0";
    ge_options["ge.graphRunMode"] = "1";

    map<AscendString, AscendString> config = {
        {"ge.exec.deviceId", rank},
        {"ge.graphRunMode", "1"},
        {"ge.exec.rankTableFile", rankTableFile},
        {"ge.exec.rankId", rank},
	{"ge.exec.isUseHcom", "1"},
	{"ge.exec.deployMode", "0"}
    	{"ge.exec.profilingMode","1"},
    	{"ge.exec.profilingOptions", prof_opts}
    };

    GECHECK(ge::GEInitialize(config));

    map<AscendString, AscendString> options;
    Session *session = new Session(options);
    if (session == nullptr) {
        cout << "create session failed." << endl;
        return -1;
    }

    GECHECK(session->AddGraph(0, graph));

    vector<Tensor> input;
    for (int i = 0; i < 100; i++) {
	    Tensor input_tensor;
	    int size = 8;
	    uint8_t input_data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
	    input_tensor.SetTensorDesc(desc_data);
	    input_tensor.SetData(input_data, size);
	    input.push_back(input_tensor);
    }
    vector<Tensor> output;

    GECHECK(session->RunGraph(0, input, output));

    for (auto tensor : output) {
        cout << "Tensor size: " << tensor.GetSize() << std::endl;
        cout << "Tensor data: " << endl;
        const uint8_t* output_data = tensor.GetData();
        for (size_t idx = 0; idx < tensor.GetSize(); ++idx) {
	  printf("%u,", output_data[idx]);
        }
        cout << endl;
    }

    delete session;
    GECHECK(GEFinalize());

    return 0;
}
