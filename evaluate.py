import numpy as np
from tvm import nd
from tvm import rpc
from tvm.contrib import graph_executor as runtime

def evaluate(host, port, libname, input_shape):
    remote = rpc.connect(host, port)
    print("Remote connected")
    remote.upload(libname)
    lib = remote.load_module(libname[len("./export_lib/"):])
    dev = remote.cpu()
    data_tvm = nd.array((np.random.uniform(size=input_shape)).astype("float32"))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input("data", data_tvm)
    print("Starting evaluate")
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print(prof_res)
    print("Mean inference time (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res)))


if __name__ == "__main__":
    import networks
    mod, params, input_shape, output_shape = networks.get("lstm")
    evaluate("0.0.0.0", 9090, "./export_lib/lstm_c_x64.so", input_shape)

