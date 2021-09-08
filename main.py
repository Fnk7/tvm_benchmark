import networks
from compile import compile
from evaluate import evaluate

net_name = "lstm"
mod, params, input_shape, output_shape = networks.get(net_name)
libname = compile(mod, params, net_name, llvm=True, arm=False)
evaluate("0.0.0.0", 9090, libname, input_shape)
