import networks
from compile import compile
from evaluate import evaluate

mod, params, input_shape, output_shape = networks.get("resnet-18")
libname = compile(mod, params, "resnet-18", llvm=False, arm=False)
evaluate("0.0.0.0", 9090, libname, input_shape)
