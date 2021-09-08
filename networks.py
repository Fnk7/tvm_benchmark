from tvm import relay
from tvm.relay import testing

def get(name, **kwargs):
    """Get the symbol definition and random weight of a network"""
    batch_size = 1 if "batch_size" not in kwargs else kwargs["batch_size"]
    dtype = "float32" if "dtype" not in kwargs else kwargs["dtype"]
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        num_layer = 18 if "num_layer" not in kwargs else kwargs["num_layer"]
        mod, params = relay.testing.resnet.get_workload(num_layers=num_layer, batch_size=batch_size, dtype=dtype)
    elif "lstm" in name:
        import lstm
        return lstm.get_workload(**kwargs)
    elif "vgg" in name: 
        num_layer = 11 if "num_layer" not in kwargs else kwargs["num_layer"]
        mod, params = relay.testing.vgg.get_workload(num_layers=num_layer, batch_size=batch_size, dtype=dtype)
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version="1.1", dtype=dtype)
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    else:
        raise ValueError("Unsupported network: " + name)
    return mod, params, input_shape, output_shape


if __name__ == "__main__":
    mod, params, input_shape, output_shape = get("lstm")
    # print("Params:")
    # print(params)
    # print("Module:")
    # print(mod)
    print("Input shape:")
    print(input_shape)
    print("Output shape:")
    print(output_shape)

