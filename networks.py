from tvm import relay
from tvm.relay import testing

def get(name, batch_size=1, dtype="float32"):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "lstm" in name:
        input_shape = (batch_size, 32, 128)
        mod, params = relay.testing.lstm.get_workload(iterations=2, num_hidden=128, batch_size=batch_size, dtype="float")
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
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
    mod, params, input_shape, output_shape = get("resnet-18")
    print("Params:")
    print(params)
    print("Module:")
    print(mod)
    print("Input shape:")
    print(input_shape)
    print("Output shape:")
    print(output_shape)

