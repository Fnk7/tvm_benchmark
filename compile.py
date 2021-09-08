import tvm
from tvm import relay

def compile(mod, params, name="net", llvm=False, arm=True, sve=False):
    # llvm doesn't support sve
    if llvm:
        target = "llvm"
    else:
        target = "c"
    if arm:
        target += " -device=arm_cpu"
        if llvm:
            target += " -mtriple=aarch64-linux-gnu -mattr=+neon"
        elif sve:
            target += " -march=armv8-a+sve"
        else:
            target += " -march=armv8-a+neon"
    target = tvm.target.Target(target)
    print("Target is {}".format(target))
    # relay compile
    if llvm:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            lib = relay.build(mod, target, params=params)
    name = "./export_lib/" + name
    name += "_llvm" if llvm else "_c"
    name += "_arm" if arm else "_x64"
    if arm:
        name += "_sve" if sve else "_neon"
    name += ".so"
    # export library
    cc = "g++"
    options = ["-O3"]
    c_target_triple="x86_64-linux-gnu"
    if arm:
        cc = "aarch64-linux-gnu-g++"
        if sve:
            options.append("-march=armv8-a+sve")
        c_target_triple="aarch64-linux-gnu"
    else:
        options.append("-march=skylake")
        options.append("-ftree-parallelize-loops=8")
        options.append("-floop-parallelize-all")

    if llvm:
        lib.export_library(name, workspace_dir="./tmp", cc=cc, options=options)
    else:
        lib.export_library(name, workspace_dir="./tmp", c_target_triple=c_target_triple, cc=cc, options=options)
    return name


if __name__ == "__main__":
    import networks
    network = "lstm"

    mod, params, _, _ = networks.get(network)

    arm = True
    llvm = False
    for sve in [True, False]:
        print(compile(mod, params, network, llvm, arm, sve))

    arm = True
    sve = False
    llvm = True
    print(compile(mod, params, network, llvm, arm, sve))

    arm = False
    sve = False
    llvm = False
    for llvm in [True, False]:
        print(compile(mod, params, network, llvm, arm, sve))
