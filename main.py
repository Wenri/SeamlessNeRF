def config_parser(cmd=None):
    import configargparse
    import train
    import merge

    parser = configargparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')
    train.config_parser(subparsers.add_parser('train'))
    merge.config_parser(subparsers.add_parser('merge'))

    return parser.parse_args(cmd)


def command(args):
    print(args)
    match args.command:
        case 'train':
            import train
            func = train.main
        case 'merge':
            import merge
            func = merge.main
        case _:
            raise ValueError(f'Unknown command: {args.command}')
    return func(args)


# A function to set up the running environment for the training
def setup_environment(cudaMallocAsync=True):
    import os
    from contextlib import suppress

    if cudaMallocAsync:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'

    import numpy as np
    import torch

    with suppress(ImportError):
        from torch.backends import cuda, cudnn
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        cudnn.allow_tf32 = True

    torch.set_default_dtype(torch.float32)

    # Set the seed for generating random numbers.
    np.random.seed(np.bitwise_xor(*np.atleast_1d(np.asarray(torch.seed(), dtype=np.uint64)).view(np.uint32)).item())

    import pyximport
    pyximport.install()

    return command


if __name__ == "__main__":
    setup_environment()(config_parser())
