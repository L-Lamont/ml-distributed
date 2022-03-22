# distributed
A collation of distributed ml examples

## Current Frameworks
- TensorFlow
    - MirroredStrategy
    - MultiWorkerMirroredStrategy
    - Horovod
- PyTorch
    - Horovod
    - DDP
- PyTorch Lightning
    - Horovod

## Issues
- MultiWorkerMirroredStrategy getting CUDA\_ERROR\_OUT\_OF\_MEMORY error
    - Needed to allow dynamic allocation of memory on GPUs
- MultiWorkerMirroredStrategy wouldn't work using MNIST dataset
    - Needed to convert dataset to a tensorflow dataset and update the AUTO\_SHARD\_POLICY to DATA
- Pytorch lightning version 1.5.10 doesn't pickup horovod
    - Pytorch lightning 1.5.9 works
- PyTorch complains at TMP being on a network filesystem
    - Doesn't stop training just gives error
