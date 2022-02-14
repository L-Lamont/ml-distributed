# distributed
A collation of distributed ml examples

## Current Frameworks
- TensorFlow
    - MirroredStrategy
    - MultiWorkerMirroredStrategy
    - Horovod
- PyTorch
    - Horovod
- PyTorch Lightning
    - Horovod

## Issues
- mpirun not setting required environment variables in newer versions
- MultiWorkerMirroredStrategy not working with single input files
    - They assume you have multiple input files to justify multiple nodes
- Pytorch lightning version 1.5.10 doesn't pickup horovod
    - Pytorch lightning 1.5.9 works
- PyTorch complains at TMP being on a network filesystem
    - Doesn't stop training just gives error
