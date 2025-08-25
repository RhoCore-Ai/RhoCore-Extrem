# Rhocore-extrem
_Hunt for Bitcoin private keys with extreme performance._

## Author
Thomas Baumann

## Description
Rhocore-extrem is a high-performance Bitcoin private key hunter optimized for NVIDIA RTX 4090 GPUs. This project is designed for use with vast.ai and can efficiently search for Bitcoin private keys across large ranges.

## Features
- Optimized for NVIDIA RTX 4090 GPUs
- Support for up to 8 GPUs
- Compressed and uncompressed address search
- File mode and single address mode
- High performance with over 3000 million keys per second (Mk/s) on 8x RTX 4090

## Changes from Original
- Removed endomorphism and curve symmetry
- Optimized for RTX 4090 with Compute Capability 8.9
- Increased STEP_SIZE to 2048 for better performance
- Grid size calculation optimized for RTX 4090 (16 * MP count)
- Increased default maxFound to 256K for better batching
- Updated for Ubuntu 22.04 with CUDA 12.x

## Performance
With 8x RTX 4090 GPUs:
- Over 3000 million keys per second (Mk/s) in compressed mode
- Up to 1500 million keys per second (Mk/s) in uncompressed mode

## Requirements
- Ubuntu 22.04
- CUDA 12.x
- NVIDIA Driver 535+
- 8x NVIDIA RTX 4090 GPUs (for maximum performance)

## Installation
```bash
# Install dependencies
make install-deps

# Compile for RTX 4090
make gpu-4090
```

## Usage
CPU and GPU can not be used together, because right now the program divides the whole input range into equal parts for all the threads, so use either CPU or GPU so that the whole range can increment by all the threads with consistency.

Minimum address should be more than 1000.

```
./Rhocore-extrem -h
Usage: Rhocore-extrem [options...]
Options:
    -v, --version          Print version
    -c, --check            Check the working of the codes
    -u, --uncomp           Search uncompressed addresses
    -b, --both             Search both uncompressed or compressed addresses
    -g, --gpu              Enable GPU calculation
    -i, --gpui             GPU ids: 0,1...: List of GPU(s) to use, default is 0
    -x, --gpux             GPU gridsize: g0x,g0y,g1y,g1y, ...: Specify GPU(s) kernel gridsize, default is 16*(Device MP count),128
    -o, --out              Outputfile: Output results to the specified file, default: Found.txt
    -m, --max              Specify maximun number of addresses found by each kernel call
    -t, --thread           threadNumber: Specify number of CPU thread, default is number of core
    -l, --list             List cuda enabled devices
    -f, --file             Ripemd160 binary hash file path
    -a, --addr             P2PKH Address (single address mode)
    -s, --start            Range start in hex
    -e, --end              Range end in hex, if not provided then endRange is set to startRange + STEP_SIZE, or to the maximum value for secp256k1 curve if range exceeds curve order
```

## Examples
```bash
# Single Address Mode with 8 GPUs:
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -s 1 -a 1YourTargetAddressHere

# File Mode with 8 GPUs:
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -s 1 -f path/to/hashes.bin

# With specific range:
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -s 80000000 -e 10000000000 -a 1YourTargetAddressHere
```

## Compilation Options
```bash
# For RTX 4090 optimization:
make gpu-4090

# For debugging:
make gpu-debug

# For CPU-only:
make all

# Install dependencies:
make install-deps
```

## License
GPLv3

This is an experimental project and right now going through a lot of changes, so bugs and errors can appear.