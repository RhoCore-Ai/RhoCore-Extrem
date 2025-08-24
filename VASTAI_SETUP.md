# Vast.ai Setup Guide for Rhocore-extrem

This guide explains how to set up and run Rhocore-extrem on vast.ai with 8x RTX 4090 GPUs.

## Template Selection

When creating an instance on vast.ai, use the following template:
- Base OS: Ubuntu 22.04
- GPU: 8x RTX 4090
- CUDA: 12.x

## Initial Setup

1. After connecting to your instance, update the system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. Install required dependencies:
   ```bash
   make install-deps
   ```

## Compilation

Compile the project for RTX 4090 GPUs:
```bash
make gpu-4090
```

This is equivalent to:
```bash
make gpu=1 CCAP=89 all
```

## Running the Software

### Single Address Mode
To search for a specific address using all 8 GPUs:
```bash
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -x 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 -s 1 -e 10000000000 -a 1YourTargetAddressHere
```

Or with automatic grid size calculation:
```bash
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -s 1 -e 10000000000 -a 1YourTargetAddressHere
```

### File Mode
To search using a hash file:
```bash
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -s 1 -e 10000000000 -f path/to/hashes.bin
```

## Performance Optimization

### Recommended Grid Sizes for RTX 4090
- For maximum speed: 256x256 or 512x128
- For memory constrained environments: 128x128

### Example with Custom Grid Sizes
```bash
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -x 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 -s 1 -e 10000000000 -a 1YourTargetAddressHere
```

## Monitoring GPU Usage

To monitor GPU utilization during execution:
```bash
nvidia-smi -l 1
```

## Expected Performance

With 8x RTX 4090 GPUs, you should expect:
- Over 3000 million keys per second (Mk/s) in compressed mode
- Up to 1500 million keys per second (Mk/s) in uncompressed mode

## Troubleshooting

### CUDA Errors
If you encounter CUDA errors:
1. Check that all GPUs are properly detected:
   ```bash
   ./Rhocore-extrem -l
   ```
2. Ensure you're not exceeding GPU memory limits

### Performance Issues
If performance is lower than expected:
1. Try different grid sizes
2. Check GPU temperatures:
   ```bash
   nvidia-smi
   ```
3. Ensure adequate power supply for all 8 GPUs

## Stopping the Program

To stop the program gracefully, press Ctrl+C. The program will display a summary of the work done before exiting.