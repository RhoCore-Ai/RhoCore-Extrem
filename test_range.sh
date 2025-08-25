#!/bin/bash

# Compile the test program
echo "Compiling test_range.cpp..."
g++ -m64 -mssse3 -Wno-write-strings -O3 -I. -lpthread -o test_range test_range.cpp Int.cpp IntMod.cpp >/dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running test..."
    ./test_range
else
    echo "Compilation failed!"
    echo "Trying with debug info..."
    g++ -m64 -mssse3 -Wno-write-strings -g -I. -lpthread -o test_range test_range.cpp Int.cpp IntMod.cpp
fi