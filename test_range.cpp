#include <iostream>
#include <string>
#include "Int.h"

int main() {
    std::cout << "Testing hex range validation...\n";
    
    // Test valid hex values
    Int start, end, max;
    
    // Test 1: Valid range
    start.SetBase16("1");
    end.SetBase16("1000");
    max.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
    
    std::cout << "Test 1 - Valid range:\n";
    std::cout << "Start: " << start.GetBase16() << "\n";
    std::cout << "End: " << end.GetBase16() << "\n";
    std::cout << "Max: " << max.GetBase16() << "\n";
    std::cout << "End >= Start: " << (end.IsGreaterOrEqual(&start) ? "true" : "false") << "\n";
    std::cout << "Start <= Max: " << (start.IsLowerOrEqual(&max) ? "true" : "false") << "\n";
    std::cout << "End <= Max: " << (end.IsLowerOrEqual(&max) ? "true" : "false") << "\n";
    
    // Test 2: Invalid range (start > end)
    start.SetBase16("1000");
    end.SetBase16("1");
    
    std::cout << "\nTest 2 - Invalid range (start > end):\n";
    std::cout << "Start: " << start.GetBase16() << "\n";
    std::cout << "End: " << end.GetBase16() << "\n";
    std::cout << "End >= Start: " << (end.IsGreaterOrEqual(&start) ? "true" : "false") << "\n";
    
    // Test 3: Values at boundary
    start.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
    end.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
    
    std::cout << "\nTest 3 - Boundary values:\n";
    std::cout << "Start: " << start.GetBase16() << "\n";
    std::cout << "End: " << end.GetBase16() << "\n";
    std::cout << "End >= Start: " << (end.IsGreaterOrEqual(&start) ? "true" : "false") << "\n";
    
    return 0;
}