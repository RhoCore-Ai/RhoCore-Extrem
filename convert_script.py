#!/usr/bin/env python3
"""
Simple script to convert Bitcoin addresses to hash160.bin
"""

import base58
import sys
import gzip

def bitcoin_address_to_hash160(address):
    """Convert a Bitcoin address to its hash160 representation"""
    try:
        # Decode the base58 address
        decoded = base58.b58decode(address.strip())
        # Remove the version byte (first) and checksum (last 4 bytes)
        hash160 = decoded[1:-4]
        return hash160
    except Exception as e:
        return None

def convert_file(input_filename, output_filename, max_lines=None):
    """Convert addresses file to hash160.bin"""
    count = 0
    valid_count = 0
    
    print(f"Converting {input_filename} to {output_filename}...")
    
    # Handle gzipped file
    if input_filename.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(input_filename, mode, encoding='utf-8', errors='ignore') as f:
        with open(output_filename, 'wb') as out_f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                count += 1
                hash160 = bitcoin_address_to_hash160(line)
                if hash160 and len(hash160) == 20:
                    out_f.write(hash160)
                    valid_count += 1
                    
                    if valid_count % 100000 == 0:
                        print(f"Converted {valid_count} addresses...")
                
                if max_lines and count >= max_lines:
                    break
    
    print(f"Conversion complete!")
    print(f"Processed {count} lines, successfully converted {valid_count} valid addresses")
    print(f"File size: {valid_count * 20} bytes")

if __name__ == "__main__":
    # Convert first 100000 addresses as a test
    convert_file("Bitcoin_addresses_LATEST.txt.gz", "hash160.bin", max_lines=100000)
    print("\nFor the full file, run:")
    print("python3 convert_script.py")