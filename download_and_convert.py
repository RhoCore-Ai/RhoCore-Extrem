#!/usr/bin/env python3
"""
Script to download and convert Bitcoin addresses to hash160.bin format
"""

import base58
import hashlib
import gzip
import requests
from urllib.parse import urlparse
import os

def bitcoin_address_to_hash160(address):
    """Convert a Bitcoin address to its hash160 representation"""
    try:
        # Decode the base58 address
        decoded = base58.b58decode(address.strip())
        # Remove the version byte (first) and checksum (last 4 bytes)
        hash160 = decoded[1:-4]
        return hash160
    except Exception as e:
        print(f"Error decoding address {address}: {e}")
        return None

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

def convert_addresses_to_hash160(input_file, output_file, max_addresses=None):
    """Convert Bitcoin addresses to hash160.bin format"""
    count = 0
    valid_count = 0
    
    print(f"Converting addresses from {input_file} to {output_file}...")
    
    # Handle gzipped file
    if input_file.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(input_file, mode, encoding='utf-8', errors='ignore') as f:
        with open(output_file, 'wb') as out_f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                count += 1
                hash160 = bitcoin_address_to_hash160(line)
                if hash160 and len(hash160) == 20:
                    out_f.write(hash160)
                    valid_count += 1
                    
                    if valid_count % 10000 == 0:
                        print(f"Processed {count} addresses, converted {valid_count} valid addresses...")
                
                if max_addresses and valid_count >= max_addresses:
                    break
    
    print(f"Conversion complete!")
    print(f"Processed {count} lines, successfully converted {valid_count} valid addresses")
    print(f"Output file: {output_file} ({valid_count * 20} bytes)")

def main():
    # URL of the Bitcoin addresses file
    url = "http://addresses.loyce.club/Bitcoin_addresses_LATEST.txt.gz"
    gz_filename = "Bitcoin_addresses_LATEST.txt.gz"
    txt_filename = "Bitcoin_addresses_LATEST.txt"
    hash160_filename = "hash160.bin"
    
    # Download the file
    if not os.path.exists(gz_filename):
        download_file(url, gz_filename)
    else:
        print(f"{gz_filename} already exists, skipping download")
    
    # Convert to hash160.bin (using first 100000 addresses for demo)
    convert_addresses_to_hash160(gz_filename, hash160_filename, max_addresses=100000)
    
    print("\nYou can now use the hash160.bin file with RhoCore-Extrem:")
    print("./Rhocore-extrem -f hash160.bin -g")
    print("(Remove the max_addresses limit in the script for the full file)")

if __name__ == "__main__":
    main()