#!/usr/bin/env python3
"""
Script to create a hash160.bin file for RhoCore-Extrem
"""

import hashlib
import binascii

import base58

def bitcoin_address_to_hash160(address):
    """Convert a Bitcoin address to its hash160 representation"""
    try:
        # Decode the base58 address
        decoded = base58.b58decode(address)
        # Remove the version byte (first) and checksum (last 4 bytes)
        hash160 = decoded[1:-4]
        return hash160
    except Exception as e:
        print(f"Error decoding address {address}: {e}")
        return None

def create_hash160_file(filename, addresses):
    """Create a hash160.bin file with Bitcoin addresses"""
    count = 0
    with open(filename, 'wb') as f:
        for address in addresses:
            hash160 = bitcoin_address_to_hash160(address)
            if hash160 and len(hash160) == 20:
                f.write(hash160)
                count += 1
                print(f"Added {address} -> {hash160.hex()}")
            else:
                print(f"Skipping invalid address: {address}")

    print(f"\nCreated {filename} with {count} hash160 entries")

# Example usage
if __name__ == "__main__":
    sample_addresses = [
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Satoshi's address
        "1BitcoinEaterAddressDontSendf59kuE",  # Bitcoin eater
        "1LoveR8itK83b36kUQb8p9ed8RJXNFxRh6b", # Love Bitcoin
        "1CounterpartyXXXXXXXXXXXXXXXUWLpVr",  # Counterparty burn address
    ]

    print("Creating hash160.bin file...")
    create_hash160_file("hash160.bin", sample_addresses)
    print("\nYou can now use it with: ./Rhocore-extrem -f hash160.bin")