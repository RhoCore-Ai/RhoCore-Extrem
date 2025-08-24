/*
 * This file is part of the Rhocore-extrem distribution.
 * Copyright (c) 2024 Thomas Baumann.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef KEYHUNTH
#define KEYHUNTH

#include "GPU/GPUEngine.h"
#include <vector>
#include <string>
// Include CUDA configuration
#include "GPU/CudaConfig.h"

// Search modes
#define SEARCH_COMPRESSED 0
#define SEARCH_UNCOMPRESSED 1
#define SEARCH_BOTH 2

// Address types
#define P2PKH 0

// CPU Group Size (must match GPU GRP_SIZE)
#define CPU_GRP_SIZE 1024

// Thread parameters structure
typedef struct {
  int  threadId;
  int  gridSizeX;
  int  gridSizeY;
  int  gpuId;
  bool isAlive;
  bool hasStarted;
  bool completed;
  uint64_t startKey;
  uint64_t endKey;
  uint64_t keysPerThread;
  uint64_t keysToSearch;
  uint64_t keysSearched;
  uint64_t keysFound;
  uint32_t collisionOffset;
  uint32_t collisionSize;
  uint32_t collisionSize2;
  uint32_t collisionSize3;
  uint32_t collisionSize4;
  uint32_t collisionSize5;
  uint32_t collisionSize6;
  uint32_t collisionSize7;
  uint32_t collisionSize8;
  uint32_t collisionSize9;
  uint32_t collisionSize10;
  uint32_t collisionSize11;
  uint32_t collisionSize12;
  uint32_t collisionSize13;
  uint32_t collisionSize14;
  uint32_t collisionSize15;
  uint32_t collisionSize16;
  uint32_t collisionSize17;
  uint32_t collisionSize18;
  uint32_t collisionSize19;
  uint32_t collisionSize20;
} TH_PARAM;

class KeyHunt {

public:
	KeyHunt(std::string addressFile, std::vector<unsigned char> addressHash, int searchMode, bool useGpu,
		std::string outputFile, bool useSSE, uint32_t maxFound, std::string rangeStart, std::string rangeEnd, bool& should_exit);
	~KeyHunt();
	void Search(int nbThread, std::vector<int>& gpuId, std::vector<int>& gridSize, bool& should_exit);
	void FindKeyGPU(TH_PARAM* p);
	void FindKeyCPU(TH_PARAM* p);

private:
	void CheckAddresses(bool compressed, Int key, Point& pubkey);
	void CheckAddressesSSE(bool compressed, Int key, Point& pubkey);
	bool CheckPublicAddress(bool compressed, std::string address);
	void output(std::string addr, std::string pAddr, std::string pvcKey);
	bool isInsideRange(Int& key);
	bool MatchHash160(uint32_t* _h);
	std::string formatThousands(uint64_t x);
	char* toTimeStr(int sec, char* timeStr);

	Int rangeStart;
	Int rangeEnd;
	Int rangeSize;
	uint64_t counters[256];
	uint64_t counters2[256];
	bool* BloomTable;
	uint64_t BLOOM_SIZE;
	uint64_t BLOOM_BITS;
	uint8_t BLOOM_HASHES;
	uint8_t* DATA;
	uint64_t TOTAL_ADDR;
	std::string addressFile;
	std::vector<unsigned char> addressHash;
	int searchMode;
	bool useGpu;
	std::string outputFile;
	FILE* rKey;
	bool useSSE;
	bool* endOfSearch;
	uint32_t maxFound;
	int searchType;
	int nbGPUThread;
	bool& should_exit;

};

#endif // KEYHUNTH