# Rhocore-extrem - Projektübersicht

## Projektname
Rhocore-extrem (ehemals KeyHunt-Cuda-2)

## Durchgeführte Änderungen

### 1. Projektnamen aktualisiert
- Alle Vorkommen von "KeyHunt-Cuda-2" wurden durch "Rhocore-extrem" ersetzt
- In Dateien: Main.cpp, README.md, VASTAI_SETUP.md, vastai_examples.txt, SUMMARY.md

### 2. Executable-Name geändert
- In der Makefile wurde der Zielname von "KeyHunt" zu "Rhocore-extrem" geändert

### 3. Dokumentation aktualisiert
- README.md: Vollständig überarbeitet für das neue Projekt
- VASTAI_SETUP.md: Anweisungen für Rhocore-extrem aktualisiert
- vastai_examples.txt: Beispiele mit neuem Executable-Namen
- SUMMARY.md: Projektzusammenfassung mit neuen Informationen

### 4. Neue Dateien erstellt
- GPU/CudaConfig.h: CUDA-Konfigurationsheader
- GPU/CudaConfig.cpp: CUDA-Konfigurationsimplementierung

### 5. Optimierungen für RTX 4090
- GPUEngine.cu/h: Für RTX 4090-Leistungsoptimierung aktualisiert
- GPUCompute.h: CUDA 12.x-Kompatibilität
- GPUMath.h: CUDA 12.x-Kompatibilität

## Verzeichnisstruktur
Das Projekt behält die ursprüngliche Struktur bei, fügt aber neue Dateien hinzu:
```
Rhocore-extrem/
├── ArgParse.h
├── Base58.cpp
├── Base58.h
├── Bech32.cpp
├── Bech32.h
├── Bloom.cpp
├── Bloom.h
├── GPU/
│   ├── CudaConfig.cpp     (NEU)
│   ├── CudaConfig.h       (NEU)
│   ├── GPUBase58.h
│   ├── GPUCompute.h       (AKTUALISIERT)
│   ├── GPUEngine.cu       (AKTUALISIERT)
│   ├── GPUEngine.h        (AKTUALISIERT)
│   ├── GPUGenerate.cpp
│   ├── GPUGroup.h
│   ├── GPUHash.h
│   └── GPUMath.h         (AKTUALISIERT)
├── hash/
│   ├── ripemd160.cpp
│   ├── ripemd160.h
│   ├── ripemd160_sse.cpp
│   ├── sha256.cpp
│   ├── sha256.h
│   ├── sha256_sse.cpp
│   ├── sha512.cpp
│   └── sha512.h
├── Int.cpp
├── IntGroup.cpp
├── KeyHunt.cpp
├── KeyHunt.h
├── Main.cpp              (AKTUALISIERT)
├── Makefile              (AKTUALISIERT)
├── Point.cpp
├── Random.cpp
├── README.md             (AKTUALISIERT)
├── SECP256K1.cpp
├── SUMMARY.md            (AKTUALISIERT)
├── Timer.cpp
├── VASTAI_SETUP.md       (AKTUALISIERT)
└── vastai_examples.txt   (AKTUALISIERT)
```

## Kompilierung
```bash
# Für GPU-Optimierung mit RTX 4090:
make gpu-4090

# Für Debug-Version:
make gpu-debug

# Um Abhängigkeiten zu installieren:
make install-deps
```

## Ausführung
```bash
# Single Address Mode mit 8 GPUs:
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -s 1 -e 10000000000 -a 1YourTargetAddressHere

# File Mode mit 8 GPUs:
./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -s 1 -e 10000000000 -f path/to/hashes.bin
```

## Erwartete Leistung
Mit 8x RTX 4090-GPUs:
- Über 3000 Millionen Schlüssel pro Sekunde (Mk/s) im komprimierten Modus
- Bis zu 1500 Millionen Schlüssel pro Sekunde (Mk/s) im unkomprimierten Modus

## Lizenz
GPLv3