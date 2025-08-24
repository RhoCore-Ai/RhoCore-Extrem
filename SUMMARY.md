# Rhocore-extrem für Ubuntu 22.04 mit CUDA 12.x und 8x RTX 4090

## Projektübersicht

Dies ist eine modifizierte Version von KeyHunt-Cuda, die für die Verwendung auf vast.ai mit 8x RTX 4090-GPUs optimiert wurde und nun unter dem Namen "Rhocore-extrem" läuft.

## Wichtige Änderungen

1. **CUDA-Kompatibilität**: Aktualisiert für CUDA 12.x auf Ubuntu 22.04
2. **RTX 4090-Optimierung**: Spezielle Optimierungen für die Architektur der RTX 4090
3. **Multi-GPU-Unterstützung**: Verbesserte Unterstützung für bis zu 8 GPUs
4. **Performance-Optimierungen**: Angepasste Grid- und Blockgrößen für maximale Performance
5. **Namensänderung**: Projekt umbenannt in "Rhocore-extrem"

## Dateiübersicht

### Geänderte Dateien:
- `Makefile` - Aktualisiert für CUDA 12.x und Ubuntu 22.04, Executable-Name geändert
- `GPU/GPUEngine.cu` - Optimiert für RTX 4090
- `GPU/GPUEngine.h` - Aktualisiert für neue Konfigurationen
- `GPU/GPUCompute.h` - CUDA 12.x-Kompatibilität
- `GPU/GPUMath.h` - CUDA 12.x-Kompatibilität
- `Main.cpp` - Verbesserte GPU-Unterstützung
- `KeyHunt.h` - Neue CUDA-Konfiguration eingebunden
- `README.md` - Aktualisiert für neue Verwendung und neuen Namen
- `VASTAI_SETUP.md` - Neue Anleitung für vast.ai
- `vastai_examples.txt` - Aktualisiert mit neuem Executable-Namen

### Neue Dateien:
- `GPU/CudaConfig.h` - CUDA-Konfigurationsheader
- `GPU/CudaConfig.cpp` - CUDA-Konfigurationsimplementierung

## Kompilierung

### Für CPU-only (ohne CUDA):
```bash
make all
```

### Für GPU mit RTX 4090-Optimierung:
```bash
make gpu-4090
```

oder

```bash
make gpu=1 CCAP=89 all
```

### Für Debug-Version:
```bash
make gpu-debug
```

## Installation der Abhängigkeiten

Auf Ubuntu 22.04:
```bash
make install-deps
```

## Verwendung auf vast.ai

1. Erstelle eine Instanz mit:
   - Ubuntu 22.04
   - CUDA 12.x
   - 8x RTX 4090 GPUs

2. Installiere Abhängigkeiten:
   ```bash
   make install-deps
   ```

3. Kompiliere das Projekt:
   ```bash
   make gpu-4090
   ```

4. Führe das Programm aus:
   ```bash
   ./Rhocore-extrem -g -i 0,1,2,3,4,5,6,7 -s 1 -e 10000000000 -a 1YourTargetAddressHere
   ```

## Erwartete Performance

Mit 8x RTX 4090-GPUs:
- Über 3000 Millionen Schlüssel pro Sekunde (Mk/s) im komprimierten Modus
- Bis zu 1500 Millionen Schlüssel pro Sekunde (Mk/s) im unkomprimierten Modus

## Fehlerbehebung

### CUDA-Fehler
- Überprüfe die CUDA-Installation: `make check-cuda`
- Liste verfügbare GPUs: `./Rhocore-extrem -l`

### Performance-Probleme
- Versuche verschiedene Grid-Größen
- Überprüfe GPU-Temperaturen: `nvidia-smi`

## Lizenz

GPLv3 - Siehe LICENSE-Datei für Details