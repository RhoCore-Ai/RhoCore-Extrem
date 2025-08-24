#---------------------------------------------------------------------
# Makefile for Rhocore-extrem
#
# Author : Thomas Baumann

SRC = Base58.cpp IntGroup.cpp Main.cpp Bloom.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      KeyHunt.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp Bech32.cpp GPU/CudaConfig.cpp

OBJDIR = obj

ifdef gpu

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o Main.o Bloom.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o KeyHunt.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o \
        GPU/GPUEngine.o Bech32.o GPU/CudaConfig.o)

else

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o Main.o Bloom.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o KeyHunt.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o Bech32.o GPU/CudaConfig.o)

endif

CXX        = g++
# Für Ubuntu 22.04 mit CUDA 12.x
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc
# nvcc requires joint notation w/o dot, i.e. "5.2" -> "52"
ccap       = $(shell echo $(CCAP) | tr -d '.')

# Standardmäßig für RTX 4090 (Compute Capability 8.9)
# Wenn keine CCAP angegeben ist, verwenden wir 89 für RTX 4090
CCAP_DEFAULT = 89
CCAP_USED    = $(if $(ccap),$(ccap),$(CCAP_DEFAULT))

# Für Ubuntu 22.04 mit CUDA 12.x
CUDA_LIBS = -L$(CUDA)/lib64 -lcudart -lcuda

ifdef gpu
ifdef debug
CXXFLAGS   = -DWITHGPU -m64 -mssse3 -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   = -DWITHGPU -m64 -mssse3 -Wno-write-strings -O3 -I. -I$(CUDA)/include
endif
# Für CUDA 12.x mit Unterstützung für RTX 4090
LFLAGS     = -lpthread $(CUDA_LIBS)
else
ifdef debug
CXXFLAGS   = -m64 -mssse3 -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   = -m64 -mssse3 -Wno-write-strings -O3 -I. -I$(CUDA)/include
endif
LFLAGS     = -lpthread
endif


#--------------------------------------------------------------------

ifdef gpu
ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include -gencode=arch=compute_$(CCAP_USED),code=sm_$(CCAP_USED) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=255 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O3 -use_fast_math -I$(CUDA)/include -gencode=arch=compute_$(CCAP_USED),code=sm_$(CCAP_USED) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/GPU/CudaConfig.o: GPU/CudaConfig.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: Rhocore-extrem

Rhocore-extrem: $(OBJET)
	@echo Making Rhocore-extrem...
	$(CXX) $(OBJET) $(LFLAGS) -o Rhocore-extrem

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p hash

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o
	@rm -f Rhocore-extrem

# Neue Ziel für die Installation der Abhängigkeiten auf Ubuntu
install-deps:
	sudo apt update
	sudo apt install -y build-essential
	sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Neue Ziel für die Kompilierung mit Standardoptionen für RTX 4090
gpu-4090:
	make gpu=1 CCAP=89 all

# Neue Ziel für die Kompilierung mit Debug-Informationen
debug:
	make debug=1 all

gpu-debug:
	make gpu=1 debug=1 CCAP=89 all

# Ziel für die Überprüfung der CUDA-Installation
check-cuda:
	@echo "Checking CUDA installation..."
	@if [ -d "$(CUDA)" ]; then \
		echo "CUDA directory found: $(CUDA)"; \
		if [ -f "$(CUDA)/bin/nvcc" ]; then \
			echo "NVCC found: $(CUDA)/bin/nvcc"; \
			$(CUDA)/bin/nvcc --version; \
		else \
			echo "NVCC not found in $(CUDA)/bin/"; \
		fi; \
	else \
		echo "CUDA directory not found: $(CUDA)"; \
	fi