ARCH = $(shell test `g++ -v 2>&1 | tail -1 | cut -d ' ' -f 3 | cut -d '.' -f 1,2` \< 4.3 && echo -march=nocona || echo -march=native)  -std=c++0x
OPTIM_FLAGS = -O3 -fomit-frame-pointer -ffast-math -fno-strict-aliasing

KENLM_DIR = /home/hal/download/kenlm
KENLM_FLAGS = -I$(KENLM_DIR) -DKENLM_MAX_ORDER=9

# for normal fast execution.
FLAGS = $(KENLM_FLAGS) $(ARCH) -Wall $(OPTIM_FLAGS) -D_FILE_OFFSET_BITS=64

# for profiling/debugging
#FLAGS = $(KENLM_FLAGS) -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g -pg

# for valgrind
#FLAGS = $(KENLM_FLAGS) -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g -O0

# for valgrind profiling: run 'valgrind --tool=callgrind PROGRAM' then 'callgrind_annotate --tree=both --inclusive=yes'
#FLAGS = $(KENLM_FLAGS) -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g $(OPTIM_FLAGS)

%.o:	%.cc %.h
	g++ $(FLAGS) -I/usr/include -c $< -o $@

%.o:	%.cc
	g++ $(FLAGS) -I/usr/include -c $< -o $@

ngdec: ngdec.o
	g++ $(FLAGS) -o $@ $< kenlm.a -lrt 

clean:
	rm -f *.o ngdec
