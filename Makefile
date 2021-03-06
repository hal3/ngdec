#G++ = g++
G++ = /usr/local/stow/gcc-4.7.2/bin/g++

ARCH = $(shell test `${G++} -v 2>&1 | tail -1 | cut -d ' ' -f 3 | cut -d '.' -f 1,2` \< 4.3 && echo -march=nocona || echo -march=native)  -std=c++0x
OPTIM_FLAGS = -O3 -ffast-math -fno-strict-aliasing -fomit-frame-pointer

KENLM_DIR = ./kenlm
KENLM_FLAGS = -I$(KENLM_DIR) -DKENLM_MAX_ORDER=9
KENLM_LIB = ${KENLM_DIR}/kenlm.a
#KENLM_LIB = ${KENLM_DIR}/kenlm-dbg.a

# for normal fast execution.
FLAGS = $(KENLM_FLAGS) $(ARCH) -Wall $(OPTIM_FLAGS) -D_FILE_OFFSET_BITS=64

# for "fast" profiling
#FLAGS = $(KENLM_FLAGS) -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -pg -O3 -ffast-math -fno-strict-aliasing

# for profiling/debugging
#FLAGS = $(KENLM_FLAGS) -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g -pg

# for valgrind
FLAGS = $(KENLM_FLAGS) -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g -O0

# for valgrind profiling: run 'valgrind --tool=callgrind PROGRAM' then 'callgrind_annotate --tree=both --inclusive=yes'
#FLAGS = $(KENLM_FLAGS) -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g $(OPTIM_FLAGS)

%.o:	%.cc %.h
	${G++} $(FLAGS) -I/usr/include -c $< -o $@

%.o:	%.cc
	${G++} $(FLAGS) -I/usr/include -c $< -o $@

ngdec: ngdec.o
	${G++} $(FLAGS) -o $@ $< ${KENLM_LIB} -lrt 

clean:
	rm -f *.o ngdec
