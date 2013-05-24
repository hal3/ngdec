ARCH = $(shell test `g++ -v 2>&1 | tail -1 | cut -d ' ' -f 3 | cut -d '.' -f 1,2` \< 4.3 && echo -march=nocona || echo -march=native)  -std=c++0x
OPTIM_FLAGS = -O3 -fomit-frame-pointer -ffast-math -fno-strict-aliasing

# for normal fast execution.
#FLAGS = $(ARCH) -Wall $(OPTIM_FLAGS) -D_FILE_OFFSET_BITS=64

# for profiling/debugging
#FLAGS = -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g -pg

# for valgrind
FLAGS = -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g -O0

# for valgrind profiling: run 'valgrind --tool=callgrind PROGRAM' then 'callgrind_annotate --tree=both --inclusive=yes'
#FLAGS = -Wall $(ARCH) -D_FILE_OFFSET_BITS=64 -g $(OPTIM_FLAGS)

%.o:	%.cc %.h
	g++ $(FLAGS) -c $< -o $@

%.o:	%.cc
	g++ $(FLAGS) -c $< -o $@

ngdec: ngdec.o hyp.o mtu.o
	g++ $(FLAGS) -o $@ $< hyp.o mtu.o

clean:
	rm -f *.o ngdec
