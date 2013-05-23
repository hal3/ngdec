#ifndef hyp_h
#define hyp_h

#include <stdio.h>
#include <stdint.h>
#include <bitset>
#include "mtu.h"

#define MAX_SENTENCE_LENGTH 256

typedef bitset<MAX_SENTENCE_LENGTH> covvec;

struct hyp {
  const mtu* m;
  covvec   cov;       // coverage vector of src
  uint32_t num_uncov; // number of uncovered words
  hyp*     prev;      // parent (or null)
  float    cost;      // total cost so far
};


struct tdata {
  uint32_t N;      // length of src
  uint32_t src[MAX_SENTENCE_LENGTH];   // actual src sentence
  mtu_dict dict;   // sentence-specific dictionary

  bool allow_reordering;
  float pruning_coefficient;   // == 10 means prune anything whose cost is >10*min_cost in this stack; <1 means don't prune

  float(*compute_cost)(void*data_,hyp*h);
};


pair< vector<hyp*>, vector<hyp*> > single_vector_decode(tdata*data);
pair< vector<hyp*>, vector<hyp*> > coverage_vector_decode(tdata*data);
vector<const mtu*> hypothesis_mtus(hyp*h);

#endif
