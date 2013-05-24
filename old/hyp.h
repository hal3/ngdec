#ifndef hyp_h
#define hyp_h

#include <stdio.h>
#include <stdint.h>
#include <bitset>
#include "mtu.h"

#define MAX_GAPS 4
#define MAX_SENTENCE_LENGTH 200
typedef unsigned char posn;    // should be large enough to store MAX_SENTENCE_LENGTH

typedef bitset<MAX_SENTENCE_LENGTH> covvec;


struct hyp {
  char    op;        // most recent operation
  const mtu* cur_mtu; // the current (optional) mtu we're working on
  posn num_in_queue;

  posn j;         // posn of last source word covered
  posn Z;         // right-most src word covered so far

  covvec   cov;       // coverage vector of src
  posn     num_uncov; // number of uncovered words
  posn     num_gaps;  // how many gaps have we inserted so far?
  posn     gap_positions[MAX_GAPS];    // where have we inserted gaps?

  hyp*     prev;      // parent (or null)
  float    cost;      // total cost so far
};

struct tdata {
  uint32_t N;      // length of src
  uint32_t src[MAX_SENTENCE_LENGTH];   // actual src sentence
  mtu_dict dict;   // sentence-specific dictionary

  float pruning_coefficient;   // == 10 means prune anything whose cost is >10*min_cost in this stack; <1 means don't prune

  float(*compute_cost)(void*data_,hyp*h);
};


pair< vector<hyp*>, vector<hyp*> > single_vector_decode(tdata*data);
pair< vector<hyp*>, vector<hyp*> > coverage_vector_decode(tdata*data);
vector<const mtu*> hypothesis_mtus(hyp*h);

#endif
