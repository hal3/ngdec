#ifndef NGDEC_H__
#define NGDEC_H__

#include <stdio.h>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <bitset>
#include <set>

using namespace std;

#define INIT_HYPOTHESIS_RING_SIZE 1024

#define MAX_PHRASE_LEN   3
#define MAX_GAPS         5
#define MAX_SENTENCE_LENGTH 200
#define NUM_MTU_OPTS     5
#define MAX_VOCAB_SIZE   10000000
// LM_CONTEXT_LEN=2 means trigram language model
// TM_CONTEXT_LEN=3 means 4-gram translation model
#define LM_CONTEXT_LEN   4
#define TM_CONTEXT_LEN   8

#define OP_UNKNOWN   0
#define OP_INIT      1
#define OP_GEN_ST    2
#define OP_CONT_W    3
#define OP_CONT_G    4
#define OP_GEN_S     5
#define OP_GEN_T     6
#define OP_GAP       7
#define OP_JUMP_B    8
#define OP_JUMP_E    9
#define OP_CONT_SKIP 10
#define OP_MAXIMUM   11

string OP_NAMES[OP_MAXIMUM] = { "unknown", "init", "gen_st", "cont_w", "cont_g", "gen_s", "gen_t", "gap", "jump_b", "jump_e", "cont_skip" };

typedef unsigned char posn;   // should be large enough to store MAX_SENTENCE_LENGTH
typedef uint32_t lexeme;
typedef uint32_t mtuid;

lexeme GAP_LEX = (lexeme)-1;
lexeme BOS_LEX = (lexeme)0;

struct mtu_item {
  posn src_len;
  posn tgt_len;

  lexeme src[MAX_PHRASE_LEN];
  lexeme tgt[MAX_PHRASE_LEN];

  mtuid  ident;
};

struct mtu_for_sent {
  mtu_item *mtu;
  // found_at[i][...] is the list of all positions that mtu_item->src[i] is found
  posn      found_at[MAX_PHRASE_LEN][NUM_MTU_OPTS];
};

typedef unordered_map< lexeme, vector<mtu_item*>* > mtu_item_dict;

struct hypothesis {
  char                  last_op;      // most recent operation
  const mtu_for_sent  * cur_mtu;      // the current (optional) mtu we're working on
  posn queue_head;                    // which word in cur_mtu is "next"
  
  bitset<MAX_SENTENCE_LENGTH> *cov_vec;
  posn cov_vec_count;
  bool cov_vec_alloc;
  uint32_t cov_vec_hash;

  posn n;                             // current position in src
  posn Z;                             // right-most position in src
  set<posn> * gaps;                   // where are the existing gaps
  bool gaps_alloc;


  lexeme * lm_context;
  bool lm_context_alloc;
  uint32_t lm_context_hash;

  mtuid * tm_context;
  uint32_t tm_context_hash;

  bool skippable; // recombination says we can be skipped!

  float   cost;
  hypothesis * prev;
};

struct hypothesis_ring {
  hypothesis * my_hypotheses;
  size_t next_hypothesis;
  size_t my_size;
  hypothesis_ring* previous_ring;
};

struct translation_info {
  lexeme sent[MAX_SENTENCE_LENGTH];
  posn N;

  hypothesis_ring * hyp_ring;

  vector< vector<mtu_for_sent*> > mtus_at;
  uint32_t operation_allowed;
  float  pruning_coefficient;
  float (*compute_cost)(void*,hypothesis*);
};

/*
  algorithm:

  - n indexes S, m indexes T

  * the operations are:
    * GEN_ST (S, T)
        PRE
          - src[n] = S[0]
          - queue is empty
          - not cov[n]
        OP
          - write down all of T at m
          - m += len(T)
          - cov[n] = true
          - n += 1
          - enqueue(S[1..])
    * CONTINUE_WORD
        PRE
          - queue is not empty
          - src[n] = queue.head
          - not cov[n]
        OP
          - queue.pop
          - cov[n] = true
          - n += 1
    * CONTINUE_GAP
        PRE
          - queue is not empty
          - queue[0] = GAP
          - queue[1] can be found at j in src[n..] and not cov[j]
        OP
          - queue.pop
          - insert gap at n
          - n <- j
          - queue.pop
          - cov[n] = true
          - n += 1
    * GAP
        PRE
          - queue is empty
          - last op was not JUMP_BACK (o/w we're recreating a gap we just closed)
          - # gaps < # uncovered F words
        OP
          - insert gap at n
    * GEN_S (S)
        PRE
          - queue is empty
          - src[n] = S
          - last op was not GAP (o/w we could have done the GEN_S first)
          - not cov[n]
        OP
          - cov[n] = true
          - n += 1
    * GEN_T (T)
        PRE
          - queue is empty (maybe???)
        OP
          - write down T at m
          - m += 1
    * JUMP_BACK (w)
        PRE
          - there are >= w many existing gaps
          - queue is empty
          - we are at the end
          - we're not jumping to the same position
        OP
          - n <- wth previous gap
          - remove wth previous gap from queue
    * JUMP_TO_END
        PRE
          - n < max src index (Z)
          - queue is empty
        OP
          - n <- Z



removed:
    * COPY (S)
        PRE
          - queue is empty
          - src[n] = S
          - not cov[n]
        OP
          - write down S at m
          - m += 1
          - cov[n] = true
          - n += 1
  this will only happen when there is no MTU at a current location
   --> so just make "fake" MTUs


claim: GAPS must be followed either by CONTINUEs or GEN_STs
proof:
  if followed by GEN_S, then it's equivalent to do whatever is contained in the gap, and then do the GEN_S
  if followed by GEN_T, then it's equivalent to GEN_T and then do the GAP
  if followed by JUMP_BACK, then (note, we must be a gap at the END):
      - if the JUMP_BACK is to me, then this is a no-op
      - if the JUMP_BACK is before me, then I could do the jump-back first and then jump end and then GAP
  if followed by EOS, then no reason to insert GAP
  if followed by JUMP_END
*/



#endif
