#ifndef NGDEC_H__
#define NGDEC_H__

#include <stdio.h>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <bitset>
#include <set>

using namespace std;

#define MAX_PHRASE_LEN   3
#define MAX_GAPS         4
#define MAX_SENTENCE_LENGTH 200
#define NUM_MTU_OPTS     5

#define OP_UNKNOWN  0
#define OP_INIT     1
#define OP_GEN_ST   2
#define OP_CONT_W   3
#define OP_CONT_G   4
#define OP_GEN_S    5
#define OP_GEN_T    6
#define OP_COPY     7
#define OP_JUMP_B   8
#define OP_JUMP_E   9

string OP_NAMES[10] = { "unknown", "init", "gen_st", "cont_w", "cont_g", "gen_s", "gen_t", "copy", "jump_b", "jump_e" };

typedef unsigned char posn;   // should be large enough to store MAX_SENTENCE_LENGTH
typedef uint32_t lexeme;

lexeme GAP_LEX = (lexeme)-1;

struct mtu_item {
  posn src_len;
  posn tgt_len;

  lexeme src[MAX_PHRASE_LEN];
  lexeme tgt[MAX_PHRASE_LEN];
};

struct mtu_for_sent {
  mtu_item *mtu;
  // found_at[i][...] is the list of all positions that mtu_item->src[i] is found
  posn      found_at[MAX_PHRASE_LEN][NUM_MTU_OPTS];
};

typedef unordered_map< lexeme, vector<mtu_item*>* > mtu_item_dict;

template <posn LEN>
struct hypothesis {
  char                  last_op;      // most recent operation
  const mtu_for_sent  * cur_mtu;      // the current (optional) mtu we're working on
  posn queue_head;                    // which word in cur_mtu is "next"
  
  bitset<LEN> * cov_vec;
  posn cov_vec_count;
  bool cov_vec_alloc;

  posn n;                             // current position in src
  posn Z;                             // right-most position in src
  set<posn> gaps;                     // where are the existing gaps
  float   cost;
  hypothesis * prev;
};

template <posn LEN>
struct translation_info {
  lexeme sent[LEN];
  vector< vector<mtu_for_sent*> > mtus_at;
  float  pruning_coefficient;
  float (*compute_cost)(void*,hypothesis<LEN>*);
};

/*
  algorithm:

  - n indexes S, m indexes T

  * the operations are:
    * GEN_ST (S, T)
        PRE
          - src[n] = S[0]
          - queue is empty
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
        OP
          - queue.pop
          - cov[n] = true
          - n += 1
    * CONTINUE_GAP
        PRE
          - queue is not empty
          - queue[0] = GAP
          - queue[1] can be found at j in src[n..]
        OP
          - queue.pop
          - insert gap at n
          - n <- j
          - queue.pop
          - cov[n] = true
          - n += 1
    * GEN_S (S)
        PRE
          - queue is empty
          - src[n] = S
        OP
          - cov[n] = true
          - n += 1
    * GEN_T (T)
        PRE
          - queue is empty (maybe???)
        OP
          - write down T at m
          - m += 1
    * COPY (S)
        PRE
          - queue is empty
          - src[n] = S
        OP
          - write down S at m
          - m += 1
          - cov[n] = true
          - n += 1
    * JUMP_BACK (w)
        PRE
          - there are >= w many existing gaps
          - queue is empty
        OP
          - n <- wth previous gap
          - remove wth previous gap from queue
    * JUMP_TO_END
        PRE
          - n < max src index (Z)
          - queue is empty
        OP
          - n <- Z

*/



#endif
