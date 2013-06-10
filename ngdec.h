#ifndef NGDEC_H__
#define NGDEC_H__

#include <stdio.h>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <bitset>
#include <set>
#include "lm/model.hh"

using namespace std;

#define INIT_HYPOTHESIS_RING_SIZE 1024

#define MAX_SENTENCE_LENGTH   200
#define NUM_MTU_OPTS           10
#define MAX_VOCAB_SIZE   10000000
#define MAX_PHRASE_LEN         10
#define NUM_RECOMB_BUCKETS  10231

#define OP_UNKNOWN   0
#define OP_INIT      1
#define OP_GEN_ST    2
#define OP_CONT_WORD 3
#define OP_CONT_GAP  4
#define OP_GEN_S     5
#define OP_GEN_T     6
#define OP_GAP       7
#define OP_JUMP_B    8
#define OP_JUMP_E    9
#define OP_MAXIMUM   10

string OP_NAMES[OP_MAXIMUM] = { "OP_UNKNOWN", "OP_INIT", "OP_GEN_ST", "OP_CONT_W", "OP_CONT_G", "OP_GEN_S", "OP_GEN_T", "OP_GAP", "OP_JUMP_B", "OP_JUMP_E" };

typedef unsigned short posn;   // should be large enough to store MAX_SENTENCE_LENGTH
typedef lm::WordIndex lexeme;
typedef uint32_t mtuid;
typedef uint32_t gap_op_t;

struct operation {
  char op;
  mtuid arg1;
  posn  arg2;
};

lexeme UNK_LEX = (lexeme)0; // this "seems" to be the kenlm standard
lexeme BOS_LEX = (lexeme)1;
lexeme EOS_LEX = (lexeme)2;

struct mtu_item {
  posn src_len;
  posn tgt_len;

  lexeme src[MAX_PHRASE_LEN];
  lexeme tgt[MAX_PHRASE_LEN];

  gap_op_t gap_option;  // gap_option[i] means could have gap AFTER lexeme i, where [i] means " (gap_option & (1 << i)) != 0 "  --  assumes max phrase length < 32

  size_t tr_freq;
  size_t tr_doc_freq;
  mtuid  ident;

  bool operator==(const mtu_item &lhs) const { 
    if ((src_len != lhs.src_len) ||
        (tgt_len != lhs.tgt_len) ||
        (gap_option != lhs.gap_option))
      return false;
    for (posn i=0; i<src_len; i++)
      if (src[i] != lhs.src[i])
        return false;
    for (posn i=0; i<tgt_len; i++)
      if (tgt[i] != lhs.tgt[i])
        return false;
    return true;
  }

  
};

struct mtu_for_sent {
  mtu_item *mtu;
  // found_at[i][...] is the list of all positions that mtu_item->src[i] is found
  posn      found_at[MAX_PHRASE_LEN][NUM_MTU_OPTS];
};

typedef unordered_map< lexeme, vector<mtu_item*> > mtu_item_dict;  // we want the mtu_items to be pointers so that mtu_for_sents can point at them easily

struct hypothesis {
  char                  last_op;      // most recent operation
  const mtu_for_sent  * cur_mtu;      // the current (optional) mtu we're working on
  size_t                op_argument;  // relevant argument to operation (optional, default = 0)
  posn queue_head;                    // which word in cur_mtu is "next"
  
  bitset<MAX_SENTENCE_LENGTH> *cov_vec;
  posn cov_vec_count;
  uint32_t cov_vec_hash;

  posn n;                             // current position in src
  posn Z;                             // right-most position in src
  set<posn> * gaps;                   // where are the existing gaps
  bool gaps_alloc;

  lm::ngram::State * lm_context;
  uint32_t lm_context_hash;

  //mtuid * tm_context;
  lm::ngram::State * tm_context;
  uint32_t tm_context_hash;

  bool skippable; // recombination says we can be skipped!

  float   cost;
  hypothesis * prev;
};
  

struct mtu_item_info {
  size_t doc_freq;
  size_t token_freq;
  gap_op_t gap;
};


typedef vector<vector<hypothesis*>> recombination_data;

struct four_lexemes {
  lexeme w[4];
};

struct bleu_stats {
  vector<four_lexemes> w;
  posn ng_counts[4];
};

struct aligned_sentence_pair {
  vector<lexeme>           F;
  vector< vector<lexeme> > E;
  vector< vector<posn>   > A;
};

template<class T> struct ring {
  T * my_T;
  size_t next_T;
  size_t my_size;
  ring<T>* previous_ring;
};

typedef ring<hypothesis> hypothesis_ring;

struct translation_info {
  lexeme sent[MAX_SENTENCE_LENGTH];
  posn N;

  hypothesis_ring * hyp_ring;
  ring<lm::ngram::State> * lm_state_ring;
  ring<lm::ngram::State> * tm_state_ring;

  recombination_data * recomb_buckets;

  lm::ngram::Model * language_model;
  lm::ngram::Model * opseq_model;

  vector< vector<mtu_for_sent*> > mtus_at;
  float (*compute_cost)(void*,hypothesis*);

  size_t bleu_intersection[4];
  size_t bleu_ref_counts[4];
  size_t bleu_total_hyp_len;
  bleu_stats bleu_total_stats;
  
  // settings
  uint32_t operation_allowed;
  float    pruning_coefficient;
  size_t   max_bucket_size;
  float    gen_s_cost;
  float    gap_cost;
  size_t   max_gaps;
  size_t   max_gap_width;
  size_t   max_phrase_len;  // must be <= MAX_PHRASE_LEN

  // status
  size_t total_sentence_count;
  size_t total_word_count;
  size_t next_sentence_print;
};

struct hyp_stack {
  vector<hypothesis*> Stack;
  float lowest_cost;
  float highest_cost;
  float prune_if_gt;
  size_t num_marked_skippable;
};

#endif
