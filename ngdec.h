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

#define MAX_SENTENCE_LENGTH   100
#define NUM_MTU_OPTS           100
#define MAX_VOCAB_SIZE   10000000
#define MAX_PHRASE_LEN         10
#define NUM_RECOMB_BUCKETS   10231
#define MAX_GAPS_TOTAL         32

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

#define W_LM     0
#define W_TM     1
#define W_GEN_S  2
#define W_GAP    3
#define W_BREV   4
#define W_COPY   5
#define W_MAX_ID 6

string OP_NAMES[OP_MAXIMUM] = { "OP_UNKNOWN", "OP_INIT", "OP_GEN_ST", "OP_CONT_WORD", "OP_CONT_GAP", "OP_GEN_S", "OP_GEN_T", "OP_GAP", "OP_JUMP_B", "OP_JUMP_E" };
char   OP_CHAR[OP_MAXIMUM+1]  = "?0GwgST_JE";   // +1 for \0

#define DEBUG (info->debug_level)

typedef unsigned short posn;   // should be large enough to store MAX_SENTENCE_LENGTH+1
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
  
  bitset<MAX_SENTENCE_LENGTH> cov_vec;
  posn cov_vec_count;
  uint32_t cov_vec_hash;

  posn n;                             // current position in src
  posn Z;                             // right-most position in src
  bitset<MAX_SENTENCE_LENGTH> * gaps;
  bool gaps_alloc;
  posn gaps_count;

  lm::ngram::State * lm_context;
  uint32_t lm_context_hash;

  lm::ngram::State * tm_context;
  uint32_t tm_context_hash;

  //bool skippable; // recombination says we can be skipped!
  bool pruned;   // we've been pruned for post (recomb_friend!=NULL tells us that we've been recombined)
  posn num_generated;  // how many target-side words have we generated

  float W[W_MAX_ID];      // this is JUST our feature values, NOT accumulated ones
  float   cost;   // = info->W * (sum_{this and all parents} W)
  hypothesis * prev;
  //vector<hypothesis*> *next;
  //hypothesis * recomb_friend;
  vector<hypothesis*> * recomb_friends;  // if we're the BEST, then recomb_friends is everything that's equivalent to us but worse
  bool recombined;  // if we're not the BEST then we've been recombined
  bool follows_optimal_path;
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

class VocabDictionary : public lm::EnumerateVocab {
 private:
  vector<char> data;
  vector<size_t> index;
  size_t max_idx;
  size_t cur_pos;

 public:
  VocabDictionary() {
    data.reserve(5 * 100000); // 100k words, avg 4 chars long + \0 terminating
    index.push_back(0);
    max_idx = 0;
    cur_pos = 0;
  }

  void Add(lexeme idx, const StringPiece &str) {
    assert(idx == max_idx);
    size_t len = str.length();
    data.reserve(cur_pos + len + 1);
    strcpy(data.data() + cur_pos, str.data());
    cur_pos += len + 1;
    index.push_back(cur_pos);
    max_idx++;
  }

  const char* Get(lexeme idx) {
    assert(idx < max_idx);
    return data.data() + index[idx];
  }
};  

/*
class VocabDictionary : public lm::EnumerateVocab {
public:
  void Add(lexeme index, const StringPiece &str) {
    strings.reserve(index+1);
    for (; max_idx < index; max_idx++)
      strings.push_back("*unk*");
    if ((index == 0) && (max_idx == 0))
      strings.push_back(str.as_string());
    else if (index == max_idx+1)
      strings.push_back(str.as_string());
    else
      strings[index] = str.as_string();
  }
  const string Get(lexeme index) {
    if (index > max_idx)
      return "*unk*";
    else
      return strings[index];
  }

private:
  vector<string> strings;
  lexeme max_idx;
};
*/

struct translation_info {
  lexeme sent[MAX_SENTENCE_LENGTH];
  posn N;

  hypothesis_ring * hyp_ring;
  ring<lm::ngram::State> * lm_state_ring;
  ring<lm::ngram::State> * tm_state_ring;

  recombination_data * recomb_buckets;

  lm::ngram::ProbingModel * language_model;
  lm::ngram::ProbingModel * opseq_model;
  //vector< pair<lexeme,lexeme> > vocab_match;
  unordered_map<lexeme,lexeme> vocab_match;

  vector< vector<mtu_for_sent*> > mtus_at;
  vector< float > estimated_cost;

  float (*compute_cost)(void*,hypothesis*);

  mtu_item_dict mtu_dict;

  VocabDictionary * vocab_dictionary;
  vector<operation> forced_op_seq;
  size_t forced_op_posn;
  set<mtuid> forced_keep_mtus;

  size_t bleu_intersection[4];
  size_t bleu_ref_counts[4];
  size_t bleu_total_hyp_len;
  bleu_stats bleu_total_stats;
  
  // settings
  uint32_t operation_allowed;
  float    pruning_coefficient;
  size_t   max_bucket_size;
  size_t   max_gaps;
  size_t   max_gap_width;
  size_t   max_phrase_len;  // must be <= MAX_PHRASE_LEN
  size_t   num_kbest_predictions;
  size_t   max_mtus_per_token;
  bool     allow_copy;
  bool     forced_decode;
  size_t   train_laso;   // 0 = don't train, >=1 means that many iterations
  size_t   debug_level;

  float W[W_MAX_ID];

  // status
  size_t total_sentence_count;
  size_t total_word_count;
  size_t next_sentence_print;
  float  total_output_cost;
  size_t num_laso_updates;
};

struct astar_item {
  hypothesis * me;
  float path_cost_to_end;
  float future_cost_to_start;
  float W[W_MAX_ID]; // accumulated over everything we've touched
  astar_item * parent;
};

struct astar_result {
  float cost;
  vector<lexeme> trans;
  float W[W_MAX_ID];
};

struct hyp_stack {
  vector<hypothesis*> Stack;
  float lowest_cost;
  float highest_cost;
  float prune_if_gt;
  size_t num_marked_skippable;
};

#endif
