#include <iostream>
#include <functional>
#include <assert.h>
#include <float.h>
#include <stack>
#include <unordered_set>
#include <queue>
#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include "ngdec.h"
#include "string.h"
#include "lm/model.hh"
//#include "mathic_heap.h"

namespace ng=lm::ngram;

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

namespace std {
  template <> struct hash<mtu_item> : public unary_function<mtu_item, size_t> {
    size_t operator()(const mtu_item& v) const {
      return util::MurmurHashNative(&v, sizeof(mtu_item));
    }
  };
}


bool lt_bleu_stat(four_lexemes a, four_lexemes b) {
  return (a.w[0] <  b.w[0]) ||
       ( (a.w[0] == b.w[0]) && (a.w[1] < b.w[1]) ) ||
       ( (a.w[0] == b.w[0]) && (a.w[1] == b.w[1]) && (a.w[2] < b.w[2]) ) ||
       ( (a.w[0] == b.w[0]) && (a.w[1] == b.w[1]) && (a.w[2] == b.w[2]) && (a.w[3] < b.w[3]) );
}

bool eq_bleu_stat(four_lexemes a, four_lexemes b) {
  return (a.w[0] == b.w[0]) &&
         (a.w[1] == b.w[1]) &&
         (a.w[2] == b.w[2]) &&
         (a.w[3] == b.w[3]);
}

void accum_bleu_intersection(bleu_stats R, bleu_stats H, size_t intersection[4]) {
  size_t i=0, j=0;
  while ((i < R.w.size()) && (j < H.w.size())) {
    if (eq_bleu_stat(R.w[i], H.w[j])) {
      if      (R.w[i].w[3] != 0) ++intersection[3];
      else if (R.w[i].w[2] != 0) ++intersection[2];
      else if (R.w[i].w[1] != 0) ++intersection[1];
      else                       ++intersection[0];
      ++i;
      ++j;
    } else if (lt_bleu_stat(R.w[i], H.w[j])) {
      ++i;
    } else {
      assert(lt_bleu_stat(H.w[j], R.w[i]));
      ++j;
    }
  }
}

void compute_bleu_stats(vector<lexeme> E, bleu_stats *stats) {
  posn N = E.size();
  stats->w.clear();

  for (posn n=0; n<N; ++n) {
    // 0 = unk will never be used!
               stats->w.push_back(  { { E[n], 0, 0, 0 } } );
    if (n+1<N) stats->w.push_back( { { E[n], E[n+1], 0, 0 } } );
    if (n+2<N) stats->w.push_back( { { E[n], E[n+1], E[n+2], 0 } } );
    if (n+3<N) stats->w.push_back( { { E[n], E[n+1], E[n+2], E[n+3] } } );
  }
  stats->ng_counts[0] = N;
  stats->ng_counts[1] = (N>=1) ? (N-1) : 0;
  stats->ng_counts[2] = (N>=2) ? (N-2) : 0;
  stats->ng_counts[3] = (N>=3) ? (N-3) : 0;
  sort(stats->w.begin(), stats->w.end(), lt_bleu_stat);
}

void update_bleu_stats(translation_info*info, vector<lexeme>trans) {
  bleu_stats hyp_stats;
  compute_bleu_stats(trans, &hyp_stats);
  accum_bleu_intersection(info->bleu_total_stats, hyp_stats, info->bleu_intersection);
  info->bleu_total_hyp_len += hyp_stats.ng_counts[0];
  //cout << '|' << bleu_intersection[0] << '|' << bleu_intersection[1] << '|' << bleu_intersection[2] << '|' << bleu_intersection[3] << "\t";
}

float compute_overall_bleu(translation_info*info, bool print) {
  float bleu = 1;
  for (posn i=0; i<4; ++i) {
    float v = static_cast<float>(info->bleu_intersection[i]) / static_cast<float>(info->bleu_ref_counts[i]);
    if (print) cerr << v << " ";
    bleu *= v;
  }
  bleu = pow(bleu, 0.25);
  if (info->bleu_total_hyp_len <= info->bleu_ref_counts[0])
    bleu *= exp(1 - ((float)info->bleu_ref_counts[0]) / ((float)info->bleu_total_hyp_len));
  if (print)
    cerr << " | " << info->bleu_total_hyp_len << " " << info->bleu_ref_counts[0] << "\t" << bleu << endl;
  return bleu;
}

char* ensure_argument(int argc, char*argv[], int&i) {
  if (i+1 < argc) {
    ++i;
    return argv[i];
  } else {
    cerr << "error: option '" << argv[i] << "' requires an argument" << endl;
    exit(-1);
  }
}

void turn_off_bit(uint32_t &data, size_t bit) {
  data &= ~(1<<bit);
}

void initialize_translation_info(translation_info &info) {
  info.N = 0;
  info.hyp_ring = NULL;
  info.recomb_buckets = new recombination_data(NUM_RECOMB_BUCKETS);
  info.language_model = NULL;
  info.opseq_model = NULL;
  info.compute_cost = NULL;

  info.forced_decode = false;
  info.train_laso = 0;

  info.tm_state_ring = NULL;
  info.lm_state_ring = NULL;

  //info.vocab_dictionary = NULL;

  memset(info.bleu_intersection, 0, 4*sizeof(size_t));
  memset(info.bleu_ref_counts, 0, 4*sizeof(size_t));
  info.bleu_total_hyp_len = 0;
  info.bleu_total_stats.w.clear();
  memset(info.bleu_total_stats.ng_counts, 0, 4*sizeof(posn));

  info.operation_allowed = (1 << OP_MAXIMUM) - 1;  // all operations
  info.pruning_coefficient = 10.;
  info.max_bucket_size = 500;
  info.max_gaps = 4;
  info.max_gap_width = 16;
  info.max_phrase_len = 5;
  info.num_kbest_predictions = 1;
  info.max_mtus_per_token = 64;
  info.allow_copy = true;

  info.W[W_GEN_S] = 10;
  info.W[W_GAP] = 10;
  info.W[W_TM] = 1;
  info.W[W_LM] = 1;
  info.W[W_BREV] = 1;
  info.W[W_COPY] = 1;

  info.total_sentence_count = 0;
  info.total_word_count = 0;
  info.next_sentence_print = 100;
  info.total_output_cost = 0.;
  info.num_laso_updates = 0;
}

bool gap_allowed_after(gap_op_t gap_option, posn i) {
  assert(i >= 0); assert(i < 31);
  return (gap_option & (1 << i)) != 0;
}

void allow_gap_after(gap_op_t &gap_option, posn i) {
  assert(i >= 0); assert(i < 31);
  gap_option |= (1 << i);
}

void get_op_string(char op_op, mtuid op_arg1, char*op_string) {
  op_string[0] = OP_CHAR[(int)op_op];
  op_string[1] = 0;
  if ((op_string[0] == 'G') && (op_arg1 == (mtuid)-1))
    op_string[0] = 'c';
  else if ((op_string[0] == 'G') || (op_string[0] == 'S') || (op_string[0] == 'T') || (op_string[0] == 'J'))
    sprintf(op_string+1, "%d", op_arg1);
}

void pretty_print_op_seq(vector<operation> ops) {
  bool first = true;
  char op_string[22];
  for (auto op : ops) {
    if (! first) cout<<" ";
    first = false;
    get_op_string(op.op, op.arg1, op_string);
    cout<<op_string;
  }
  cout<<endl;
}

void print_coverage(bitset<MAX_SENTENCE_LENGTH> cov, posn N, posn cursor) {
  for (posn i=0; i<N; ++i) {
    if (cov[i])
      cout << "X";
    else if (i < cursor)
      cout << "_";
    else
      cout << ".";
  }
}


void print_mtu_half(const lexeme*d, posn len, gap_op_t gap_option=0) {
  for (posn i=0; i<len; ++i) {
    if (i > 0) cout << " ";
    cout << d[i];
    if (gap_allowed_after(gap_option, i))
        cout << " _";
  }
}

void print_mtu(mtu_item mtu, bool short_form=false) {
  if (! short_form)
    cout << "[" << mtu.ident << "] " << mtu.tr_doc_freq << "," << mtu.tr_freq << "\t";
  print_mtu_half(mtu.src, mtu.src_len, mtu.gap_option);
  cout << " | ";
  print_mtu_half(mtu.tgt, mtu.tgt_len, 0);
}

void print_mtu_set(unordered_map<mtu_item, mtu_item_info > mtus, bool renumber_mtus=false) {
  size_t id = 0;
  for (auto mtu_gap : mtus) {
    if (renumber_mtus) {
      cout << id << "\t";
      ++id;
    } else
      cout << mtu_gap.first.ident << "\t";
    
    cout << mtu_gap.second.doc_freq << " " << mtu_gap.second.token_freq << "\t";  // counts (docs and tokens)

    print_mtu_half(mtu_gap.first.src, mtu_gap.first.src_len, mtu_gap.second.gap);
    cout << " | ";
    print_mtu_half(mtu_gap.first.tgt, mtu_gap.first.tgt_len);
    cout << endl;
  }
}


posn get_translation_length(hypothesis *h) {
  posn len = 0;

  while (h != NULL) {
    if (h->last_op == OP_GEN_ST) {
      if (h->cur_mtu->mtu->tgt_len == 0) // COPY
        ++ len;
      else  // GEN_ST
        len += h->cur_mtu->mtu->tgt_len;
    }
    else if (h->last_op == OP_GEN_T)
      ++len;
    h = h->prev;
  }

  return len;
}

lexeme lookup_vocab_match(unordered_map<lexeme,lexeme> &vmatch, lexeme src) {
  auto entry = vmatch.find(src);
  if (entry == vmatch.end()) return 0;
  return entry->second;
}

lexeme get_copy_id(unordered_map<lexeme,lexeme> &vocab_match, lexeme src) {
  if (src >= MAX_VOCAB_SIZE) return src; // if it's "big" then it's copy is itself
  return lookup_vocab_match(vocab_match, src); // otherwise look it up in the match file
}

unordered_map<lexeme,lexeme> read_vocab_match(char* filename) {
  unordered_map<lexeme,lexeme> vmatch;
  FILE* f = fopen(filename, "r");
  if (f == 0) {
    cerr << "warning: cannot read vocab_match file: " << filename << endl;
    return vmatch;
  }

  while (!feof(f)) {
    lexeme src,tgt;
    size_t nr;
    nr = fscanf(f, "%ud\t%ud\n", &src, &tgt);
    if (nr < 2) break;
    vmatch.insert( { src, tgt } );
  }

  fclose(f);

  // assume it's sorted by src id

  return vmatch;
}

vector<lexeme> get_translation(translation_info *info, hypothesis *h) {
  posn len = get_translation_length(h);
  vector<lexeme> trans(len);

  posn m = len-1;
  while (h != NULL) {
    if (h->last_op == OP_GEN_ST) {
      if (h->cur_mtu->mtu->tgt_len == 0) { // COPY
        trans[m] = get_copy_id(info->vocab_match, info->sent[h->n-1] );
        m--;
      } else { // GEN_ST
        for (posn i=0; i<h->cur_mtu->mtu->tgt_len; ++i) {
          trans[m] = h->cur_mtu->mtu->tgt[h->cur_mtu->mtu->tgt_len - 1 - i];
          m--;
        }
      }
    } else if (h->last_op == OP_GEN_T) {
      trans[m] = h->cur_mtu->mtu->tgt[0];
      m--;
    }

    h = h->prev;
  }

  return trans;
}


void print_hypothesis(translation_info* info, hypothesis *h) {
  cerr << h;
  cerr << " f=" << h->follows_optimal_path;
  cerr << "\top="<<OP_NAMES[(size_t)h->last_op];
  cerr << " ["<<h->op_argument<<"]";
  cerr << "\tmtu=";
  //if (((h->last_op == OP_GEN_ST) || (h->last_op == OP_CONT_WORD) || (h->last_op == OP_CONT_GAP) /* || (h->last_op == OP_CONT_SKIP) */ ) && (h->cur_mtu != NULL)) print_mtu(*h->cur_mtu->mtu); else
  if (h->cur_mtu != NULL) print_mtu(*h->cur_mtu->mtu, true); else
  cerr<<"___";
  cerr << "\tqhd="<<h->queue_head;
  cerr << "\tn="<<h->n;
  cerr << "\tZ="<<h->Z;
  cerr << "\tc "/* <<h->cov_vec */<<" "<<h->cov_vec_count<<"="; 
  if (info == NULL) print_coverage(h->cov_vec, h->Z, h->Z);
  else print_coverage(h->cov_vec, info->N, h->Z);
  cerr << "\t#g="<<h->gaps_count;
  cerr << "\tcst="<<h->cost;
  cerr << "\tprv="<<h->prev;
  cerr << endl;
}

bool is_covered(hypothesis *h, posn n) {
  return h->cov_vec[n];
}

void set_covered(hypothesis *h, posn n) {
  assert(! h->cov_vec[n] );
  h->cov_vec[n] = true;
  h->cov_vec_hash += n * 38904831901;
  ++h->cov_vec_count;
  assert( h->cov_vec_count == h->cov_vec.count() );
  assert( h->cov_vec.count() == h->prev->cov_vec.count()+1);
}


void free_one_hypothesis(hypothesis*h) {
  //if (h->tm_context)       delete h->tm_context;
  //if (h->lm_context)       delete h->lm_context;
  //if (h->cov_vec)          delete h->cov_vec;
  if (h->gaps_alloc)       delete h->gaps;
  //if (h->next)             delete h->next;
  if (h->recomb_friends) delete h->recomb_friends;
}

template<class T> void free_ring(ring<T>*r, void(*delete_T)(T*)) {
  vector< ring<T>* > to_free;
  while (r != NULL) {
    for (size_t i=0; i<r->my_size; ++i) 
      delete_T( r->my_T + i );

    free(r->my_T);
    to_free.push_back(r);
    r = r->previous_ring;
  }
  for (auto r : to_free)
    free(r);
}

template<class T> void free_ring(ring<T>*r) {
  vector< ring<T>* > to_free;
  while (r != NULL) {
    free(r->my_T);
    to_free.push_back(r);
    r = r->previous_ring;
  }
  for (auto r : to_free)
    free(r);
}

template<class T> ring<T>* initialize_ring(size_t desired_size) {
  ring<T> *r = (ring<T>*) calloc(1, sizeof(ring<T>));
  if (r == NULL) {
    cerr << "error: initialize_ring cannot allocate more memory :(" << endl;
    exit(-1);
  }
  r->my_T = (T*) calloc(desired_size, sizeof(T));
  if (r->my_T == NULL) {
    cerr << "error: initialize_ring cannot allocate more memory :(" << endl;
    exit(-1);
  }
  r->next_T = 0;
  r->my_size = desired_size;
  r->previous_ring = NULL;
  return r;
}

template<class T> T* get_next_ring_element(ring<T> *&r) {
  if (r->next_T >= r->my_size) {
    // need to make more hypotheses!
    size_t new_size = r->my_size * 2;
    if (new_size < r->my_size)
      new_size = r->my_size;
    ring<T> * new_ring = initialize_ring<T>(new_size);
    new_ring->previous_ring = r;
    r = new_ring;
  }

  T* ret = r->my_T + r->next_T;
  r->next_T ++;
  return ret;
}


void reset_next_hypothesis(translation_info*info, hypothesis *h, hypothesis *h2) {
  assert(h2 != NULL);

  if (h2->gaps_alloc) delete h2->gaps;

  memcpy(h2, h, sizeof(hypothesis));
  h2->last_op = OP_UNKNOWN;
  h2->gaps_alloc = false;
  if (info->language_model != NULL) {
    //h2->lm_context = get_next_ring_element(info->lm_state_ring);
    //h2->lm_context->ZeroRemaining();
    h2->lm_context = get_next_ring_element(info->lm_state_ring);
    if (h->lm_context != NULL) {
      memcpy(h2->lm_context, h->lm_context, sizeof(ng::State));
    } else {
      h2->lm_context->ZeroRemaining();
      h2->lm_context->length = 0;
    }
  }
  if (info->opseq_model != NULL) {
    //h2->tm_context = get_next_ring_element(info->tm_state_ring);
    //h2->tm_context->ZeroRemaining();
    h2->tm_context = get_next_ring_element(info->tm_state_ring);
    if (h->tm_context != NULL) {
      memcpy(h2->tm_context, h->tm_context, sizeof(ng::State));
    } else {
      h2->tm_context->ZeroRemaining();
      h2->tm_context->length = 0;
    }
  }
  h2->pruned = false;
  h2->prev = h;
  h2->recomb_friends = NULL;
  h2->recombined = false;
  for (size_t i=0; i<W_MAX_ID; i++) h2->W[i] = 0.;
}

hypothesis* get_next_hypothesis(translation_info*info, hypothesis *h) {
  hypothesis *h2 = get_next_ring_element<hypothesis>(info->hyp_ring);

  if (h == NULL) {  // asking for initial hypothesis
    h2->last_op = OP_INIT;
    h2->op_argument = 0;
    h2->cur_mtu = NULL;
    h2->queue_head = 0;
    h2->cov_vec_count = 0;
    h2->cov_vec_hash = 0;
    h2->n = 0;
    h2->Z = 0;
    h2->gaps = new bitset<MAX_SENTENCE_LENGTH>(); h2->gaps_count = 0;
    h2->gaps_alloc = true;
    h2->lm_context = NULL; // this means "BOS"
    if (info->language_model != NULL)
      h2->lm_context_hash = ng::hash_value(info->language_model->BeginSentenceState());
    h2->tm_context = NULL; // this means "BOS"
    if (info->opseq_model != NULL)
      h2->tm_context_hash = ng::hash_value(info->opseq_model->BeginSentenceState());
    h2->pruned = false;
    h2->cost = 0.;
    h2->prev = NULL;
    h2->recomb_friends = NULL;
    h2->recombined = false;
    h2->follows_optimal_path = true;
    h2->num_generated = 0;
    for (size_t i=0; i<W_MAX_ID; i++) h2->W[i] = 0.;
  } else {
    reset_next_hypothesis(info, h, h2);
  }

  return h2;
}

void free_sentence_mtus(vector< vector<mtu_for_sent*> > sent_mtus) {
  for (auto &it1 : sent_mtus)
    for (mtu_for_sent* &it2 : it1) {
      if (it2->mtu->tgt_len == 0)
        free(it2->mtu);
      delete it2;
    }
}

template<class T> bool contains(set<T>* S, T t) {
  return S->find(t) != S->end();
}

template<class T> bool unordered_contains(unordered_set<T>* S, T t) {
  return S->find(t) != S->end();
}

bool need_op_sequence(translation_info *info) {
  return info->forced_decode || (info->train_laso > 0);
}

bool sort_mtu_by_score(mtu_for_sent* a, mtu_for_sent* b) {
  return a->mtu->tr_freq > b->mtu->tr_freq;
}

void build_sentence_mtus(translation_info *info) {
  posn N = info->N;
  vector< vector<mtu_for_sent*> > mtus_at(N);
  //vector<float> estimated_cost(N);

  mtu_item_dict dict = info->mtu_dict;

  for (posn n=0; n<N; ++n) {
    auto dict_entry = dict.find(info->sent[n]);

    if (dict_entry != dict.end()) {
      vector<mtu_item*> mtus = dict_entry->second;
      float last_mtu_tf = FLT_MAX;
      for (mtu_item* mtu_iter : mtus) {
        mtu_for_sent* mtu = new mtu_for_sent(); // (mtu_for_sent*)calloc(1, sizeof(mtu_for_sent));
        mtu->mtu = mtu_iter;

        assert( mtu_iter->tr_freq <= last_mtu_tf );
        last_mtu_tf = mtu_iter->tr_freq;

        for (posn m=0; m<info->max_phrase_len; ++m)
          for (posn i=0; i<NUM_MTU_OPTS; ++i)
            mtu->found_at[m][i] = MAX_SENTENCE_LENGTH+1;

        mtu->found_at[0][0] = n;
        posn last_position = n;
        bool success = true;
        for (posn i=1; i<mtu->mtu->src_len; ++i) {
          posn num_found = 0;
          for (posn n2=last_position+1; n2<N; ++n2) {
            if (n2 > last_position+1 + info->max_gap_width) break;
            if (info->sent[n2] == mtu->mtu->src[i]) {
              mtu->found_at[i][num_found] = n2;
              ++num_found;
              if (num_found > NUM_MTU_OPTS)
                break;
            }
          }
          if (num_found == 0) {
            success = false;
            break;
          }
          last_position = mtu->found_at[i][0];
        }

        if (success) mtus_at[n].push_back(mtu);
        else delete mtu;
      }
    }

    if (mtus_at[n].size() > info->max_mtus_per_token) {
      auto begin  = mtus_at[n].begin();
      auto middle = begin + info->max_mtus_per_token;
      auto end    = mtus_at[n].end();
      partial_sort(begin, middle, end, sort_mtu_by_score);

      begin  = mtus_at[n].begin();
      middle = begin + info->max_mtus_per_token;
      end    = mtus_at[n].end();
      vector<mtu_for_sent*> backup;
      for (auto &mtu_it = middle; mtu_it != end; ++mtu_it) {
        mtu_for_sent* mtu = *mtu_it;
        if (need_op_sequence(info) && contains(&info->forced_keep_mtus, (mtuid)mtu->mtu->ident))
          backup.push_back(mtu);
        else
          delete mtu;
      }

      mtus_at[n].erase(mtus_at[n].begin() + info->max_mtus_per_token, mtus_at[n].end());
      for (mtu_for_sent* mtu : backup)
        mtus_at[n].push_back(mtu);
    }

    if (info->allow_copy  &&
        (get_copy_id(info->vocab_match, info->sent[n]) > 0)) {
      mtu_for_sent* mtu = (mtu_for_sent*)calloc(1, sizeof(mtu_for_sent));

      mtu->mtu = (mtu_item*)calloc(1, sizeof(mtu_item));
      memset(mtu->mtu, 0, sizeof(mtu));
      mtu->mtu->src[0] = info->sent[n];
      mtu->mtu->src_len = 1;
      mtu->mtu->tgt_len = 0;
      mtu->mtu->ident = (mtuid)-1;
      //cerr << "copying " << info->sent[n] << endl;

      for (posn m=0; m<info->max_phrase_len; ++m)
        for (posn i=0; i<NUM_MTU_OPTS; ++i)
          mtu->found_at[m][i] = MAX_SENTENCE_LENGTH+1;

      mtu->found_at[0][0] = n;
      
      mtus_at[n].push_back(mtu);
    }
  }

  info->mtus_at = mtus_at;
  //info->estimated_cost = estimated_cost;
}

bool force_decode_allowed(vector<operation> forced_op_seq, size_t fpos, operation cur_op) {
  if (fpos >= forced_op_seq.size()) return false;
  operation forced_op = forced_op_seq[fpos];

  if (cur_op.op != forced_op.op)
    return false;

  switch(forced_op.op) {
  case OP_GEN_S:    // arg1 = the src word to gen, op_argument = src word
    if ((forced_op.arg1 != cur_op.arg1) || (forced_op.arg2 != cur_op.arg2))
      return false;
    break;

  case OP_CONT_GAP:   // no arguments
    //if (arg2 != h->n-1)
    //  return false;
    break;

  case OP_GAP:      // no arguments
    //if (arg2 != h->n-1)
    //  return false;
    break;

  case OP_JUMP_E:   // no args
    break;

  case OP_JUMP_B:   // arg1 = gap id (counting from right), op_argument = gap id
    if (forced_op.arg1 != cur_op.arg1)
      return false;
    break;

  case OP_GEN_ST:   // arg1 = mtu id, h->cur_mtu->mtu->ident = mtu id
    if ((forced_op.arg1 != cur_op.arg1) || (forced_op.arg2 != cur_op.arg2))
    //if ((h->cur_mtu == NULL) || (h->cur_mtu->mtu == NULL) || 
    //    (arg1 != h->cur_mtu->mtu->ident) ||
    //    (arg2 != h->n-1))
      return false;
    break;

  case OP_CONT_WORD:   // no args
    if (forced_op.arg2 != cur_op.arg2)
      return false;
    break;

  default:
    break;
  }

  return true;
}

bool is_final_hypothesis(translation_info *info, hypothesis *h) {
  posn N = info->N;
  if (h->n < N) return false;
  if (h->Z < N) return false;
  for (posn n=0; n<N; ++n)
    if (! is_covered(h, n))
      return false;
  return true;
}

void add_weights(float* dst, float* a, float* b, float bmult=1.) {
  for (size_t i=0; i<W_MAX_ID; i++)
    dst[i] = a[i] + b[i] * bmult;
};

bool operation_allowed(translation_info *info, char op) {
  return (info->operation_allowed & (1 << op)) > 0;
}

posn diff(posn a, posn b) {
  return (a > b) ? (a-b) : (b-a);
}

inline float score_opseq_model(translation_info*info, ng::State *from, size_t this_op, ng::State *to) {
  //if (DEBUG >= 6) cerr << "score_opseq_model from=" << from << "\top=" << this_op << "\tto=" << to << endl;
  return info->opseq_model->Score(*from,
                                  this_op,
                                  *to);
}

inline float score_opseq_model(translation_info*info, ng::State *from, operation &op, ng::State *to) {
  char op_string[2 + 20];  // op + (64 bits as decimal) + \0
  get_op_string(op.op, op.arg1, op_string);
  size_t this_op = info->opseq_model->GetVocabulary().Index(op_string);
  if (DEBUG >= 6) {
    cerr << "score_opseq_model from=";
    for (size_t ii=0; ii<from->length; ii++) cerr << from->words[ii] << " ";
    cerr << "\top=" << op_string << "=" << this_op << "\tto=";
    for (size_t ii=0; ii<to->length; ii++) cerr << to->words[ii] << " ";
  }
  float lp = score_opseq_model(info, from, this_op, to);
  if (DEBUG >= 6) {
    cerr << "\tnewto=";
    for (size_t ii=0; ii<to->length; ii++) cerr << to->words[ii] << " ";
    cerr<<endl;
  }
  return lp;
}

inline float score_language_model(translation_info*info, ng::State *from, lexeme id, ng::State *to) {
  if (DEBUG >= 6) {
    cerr << "score_language_model from=";
    for (size_t ii=0; ii<from->length; ii++) cerr << from->words[ii] << " ";
    cerr << "\tw=" << id << "\tto=";
    for (size_t ii=0; ii<to->length; ii++) cerr << to->words[ii] << " ";
  }
  float lp = info->language_model->Score(*from, id, *to);
  if (DEBUG >= 6) {
    cerr << "\tnewto=";
    for (size_t ii=0; ii<to->length; ii++) cerr << to->words[ii] << " ";
    cerr<<"\tlp=" << lp << endl;
  }
  return lp;
}

void try_single_expansion(translation_info*info, operation op, size_t num_skipped, bool follows_optimal_path0, hypothesis*h, hypothesis*&h2, mtu_for_sent*mtu, posn m, float incr_cost, float prune_if_gt, float*incrW, bitset<MAX_SENTENCE_LENGTH>*new_gaps, bool &new_gaps_alloc, posn new_gaps_count, function<void(hypothesis*)> add_operation, ng::State cur_tm_state) {
  bool follows_optimal_path = follows_optimal_path0;

  if (follows_optimal_path && need_op_sequence(info)) {
    size_t pos = info->forced_op_posn + num_skipped; // skipped.size();
    follows_optimal_path &= force_decode_allowed(info->forced_op_seq, pos, op);
    if (DEBUG >= 5) cerr << "forcedB " << follows_optimal_path << ":\t" << pos << "=" << OP_CHAR[(int)op.op] << " " << op.arg1 << "/" << op.arg2 << " vs " << pos << "=" << OP_CHAR[(int)info->forced_op_seq[pos].op] << " " << info->forced_op_seq[pos].arg1 << "/" << info->forced_op_seq[pos].arg2 << endl;
    if (info->forced_decode && !follows_optimal_path) return;
  }

  float lm_log_prob = 0;
  float incr_cost2 = 0.;

  if (h2 == NULL)
    h2 = get_next_hypothesis(info, h);
  else
    reset_next_hypothesis(info, h, h2);

  ng::State cur_lm_state = h->lm_context ? *(h->lm_context) : ng::State(info->language_model->BeginSentenceState()); // TODO: move this out of the loop
  bool set_h2_lm_state = false;

  if (op.op == OP_GEN_S) {
    incr_cost2 += info->W[W_GEN_S];
    h2->W[W_GEN_S] += 1.;

    h2->num_generated = h->num_generated;;
    float num_to_be_covered = (float)(h->cov_vec_count + 1);
    float new_brevity = MAX(0, num_to_be_covered - (float)h2->num_generated);
    h2->W[W_BREV] = MAX(new_brevity, h->W[W_BREV]);
    h2->cost += (h2->W[W_BREV] - h->W[W_BREV]) * info->W[W_BREV];
  } else if (op.op == OP_GEN_ST) {
    if (mtu->mtu->tgt_len == 0) { // COPY
      lexeme t = get_copy_id(info->vocab_match, info->sent[m]);
      if (t >= MAX_VOCAB_SIZE) t = UNK_LEX;
      //lm_log_prob += info->language_model->Score(cur_lm_state, t, *(h2->lm_context));
      lm_log_prob += score_language_model(info, &cur_lm_state, t, h2->lm_context);
      set_h2_lm_state = true;
      cur_lm_state = *(h2->lm_context);
    } else {
      for (posn id=0; id<mtu->mtu->tgt_len; ++id) {
        //lm_log_prob += info->language_model->Score(cur_lm_state, mtu->mtu->tgt[id], *(h2->lm_context));
        lm_log_prob += score_language_model(info, &cur_lm_state, mtu->mtu->tgt[id], h2->lm_context);
        cur_lm_state = *(h2->lm_context);
      }
      set_h2_lm_state = true;
    }
    h2->W[W_LM] -= lm_log_prob;
    h2->cost -= lm_log_prob * info->W[W_LM];

    // brevity penalty: you pay $bn if you (will have) generated n fewer E words than you've consumed F words
    h2->num_generated = h->num_generated + MAX(1, mtu->mtu->tgt_len);
    float num_to_be_covered = (float)(h->cov_vec_count + MAX(1, mtu->mtu->src_len));
    float new_brevity = MAX(0, num_to_be_covered - (float)h2->num_generated);
    h2->W[W_BREV] = MAX(new_brevity, h->W[W_BREV]);
    h2->cost += (h2->W[W_BREV] - h->W[W_BREV]) * info->W[W_BREV];
  }

  if (!(follows_optimal_path && (info->train_laso>0)))
    if (h->cost + incr_cost + incr_cost2 > prune_if_gt) { 
      if (DEBUG >= 5) cerr << "pruned! (cost = " << h->cost << " + " << incr_cost << " + " << incr_cost2 << " = " << (h->cost + incr_cost + incr_cost2) << " > " << prune_if_gt << " = prune_if_gt)" << endl;
      return;
    }

  h2->last_op = op.op;
  h2->op_argument = op.arg1;
  if (mtu != NULL)
    h2->cur_mtu = mtu;
  set_covered(h2, m);
  h2->n = m+1;
  add_weights(h2->W, h2->W, incrW);
  h2->cost += incr_cost + incr_cost2;
  if (op.op == OP_GEN_ST) {
    h2->queue_head = 1;
  } else if (op.op == OP_CONT_WORD) {
    h2->queue_head++;
  }

  h2->gaps = new_gaps;
  h2->gaps_alloc = new_gaps_alloc;
  h2->gaps_count = new_gaps_count;

  h2->Z = MAX(h2->Z, h2->n);

  float log_prob = score_opseq_model(info, &cur_tm_state, op, h2->tm_context);

  h2->W[W_TM] -= log_prob;
  h2->cost -= log_prob * info->W[W_TM];

  if ((op.op == OP_GEN_ST) && (op.arg1 == 0)) { // TODO: make sure this is right!
    h2->W[W_COPY] ++;
    h2->cost += info->W[W_COPY];
  }

  if (!(follows_optimal_path && (info->train_laso>0)))
    if (h2->cost > prune_if_gt) { 
      if (DEBUG >= 5) cerr << "pruned! (cost = " << h2->cost << " > " << prune_if_gt << " = prune_if_gt)" << endl;
      return;
    }

  if (h2->cov_vec_count == info->N) {
    // we're done! maybe JUMP_E and then score <eos> with both models
    log_prob = 0.;
    if (h2->Z != h2->n) {
      if (!operation_allowed(info, OP_JUMP_E))
        return;

      operation jump_op = { OP_JUMP_E, 0, 0 };
      cur_tm_state = *(h2->tm_context);
      log_prob += score_opseq_model(info, &cur_tm_state, jump_op, h2->tm_context);
      //get_op_string(OP_JUMP_E, 0, op_string);
      //this_op = info->opseq_model->GetVocabulary().Index(op_string);
      //log_prob += info->opseq_model->Score(cur_tm_state, this_op, *h2->tm_context);

      h2->n = h2->Z;
    }

    cur_tm_state = *h2->tm_context;
    log_prob += score_opseq_model(info, &cur_tm_state, EOS_LEX, h2->tm_context);
    //log_prob += info->opseq_model->Score(cur_tm_state, EOS_LEX, *h2->tm_context);
    h2->W[W_TM] -= log_prob;
    h2->cost -= log_prob * info->W[W_TM];
    if (!(follows_optimal_path && (info->train_laso>0)))
      if (h2->cost > prune_if_gt) { 
        if (DEBUG >= 5) cerr << "pruned! (cost = " << h2->cost << " > " << prune_if_gt << " = prune_if_gt)" << endl;
        return;
      }


    lm_log_prob = 0. * score_language_model(info, &cur_lm_state, EOS_LEX, h2->lm_context);
    //lm_log_prob = info->language_model->Score(cur_lm_state, EOS_LEX, *h2->lm_context);
    cur_lm_state = *(h2->lm_context);
    set_h2_lm_state = true;
    h2->W[W_LM] -= lm_log_prob;
    h2->cost -= lm_log_prob * info->W[W_LM];
    if (!(follows_optimal_path && (info->train_laso>0)))
      if (h2->cost > prune_if_gt) { 
        if (DEBUG >= 5) cerr << "pruned! (cost = " << h2->cost << " > " << prune_if_gt << " = prune_if_gt)" << endl;
        return;
      }
  }

  //  if (!set_h2_lm_state)
  //memcpy(h2->lm_context, &cur_lm_state, sizeof(ng::State));

  h2->tm_context_hash = ng::hash_value(*h2->tm_context);
  h2->lm_context_hash = ng::hash_value(*h2->lm_context);

  if (!(follows_optimal_path && (info->train_laso>0)))
    if (h2->cost > prune_if_gt) { 
      if (DEBUG >= 5) cerr << "pruned! (cost = " << h2->cost << " > " << prune_if_gt << " = prune_if_gt)" << endl;
      return;
    }

  h2->follows_optimal_path = follows_optimal_path;
  add_operation(h2);

  h2 = NULL;  // so we don't try to reuse it!
  new_gaps_alloc = false; // tricky! only want first hypothesis to actually keep it!
}

void expand_to_generation_new(translation_info *info, hypothesis *h, function<void(hypothesis*)> add_operation, float prune_if_gt=FLT_MAX) {
  // we force ourselves to do one of OP_GEN_ST, OP_CONT_WORD,
  // OP_GEN_S. the question is: where does this occur and how do we
  // "get there"?
  //
  // really, ANY (uncovered) location is valid (except for
  // OP_CONT_WORD, in which case it has to be >= h->n). the question
  // is how to get there. let's say the desired location is "m".
  // we just apply the algorith from durrani et al!
  //
  // "failure" cases, in which we eventually get to an "oops -- no
  // expansions" include:
  //   * GEN_ST something with a gap, where we don't have enough gaps left (fixed, I think)
  //   * GEN_ST something that needs a CONT_W but then we cover the CONT_W with a GEN_S (unfixed)
  //   * GEN_ST something that needs a CONT_W but that word is already covered (fixed, I think)
  //   * GEN_ST with >1 word and to produce the second word we'd go beyond out gap distance (fixed, I think)
  //
  // 9875 last count
  
  posn N = info->N;
  posn n = h->n;
  posn Z = h->Z;

  vector<operation> skipped;

  bool queue_empty = true;
  if (//((h->last_op == OP_GEN_ST) || (h->last_op == OP_CONT_WORD) || (h->last_op == OP_CONT_GAP) /* || (h->last_op == OP_CONT_SKIP) */ ) &&
      (h->cur_mtu != NULL) && 
      (h->queue_head < h->cur_mtu->mtu->src_len))
    queue_empty = false;

  char MY_GAP = queue_empty ? OP_GAP : OP_CONT_GAP;
  size_t cont_idx = 0;

  hypothesis *h2 = NULL;
  ng::State cur_tm_state, tmp_tm_state;
  cur_tm_state.ZeroRemaining(); cur_tm_state.length = 0;
  tmp_tm_state.ZeroRemaining(); tmp_tm_state.length = 0;

  posn num_gaps_passed = 0;
  posn first_gap_posn = 0;
  posn recent_gap_posn = 0;

  float incrW[W_MAX_ID];

  posn hi = MIN(N, n+info->max_gap_width+1);
  for (posn m=0; m<hi; m++) {
    if (DEBUG >= 5) cerr << "expand @ m=" << m << endl;

    if ((*h->gaps)[m]) {
      if (num_gaps_passed == 0)
        first_gap_posn = m;    // we need to make sure we don't generate too far away from this!
      recent_gap_posn = m;
      num_gaps_passed++;
    }

    if ((!queue_empty) && (m<n)) continue;

    if (h->cov_vec[m]) {
      if (DEBUG >= 5) cerr << "skipping (m="<<m<<" covered)" << endl;  
      continue;
    }

    if (((m > n) && (m > n + info->max_gap_width)) ||
        ((m < n) && (n > m + 1 + info->max_gap_width))) {
      if (DEBUG >= 5) cerr << "skipping (m="<<m<<"-"<<n<<"=n too big)" << endl;  
      continue; 
    }
    if ((num_gaps_passed>0) && (diff(m,first_gap_posn) > info->max_gap_width)) {  // TODO: if it's CONT_W then it's even worse!
      if (DEBUG >= 5) cerr << "skipping (m="<<m<<"-"<<first_gap_posn<<"=first_gap_posn too big)" << endl;  
      continue;
    }

    posn here = 0;
    if (! queue_empty) { // we're going to cont_word
      while (cont_idx < NUM_MTU_OPTS) {
        here = h->cur_mtu->found_at[h->queue_head][cont_idx];
        if (here > MAX_SENTENCE_LENGTH) break;
        if (m <= here) break;
        cont_idx++;
      }
      if (cont_idx >= NUM_MTU_OPTS) break;
      if (m > here) break;
    }

    skipped.clear();

    posn new_gaps = 0;
    bool too_many_gaps = false;
    posn cur_pos = n;
    bool jumped_before_m = false;
    // j  = cur_pos       j' = m
    // TODO: make sure these gaps aren't too long!
    if (cur_pos < m) {
      if (! h->cov_vec[cur_pos]) {
        skipped.push_back( { MY_GAP, 0, cur_pos } );
        new_gaps++;
        if (h->gaps_count + new_gaps > info->max_gaps) too_many_gaps = true;
      }
      bool any_intervening = false;
      for (posn t=cur_pos; t<m; t++)
        if (h->cov_vec[t]) {
          any_intervening = true;
          break;
        }
      if ((cur_pos == Z) || (! any_intervening)) {
        cur_pos = m;
      } else {
        skipped.push_back( { OP_JUMP_E, 0, cur_pos } );
        cur_pos = Z;
      }
    }
    if (m < cur_pos) {
      if ((cur_pos < Z) && (! h->cov_vec[cur_pos])) {
        skipped.push_back( { MY_GAP, 0, cur_pos } );
        new_gaps++;
        if (h->gaps_count + new_gaps > info->max_gaps) too_many_gaps = true;
      }
      if (cur_pos < Z) {
        skipped.push_back( { OP_JUMP_E, 0, cur_pos } );
        cur_pos = Z;
      }

      posn target_gap = 0;
      for (posn t=m+1; t<=cur_pos; t++)
        if ((*h->gaps)[t]) target_gap++;
      for (operation op : skipped)
        if ((op.op == MY_GAP) && (op.arg2 > m) && (op.arg2 <= cur_pos))
          target_gap++;

      //cerr << "JUMP_B: gap_count-num_pass+new = " << h->gaps_count << " - " << num_gaps_passed << " + " << new_gaps << endl;
      skipped.push_back( { OP_JUMP_B, target_gap, recent_gap_posn } );
      new_gaps--;
      cur_pos = recent_gap_posn;
      if (cur_pos < m) jumped_before_m = true;
    }
    if (cur_pos < m) {
      skipped.push_back( { MY_GAP, 0, cur_pos } );
      new_gaps++;
      if (h->gaps_count + new_gaps > info->max_gaps) too_many_gaps = true;
      cur_pos = m;
    }

    if (jumped_before_m && !queue_empty) {
      if (DEBUG >= 5) cerr << "skipping (m,n="<<m<<","<<n<<" because of jumping before m with non-empty queue)"<<endl;
      continue;
    }

    if (too_many_gaps) {
      if (DEBUG >= 5) cerr << "skipping (m,n="<<m<<","<<n<<" because of too many gaps)"<<endl;
      continue;
    }

    bool invalid_op = false;
    for (operation op : skipped)
      if (!operation_allowed(info, op.op)) {
        invalid_op = true;
        break;
      }
    if (invalid_op) continue;

    bool follows_optimal_path = h->follows_optimal_path;
    if (follows_optimal_path && need_op_sequence(info)) {
      for (size_t i=0; i<skipped.size(); i++) {
        follows_optimal_path &= force_decode_allowed(info->forced_op_seq, info->forced_op_posn+i, skipped[i]);
        if (DEBUG >= 5) cerr << "forcedA " << follows_optimal_path << ":\t" << info->forced_op_posn << "=" << OP_CHAR[(int)skipped[i].op] << " " << skipped[i].arg1 << "/" << skipped[i].arg2 << " vs "
             << OP_CHAR[(int)info->forced_op_seq[info->forced_op_posn+i].op] << " " << info->forced_op_seq[info->forced_op_posn+i].arg1 << "/" << info->forced_op_seq[info->forced_op_posn+i].arg2 << endl;
        if (!follows_optimal_path) break;
      }
      if (info->forced_decode && !follows_optimal_path) continue;
      if (DEBUG >= 5) cerr << "hi m="<<m<<" skipped.size="<<skipped.size()<<endl;
    }

    for (size_t i=0; i<W_MAX_ID; i++) incrW[i] = 0.;

    cur_tm_state = h->tm_context ? *(h->tm_context) : ng::State(info->opseq_model->BeginSentenceState());
    float incr_cost = 0.;

    bitset<MAX_SENTENCE_LENGTH>*new_gaps_ptr = h->gaps;
    bool new_gaps_alloc = false;
    size_t new_gaps_count = h->gaps_count;
  
    for (operation op : skipped) {
      //get_op_string(op.op, op.arg1, op_string);
      //size_t this_op = info->opseq_model->GetVocabulary().Index(op_string);
      //float log_prob = info->opseq_model->Score(cur_tm_state, this_op, tmp_tm_state);
      float log_prob = score_opseq_model(info, &cur_tm_state, op, &tmp_tm_state);
      incrW[W_TM] -= log_prob;
      incr_cost -= log_prob * info->W[W_TM];
      cur_tm_state = tmp_tm_state;

      if (op.op == MY_GAP) {
        incrW[W_GAP] += 1.;
        incr_cost += info->W[W_GAP];

        if (! new_gaps_alloc) {
          new_gaps_ptr = new bitset<MAX_SENTENCE_LENGTH>(*h->gaps);
          new_gaps_alloc = true;
        }
        assert( ! (*new_gaps_ptr)[op.arg2] );
        (*new_gaps_ptr)[op.arg2] = true;
        new_gaps_count++;
      } else if (op.op == OP_JUMP_B) {
        if (! new_gaps_alloc) {
          new_gaps_ptr = new bitset<MAX_SENTENCE_LENGTH>(*h->gaps);
          new_gaps_alloc = true;
        }
        assert( (*new_gaps_ptr)[op.arg2] );
        (*new_gaps_ptr)[op.arg2] = false;
        new_gaps_count--;
      }

      if (!(follows_optimal_path && (info->train_laso>0)))
        if (h->cost + incr_cost > prune_if_gt) break;
    }
    if (!(follows_optimal_path && (info->train_laso>0)))
      if (h->cost + incr_cost > prune_if_gt) continue;

    // if we've made it this far, we can actually try generating words
    if (! queue_empty) {
      if (operation_allowed(info, OP_CONT_WORD) && (here == m)) {
        if (DEBUG >= 5) cerr << "OP_CONT_WORD: " << m << endl;
        operation my_op = { OP_CONT_WORD, 0, m };
        try_single_expansion(info, my_op, skipped.size(), follows_optimal_path, h, h2, 0, m, incr_cost, prune_if_gt, incrW, new_gaps_ptr, new_gaps_alloc, new_gaps_count, add_operation, cur_tm_state);
      }
    } else {
      for (mtu_for_sent* mtu : info->mtus_at[m]) {
        if ( (mtu->mtu->tgt_len == 0) && !info->allow_copy) continue;
        if ( (mtu->mtu->tgt_len >  0) && !operation_allowed(info, OP_GEN_ST)) continue;

        operation my_op = { OP_GEN_ST, mtu->mtu->ident, m };
        if (DEBUG >= 5) cerr << "OP_GEN_ST: " << mtu->mtu->ident << endl;

        posn num_gaps_required = 0;
        posn next_word_pos = m+1;
        for (size_t idx=1; idx<mtu->mtu->tgt_len; idx++) {
          bool found = false;
          for (size_t opt=0; opt<NUM_MTU_OPTS; opt++) {
            posn pos = mtu->found_at[idx][opt];
            if (pos > MAX_SENTENCE_LENGTH) break;
            if (pos < next_word_pos) continue;
            if (h->cov_vec[pos]) {
              //if (pos < MAX(m+1,h->Z)) num_gaps_required++;
              continue;
            }
            if (pos >= next_word_pos) {
              next_word_pos = pos + 1;
              found = true;
              break;
            }
          }
          if (!found) num_gaps_required = MAX_SENTENCE_LENGTH+1;
          if (h->gaps_count + num_gaps_required > info->max_gaps) break;
        }

        // if ((num_gaps_passed>0) && (diff(next_word_pos,first_gap_posn) > info->max_gap_width)) {
        // } else
        if (true || h->gaps_count + num_gaps_required <= info->max_gaps) {
          try_single_expansion(info, my_op, skipped.size(), follows_optimal_path, h, h2, mtu, m, incr_cost, prune_if_gt, incrW, new_gaps_ptr, new_gaps_alloc, new_gaps_count, add_operation, cur_tm_state);
        } else {
          //cerr << "?";
        }
      }
    }
    if (operation_allowed(info, OP_GEN_S)) {
      operation my_op = { OP_GEN_S, info->sent[m], m };
      try_single_expansion(info, my_op, skipped.size(), follows_optimal_path, h, h2, 0, m, incr_cost, prune_if_gt, incrW, new_gaps_ptr, new_gaps_alloc, new_gaps_count, add_operation, cur_tm_state);

      //cerr << "OP_GEN_S: " << info->sent[m] << endl;
      //possible_ops.push_back( my_op );
      //possible_mtus.push_back(0);
    }
    //bool follows_optimal_path0 = follows_optimal_path;
    //assert(possible_ops.size() == possible_mtus.size());
  }
}

class astar_hypothesis_lt {
public:
  bool operator()(astar_item* a, astar_item* b) { // true if a<b
    return (a->path_cost_to_end + a->future_cost_to_start) > 
           (b->path_cost_to_end + b->future_cost_to_start);
  }
};
typedef priority_queue<astar_item*,vector<astar_item*>,astar_hypothesis_lt> astar_pqueue;

// class astar_hypothesis_config {
// public:
//   typedef astar_item* Entry;
//   typedef float CompareResult;
//   inline const CompareResult compare(Entry a, Entry b) {
//     return (a->path_cost_to_end + a->future_cost_to_start) -
//            (b->path_cost_to_end + b->future_cost_to_start);
//   }
//   inline const bool cmpLessThan(CompareResult r) { return r  > 0; }
//   inline const bool cmpEqual(CompareResult r)    { return r == 0; }
//   static const bool supportDeduplication = false;
//   static const bool fastIndex = true;
//   static const Entry deduplicate(Entry a, Entry value) { return a; }
// };
// typedef mathic::Heap<astar_hypothesis_config> astar_pqueue;

vector<lexeme> get_astar_translation(translation_info*info, astar_item *item) {
  vector<lexeme> ret;
  float sumlm=0;
  for (; item != NULL; item = item->parent) {
    hypothesis*h = item->me;
    sumlm += h->W[W_LM];
    if (h->last_op == OP_GEN_ST) {
      if (h->cur_mtu->mtu->tgt_len == 0) { // COPY
        ret.push_back( get_copy_id( info->vocab_match, info->sent[h->n] ) );
        //cerr << "COPY  " << info->sent[h->n];
        //cerr << " (" << sumlm << ")" << "\t" << h << endl;
      } else {
        // cerr << "GENST";
        for (posn i=0; i<h->cur_mtu->mtu->tgt_len; ++i) {
          ret.push_back(h->cur_mtu->mtu->tgt[i]);
          //cerr << " " << h->cur_mtu->mtu->tgt[i];
        }
        // cerr << " <-";
        // for (posn i=0; i<h->cur_mtu->mtu->src_len; ++i) {
        //   cerr << " " << h->cur_mtu->mtu->src[i];
        // }
        // cerr << "(" << sumlm << ")" << "\t" << h << endl;
      }
    } else if (h->last_op == OP_GEN_T) {
      ret.push_back(h->cur_mtu->mtu->tgt[0]);
    } else {
      // cerr << "op: " << OP_NAMES[h->last_op] << " " << h->op_argument;
      // cerr << " (" << sumlm << ")" << "\t" << h << endl;
    }
  }
  // cerr << endl;
  return ret;
}

void assert_not_all_same_ops(hypothesis*a0, hypothesis*b0) {
  hypothesis*a = a0;
  hypothesis*b = b0;
  while (true) {
    if (a == b) assert(false);
    if (a == NULL) return;
    if (b == NULL) return;
    if (a->last_op != b->last_op) return;
    if (a->op_argument != b->op_argument) return;
    a = a->prev;
    b = b->prev;
  }
}

vector<astar_result> astar_kbest_search(translation_info*info, vector<hypothesis*> Goals) {
  vector<astar_result> found_items;
  astar_pqueue Q;
  size_t num_final_in_Q = 0;
  float highest_cost_in_Q = FLT_MIN;
  unordered_set<hypothesis*> visited;

  ring<astar_item> *astar_ring = initialize_ring<astar_item>(10001);

  for (hypothesis *h_end : Goals) {
    //astar_item *item = new astar_item();
    if (unordered_contains(&visited, h_end)) continue;
    visited.insert(h_end);

    astar_item *item = get_next_ring_element(astar_ring);
    item->me = h_end;
    item->path_cost_to_end = 0;
    item->future_cost_to_start = h_end->cost;
    item->parent = NULL;
    memcpy(item->W, h_end->W, sizeof(float) * W_MAX_ID);
    Q.push(item);
    highest_cost_in_Q = MAX(highest_cost_in_Q, item->path_cost_to_end + item->future_cost_to_start);
    if (item->me->prev == NULL) num_final_in_Q++;

    if (h_end->recomb_friends != NULL)
      for (hypothesis *fr : *h_end->recomb_friends) {
        if (unordered_contains(&visited, fr)) continue;
        visited.insert(fr);

        item = get_next_ring_element(astar_ring);
        item->me = fr;
        item->path_cost_to_end = 0;
        item->future_cost_to_start = fr->cost;
        item->parent = NULL;
        memcpy(item->W, fr->W, sizeof(float) * W_MAX_ID);
        Q.push(item);
        highest_cost_in_Q = MAX(highest_cost_in_Q, item->path_cost_to_end + item->future_cost_to_start);
        if (item->me->prev == NULL) num_final_in_Q++;
      }
  }

  while (!Q.empty()) {
    astar_item *item = Q.top(); Q.pop();
    hypothesis *h = item->me;

    assert( h != NULL );
    if (h->pruned) continue;

    if (h->prev == NULL) {  // this is an initial item
      astar_result res;
      res.cost = item->path_cost_to_end;
      res.trans = get_astar_translation(info, item);
      memcpy(res.W, item->W, W_MAX_ID * sizeof(float));
      found_items.push_back(res);
      if (found_items.size() >= info->num_kbest_predictions)
        break;
    } else {  // this is not a final item
      // the first predecessor is "prev"
      hypothesis *p = h->prev;
      if (! p->pruned) {
        astar_item *new_item = get_next_ring_element(astar_ring);
        new_item->me = p;
        new_item->path_cost_to_end = item->path_cost_to_end + h->cost - p->cost;
        new_item->future_cost_to_start = p->cost;
        add_weights(new_item->W, item->W, p->W);
        new_item->parent = item;
        if ((num_final_in_Q < info->num_kbest_predictions) || (new_item->path_cost_to_end + new_item->future_cost_to_start < highest_cost_in_Q)) {
          Q.push(new_item);
          highest_cost_in_Q = MAX(highest_cost_in_Q, new_item->path_cost_to_end + new_item->future_cost_to_start);
          if (new_item->me->prev == NULL) num_final_in_Q++;
        }
        // the other possible predecessors are prevs of my "friends"
        if (p->recomb_friends != NULL) {
          for (hypothesis *fr : *(p->recomb_friends)) {
            assert(fr->recombined);
            //assert(fr->prev != NULL);
            //assert(! fr->prev->recombined );

            if (! fr->pruned) {
              new_item = get_next_ring_element(astar_ring);
              new_item->me = fr;
              new_item->path_cost_to_end = item->path_cost_to_end + h->cost - p->cost;
              new_item->future_cost_to_start = fr->cost;
              add_weights(new_item->W, item->W, fr->W);
              new_item->parent = item;
              if ((num_final_in_Q < info->num_kbest_predictions) || (new_item->path_cost_to_end + new_item->future_cost_to_start < highest_cost_in_Q)) {
                Q.push(new_item);
                highest_cost_in_Q = MAX(highest_cost_in_Q, new_item->path_cost_to_end + new_item->future_cost_to_start);
                if (new_item->me->prev == NULL) num_final_in_Q++;
              }
            }
          }
        }
      }
    }
  }

  free_ring<astar_item>(astar_ring);

  return found_items;
}


size_t bucket_contains_equiv(translation_info*info, vector<hypothesis*> bucket, hypothesis *h) {
  for (size_t pos=0; pos<bucket.size(); ++pos) {
    hypothesis *h2 = bucket[pos];
    if (h2->pruned) continue;
    if (h2->recombined) continue;
    if (h->lm_context_hash != h2->lm_context_hash) continue;
    if (h->tm_context_hash != h2->tm_context_hash) continue;
    if (h->cov_vec_hash    != h2->cov_vec_hash   ) continue;

    if (info->language_model != NULL)
      if (h->lm_context != h2->lm_context)
        if (! (*(h->lm_context) == *(h2->lm_context)))
          continue;

    if (h->tm_context != h2->tm_context)
      if (! ((*h->tm_context) == (*h2->tm_context)))
        continue;

    (h->cov_vec) ^= (h2->cov_vec);
    bool cov_vec_eq = ! h->cov_vec.any();
    (h->cov_vec) ^= (h2->cov_vec);

    if (! cov_vec_eq)
        continue;

    return pos;
  }
  return (size_t)-1;
}

void initialize_hyp_stack(hyp_stack* stack) {
  stack->lowest_cost  = FLT_MAX;
  stack->highest_cost = FLT_MIN;
  stack->prune_if_gt  = FLT_MAX;
  stack->num_marked_skippable = 0;
}

void add_recomb_friend(translation_info *info, hypothesis *better, hypothesis *worse) {
  assert(better->cost <= worse->cost);
  if (better->recomb_friends == NULL) {
    if (worse->recomb_friends != NULL) {
      better->recomb_friends = worse->recomb_friends;
      worse->recomb_friends = NULL;
    } else {
      better->recomb_friends = new vector<hypothesis*>;
    }
  } else {
    assert( worse->recomb_friends == NULL );
  }

  better->recomb_friends->push_back(worse);
  if (DEBUG >= 3) { cerr << "combinging worse=" << worse << " with better=" << better << endl; }
  worse->recombined = true;
}

void add_to_hyp_stack(translation_info*info, hyp_stack* stack, hypothesis* h) {
  if ((h->cost > stack->prune_if_gt) ||
      ((stack->Stack.size() >= info->max_bucket_size + stack->num_marked_skippable) && 
       (h->cost >= stack->highest_cost)))
    h->pruned = true;

  if (h->pruned || h->recombined) {
    if (h->follows_optimal_path && (DEBUG>=4)) cerr << "optimal path pruned!" << endl;
    return;
  }

  bool we_were_worse = false;
  if (info->recomb_buckets) {  // try recombining before we try pruning
    recombination_data *buckets = info->recomb_buckets;
    size_t mod = buckets->size();

    size_t id = (h->lm_context_hash * 3481183 +
                 h->tm_context_hash * 8942137 +
                 h->cov_vec_hash    * 9138921) % mod;
  
    size_t equiv_pos = bucket_contains_equiv(info, (*buckets)[id], h);

    if (equiv_pos != (size_t) -1) {   // we can recombine at equiv_pos
      if (h->cost >= (*buckets)[id][equiv_pos]->cost) {
        // we're more expensive, so ignore
        we_were_worse = true;
        add_recomb_friend(info, (*buckets)[id][equiv_pos], h);
      } else {
        // we're cheaper
        assert(h->recomb_friends == NULL);
        h->recomb_friends = (*buckets)[id][equiv_pos]->recomb_friends;
        (*buckets)[id][equiv_pos]->recomb_friends = NULL;
        add_recomb_friend(info, h, (*buckets)[id][equiv_pos]);
        stack->num_marked_skippable++;
      }
    } else {
      (*buckets)[id].push_back(h);
    }      
  }

  if (we_were_worse) {
    if (h->follows_optimal_path && (DEBUG>=4)) cerr << "optimal path was worse!" << endl;
    return;
  }

  stack->Stack.push_back(h);

  if (h->cost > stack->highest_cost) 
    stack->highest_cost = h->cost;

  if (h->cost < stack->lowest_cost ) { 
    stack->lowest_cost  = h->cost;
    if (info->pruning_coefficient >= 1)
      stack->prune_if_gt = stack->lowest_cost * info->pruning_coefficient;
  }

}

bool sort_hyp_by_cost(hypothesis*a, hypothesis*b) { 
  bool a_skippable = a->pruned || (a->recombined);
  bool b_skippable = b->pruned || (b->recombined);
  if (a_skippable && b_skippable) return true;
  if (a_skippable) return false; // a is skippable, b not
  if (b_skippable) return true;  // b is skippable, a not
  return a->cost < b->cost; 
}

void shrink_hyp_stack(translation_info*info, hyp_stack* stack, bool force_shrink=false) {
  if ((!force_shrink) && (stack->Stack.size() /* - stack->num_marked_skippable */ <= info->max_bucket_size))
    return;

  if ((!force_shrink) && (stack->Stack.size() <= 2*info->max_bucket_size))
    return;

  if (stack->Stack.size() > info->max_bucket_size) {
    auto begin  = stack->Stack.begin();
    auto end    = stack->Stack.end();
    auto middle = begin + info->max_bucket_size;
    partial_sort(begin, middle, end, sort_hyp_by_cost);

    middle = stack->Stack.begin() + info->max_bucket_size;
    end    = stack->Stack.end();
    for (auto h=middle; h<end; ++h) {
      if ((*h)->follows_optimal_path && (DEBUG>=4)) cerr << " (optimal path got shrunk out) ";
      (*h)->pruned = true;
    }

    stack->Stack.erase(stack->Stack.begin() + info->max_bucket_size, stack->Stack.end());
  }

  stack->num_marked_skippable = 0;
  for (hypothesis *h : stack->Stack) {
    if (h->cost > stack->prune_if_gt) h->pruned = true;
    stack->num_marked_skippable += (h->pruned || h->recombined);
  }
}

bool is_movement_operator(operation op) {
  return (op.op == OP_CONT_GAP) || (op.op == OP_GAP) || (op.op == OP_JUMP_B) || (op.op == OP_JUMP_E);
}

void make_laso_update(translation_info*info, hypothesis*opt, hyp_stack*stack, float eta=0.1) {
  if (stack->Stack.size() == 0)
    return;

  info->num_laso_updates++;

  float sqrt_T = 1.; // sqrt((float)(info->num_laso_updates));

  // update toward opt
  
  float stack_update = -eta / sqrt_T;
  add_weights(info->W, info->W, opt->W, stack_update);

  stack_update = eta / (float)stack->Stack.size() / sqrt_T;
  // and away from the stack; TODO: away from recomb too???
  for (hypothesis *bad : stack->Stack)
    add_weights(info->W, info->W, bad->W, stack_update);

  // ensure none dropped below zero
  for (size_t i=0; i<W_MAX_ID; i++)
    if (info->W[i] < 0.)
      info->W[i] = 0.;

  if (DEBUG >= 5) {
    cerr << "current weight vector:";
    for (size_t i=0; i<W_MAX_ID; i++) {
      if (i > 0) cerr << ",";
      cerr << info->W[i];
    }
    cerr << endl;
  }
}

vector<hypothesis*> stack_generic_search(translation_info *info, size_t (*get_stack_id)(hypothesis*), size_t num_stacks_reserve=0) {
  size_t GoalStack = (size_t)-1;
  unordered_map< size_t, hyp_stack* > Stacks;
  unordered_map< size_t, hypothesis* > StackOptimum;
  stack< size_t > NextStacks;

  if (num_stacks_reserve > 0)
    Stacks.reserve(num_stacks_reserve);

  hypothesis *h0 = get_next_hypothesis(info, NULL);
  size_t stack0 = get_stack_id(h0);
  NextStacks.push(stack0);
  Stacks[stack0] = new hyp_stack();
  initialize_hyp_stack(Stacks[stack0]);
  add_to_hyp_stack(info, Stacks[stack0], h0);

  Stacks[GoalStack] = new hyp_stack();
  initialize_hyp_stack(Stacks[GoalStack]);

  while (! NextStacks.empty()) {
    size_t cur_stack = NextStacks.top(); NextStacks.pop();
    if (Stacks[cur_stack] == NULL) continue;

    auto it = Stacks[cur_stack]->Stack.begin();
    size_t num_processed = 0;
    float prune_if_gt = FLT_MAX;
    //float lowest_goal_cost = FLT_MAX;
    //float highest_goal_cost = FLT_MIN;

    while (it != Stacks[cur_stack]->Stack.end()) {
      num_processed++;
      hypothesis *h = *it;

      if (DEBUG >= 5) cerr << "cost=" << h->cost << " prune_if_gt=" << Stacks[cur_stack]->prune_if_gt << " pruned=" << h->pruned << " recombined=" << h->recombined << endl;

      if (h->cost > Stacks[cur_stack]->prune_if_gt) { 
        assert(!((info->train_laso>0) && h->follows_optimal_path)); 
        h->pruned = true; 
      }
      if (h->pruned) { assert(!((info->train_laso>0) && h->follows_optimal_path)); ++it; continue; }
      if (h->recombined) { assert(!((info->train_laso>0) && h->follows_optimal_path)); ++it; continue; }

      size_t diff = it - Stacks[cur_stack]->Stack.begin();

      if (DEBUG >= 3) { cerr << "expanding: "; print_hypothesis(info, h); }
      if (DEBUG >= 5 && info->forced_decode) {
        posn ipos = info->forced_op_posn;
        for (; ipos<info->forced_op_seq.size() && is_movement_operator(info->forced_op_seq[ipos]); ipos++)
          cerr << "forced: " << OP_NAMES[(short)info->forced_op_seq[ipos].op] << " : " << info->forced_op_seq[ipos].arg1 << " : " << info->forced_op_seq[ipos].arg2 << endl;
        cerr << "forced: " << OP_NAMES[(short)info->forced_op_seq[ipos].op] << " : " << info->forced_op_seq[ipos].arg1 << " : " << info->forced_op_seq[ipos].arg2 << endl;
      }

      if (NextStacks.size() == 1) {
        size_t next_stack = NextStacks.top();
        prune_if_gt = Stacks[next_stack]->prune_if_gt;
        /*      } else if ((NextStacks.size() == 0) && (Goals.size() > 0)) {
        if (info->num_kbest_predictions <= 1)
          prune_if_gt = lowest_goal_cost;
        else
        prune_if_gt = highest_goal_cost; */
      } else {
        prune_if_gt = FLT_MAX;
      }

      size_t num_expansions = 0;
      expand_to_generation_new(info, h, [info,cur_stack,&StackOptimum,GoalStack,&Stacks,&NextStacks,get_stack_id,&num_expansions](hypothesis* next) mutable -> void {
          if (DEBUG >= 3) { cerr << "     next: "; print_hypothesis(info, next); }
          num_expansions++;

          size_t next_stack = is_final_hypothesis(info, next) ? GoalStack : get_stack_id(next);
          assert(next_stack >= cur_stack);

          if ((info->train_laso>0) && next->follows_optimal_path) {
            if (DEBUG >= 5) cerr << "found hypothesis that follows optimal path (laso) for stack " << next_stack << endl;
            StackOptimum[next_stack] = next;
          }

          //cerr << "cur_stack=" << cur_stack << "\tnext_stack=" << next_stack << endl;
          if (next_stack == cur_stack) {
            add_to_hyp_stack(info, Stacks[cur_stack], next);
          } else { // different stack
            if (Stacks[next_stack] == NULL) {
              Stacks[next_stack] = new hyp_stack();
              initialize_hyp_stack(Stacks[next_stack]);
              NextStacks.push(next_stack);
            }
            add_to_hyp_stack(info, Stacks[next_stack], next);
          }
        },
        prune_if_gt);

      if (info->forced_decode) {
        assert(num_expansions <= 1);
        assert(num_expansions >= 1);
        while ((info->forced_op_posn < info->forced_op_seq.size()) &&
               is_movement_operator(info->forced_op_seq[info->forced_op_posn]))
          info->forced_op_posn++;
        info->forced_op_posn++;
      }

      if ((DEBUG >= 5) && (num_expansions == 0)) cerr << "oops -- no expansions!!!" << endl;
      it = Stacks[cur_stack]->Stack.begin() + diff;
      ++it;
    }
    //cerr<<"("<<num_processed<<")";
    Stacks[cur_stack]->Stack.clear();


    if ((NextStacks.size() == 1) && info->recomb_buckets)
      for (auto vec = info->recomb_buckets->begin(); vec != info->recomb_buckets->end(); ++vec)
        (*vec).clear();

    if (NextStacks.size() > 0) {
      size_t next_stack = NextStacks.top();
      if (DEBUG >= 3) cerr<<"next_stack="<<next_stack<<": ["<<(Stacks[next_stack]->Stack.size())<<"-"<<Stacks[next_stack]->num_marked_skippable<<"="<<(Stacks[next_stack]->Stack.size()-Stacks[next_stack]->num_marked_skippable);
      shrink_hyp_stack(info, Stacks[next_stack], true);
      if (DEBUG >= 3) cerr<<" / after shrinking: "<<(Stacks[next_stack]->Stack.size())<<"-"<<Stacks[next_stack]->num_marked_skippable<<"="<<(Stacks[next_stack]->Stack.size()-Stacks[next_stack]->num_marked_skippable)<<"]"<<endl;
    }

    if ((info->train_laso>0)) {
      assert(NextStacks.size() <= 1);
      size_t next_stack = NextStacks.empty() ? GoalStack : NextStacks.top();
      if (DEBUG >= 5) cerr << "train_laso looking for hypothesis on " << next_stack << endl;

      hypothesis* opt = StackOptimum[next_stack];
      assert(opt != NULL);

      if (opt->pruned || opt->recombined) { // oops!
        if (DEBUG >= 1) { cerr << "train_laso making an update on " << next_stack << endl << "  opt = "; print_hypothesis(info, opt); }

        make_laso_update(info, opt, Stacks[next_stack], 0.1);
        
        for (hypothesis* hold : Stacks[next_stack]->Stack) {
          hold->pruned = true;
          if (hold->recomb_friends)
            hold->recomb_friends->clear();
        }
          
        opt->pruned = false; // TODO: what if it gets recombined???
        opt->recombined = false;
        if (opt->recomb_friends)
          opt->recomb_friends->clear();


        Stacks[next_stack]->Stack.clear();
        Stacks[next_stack]->Stack.push_back(opt);
        Stacks[next_stack]->lowest_cost  = opt->cost;
        Stacks[next_stack]->highest_cost = opt->cost;
        Stacks[next_stack]->prune_if_gt  = FLT_MAX;
        Stacks[next_stack]->num_marked_skippable = 0;
      }

      while ((info->forced_op_posn < info->forced_op_seq.size()) &&
             is_movement_operator(info->forced_op_seq[info->forced_op_posn]))
        info->forced_op_posn++;
      info->forced_op_posn++;
    }
  }


  if (info->recomb_buckets)
    for (auto vec = info->recomb_buckets->begin(); vec != info->recomb_buckets->end(); ++vec)
      (*vec).clear();

  for (auto it : Stacks) 
    if (it.first != GoalStack)
      delete it.second;

  // for (hypothesis *h : Stacks[GoalStack]->Stack) {
  //   float c = 0.;
  //   for (size_t i=0; i<W_MAX_ID; i++) {
  //     float thisc = 0.;
  //     for (hypothesis *me=h; me!=NULL; me=me->prev) {
  //       thisc += me->W[i];
  //     }
  //     cerr << thisc << " * " << info->W[i] << "\t";
  //     c += thisc * info->W[i];
  //   }
  //   cerr << "= " << c << " vs " << h->cost << ", diff = " << fabs(c-h->cost) << endl;

  //   for (hypothesis *me=h; me!=NULL; me=me->prev) cerr << "\t" << OP_NAMES[me->last_op]; cerr << endl;
  //   for (hypothesis *me=h; me!=NULL; me=me->prev) cerr << "\t" << me->W[0]; cerr << endl;
  //   for (hypothesis *me=h; me!=NULL; me=me->prev) cerr << "\t" << me->cost; cerr << endl;
  //   assert( fabs(c - h->cost) < 1e-3 );
  // }

  return Stacks[GoalStack]->Stack;
}

/*
vector<hypothesis*> stack_search(translation_info *info, vector<operation> *force_decode_op_seq = NULL) {
  hypothesis *h0 = get_next_hypothesis(info, NULL);

  stack<hypothesis*> S;
  vector<hypothesis*> Goals;
  size_t decode_step = 0;

  S.push(h0);
  
  while (!S.empty()) {
    if (force_decode_op_seq)
      assert(S.size() == 1);

    hypothesis *h = S.top(); S.pop();
    // cout<<endl<< "pop\t";
    // cout << "op=" << OP_NAMES[(uint32_t)((*force_decode_op_seq)[decode_step]).op] << "\targ=(" << ((*force_decode_op_seq)[decode_step]).arg1 << ", " << ((*force_decode_op_seq)[decode_step]).arg2 << ")" << endl;
    // print_hypothesis(info, h);

    expand_one_step(info, h, [info, &Goals, &S, force_decode_op_seq, decode_step](hypothesis*next) mutable -> void {
        if ((force_decode_op_seq == NULL) ||
            (force_decode_allowed(info, *force_decode_op_seq, next, decode_step))) {
          if (is_final_hypothesis(info, next))
            Goals.push_back( add_final_lm_scores(info, next) );
          else
            S.push(next);
          // cout << "   added: "; print_hypothesis(info, next);
        } else {
          // cout << "rejected: "; print_hypothesis(info, next);
        }
      });
    decode_step++;
  }

  return Goals;
}
*/

void mtu_add_unit(mtu_item_dict &dict, mtu_item*mtu) {
  dict[mtu->src[0]].push_back(mtu);
}

void mtu_add_item_string(mtu_item_dict &dict, mtuid ident, string src, string tgt) {
  mtu_item *mtu = (mtu_item*)calloc(1, sizeof(mtu_item));
  mtu->src_len = src.length();
  posn j = 0;
  for (uint32_t n=0; n<mtu->src_len; ++n) {
    if (src[n] == '_')
      allow_gap_after(mtu->gap_option, j-1);
    else {
      mtu->src[j] = src[n];
      j++;
    }
  }

  mtu->tgt_len = tgt.length();
  for (uint32_t n=0; n<mtu->tgt_len; ++n) {
    mtu->tgt[n] = tgt[n];
  }

  mtu->ident = ident;

  mtu_add_unit(dict, mtu);
}

void free_dict(mtu_item_dict &dict) {
  for (auto it=dict.begin(); it!=dict.end(); ++it) {
    vector<mtu_item*> &mtus = (*it).second;
    for (mtu_item*&mtu_it : mtus) //=mtus.begin(); mtu_it!=mtus.end(); ++mtu_it)
      free(mtu_it);
    mtus.clear();
  }
  dict.clear();  
}

float simple_compute_cost(void*info_, hypothesis*h) {
  //translation_info *info = (translation_info*)info_;
  return h->prev->cost + 1.f;
}

bool is_unaligned(vector< vector<posn> > A, posn j) {
  for (auto &vec : A)
    for (auto &p : vec)
      if (p == j)
        return false;
  return true;
}

bool find_mtu_in_dict(mtu_item_dict *dict, vector<lexeme> e, vector<lexeme> f, vector<posn> a, lexeme &identity, mtu_item**found_mtu=NULL) {
  // cout << "find_mtu:"<<endl;
  // cout << " e:"; for (auto l : e) cout << " " << l; cout << endl; 
  // cout << " f:"; for (auto l : a) cout << " " << f[l]; cout << endl; 
  // cout << " a:"; for (auto l : a) cout << " " << (short)l; cout << endl; 

  auto it = dict->find(f[a[0]]);
  if (it == dict->end()) {
    // cout << "    FAIL: it == dict->end" << endl;
    return false;
  }

  vector<mtu_item*> mtus = it->second;
  posn e_len = e.size();
  posn f_len = a.size();
  for (auto mtu : mtus) {
    //cout << "        "; print_mtu_half(mtu->src, mtu->src_len, mtu->gap_option); cout << " | "; print_mtu_half(mtu->tgt, mtu->tgt_len, 0); cout << endl;
    // check to see if this matches src=f[a] and tgt=e
    if ((f_len != mtu->src_len) || (e_len != mtu->tgt_len))
      continue;

    bool ok = true;
    for (posn i=1; i<f_len; ++i)
      if (f[a[i]] != mtu->src[i]) {
        ok = false;
        break;
      }
    if (!ok) continue;

    for (posn i=0; i<e_len; ++i)
      if (e[i] != mtu->tgt[i]) {
        ok = false;
        break;
      }
    if (!ok) continue;

    for (posn i=0; i<f_len-1; ++i) {
      if ((a[i]+1 != a[i+1]) && // there is a gap after i
          (! gap_allowed_after(mtu->gap_option, i))) {
        //cout << "    gap error on " << (uint32_t)i << endl;
        ok = false;
        break;
      }
    }
    if (!ok) break; // if the only thing that failed is gap, then there's no hope anyone else will match
    
    //cout << "    SUCCESS!"<<endl;
    identity = mtu->ident;
    if (found_mtu != NULL)
      *found_mtu = mtu;
    return true;
  }

  //cout << "    FAIL" << endl;
  return false;
}

posn filter_gap_width(posn j, bitset<MAX_SENTENCE_LENGTH> fcov, posn init_gap_width) {
  for (posn real_gap_width=1; real_gap_width<=init_gap_width; ++real_gap_width) {
    if (j + real_gap_width >= MAX_SENTENCE_LENGTH) {
      //cout << "filter_gap_width: MAX_SENTENCE_LENGTH case"<<endl;
      return MAX_SENTENCE_LENGTH-j;
    }
    else if (fcov[j+real_gap_width]) {
      //cout << "filter_gap_width: fcov["<<j<<"+"<<real_gap_width<<"] is true"<<endl;
      return real_gap_width;
    }
  }
  //cout << "filter_gap_width: returning init_gap_width="<<init_gap_width<<endl;
  return init_gap_width;
}

// TODO: don't allow predict_forced if alignments don't exist!

vector<operation> get_operation_sequence(unordered_map<lexeme,lexeme> vocab_match, aligned_sentence_pair data, mtu_item_dict *dict, set<mtuid> *keep_mtus) {
  auto f = data.F;
  auto E = data.E;
  auto A = data.A;
  vector<operation> op_seq;
  set<posn> gaps;

  if (dict != NULL) { // remove any alignments that are not mtus in the dictionary
    for (posn i=0; i<A.size(); ++i) {
      lexeme identity = 0;
      if (! find_mtu_in_dict(dict, E[i], f, A[i], identity))
        A[i].clear();
    }
  }

  posn i,j,j2,k,N,Z;
  N = E.size();
  i = 0; j = 0; k = 0;
  
  bitset<MAX_SENTENCE_LENGTH> fcov;

  while ((j < f.size()) && is_unaligned(A, j)) {
    op_seq.push_back( { OP_GEN_S, f[j], j } );
    fcov[j] = true;
    j++;
  }
  Z = j;

  while (i < N) {
    while (A[i].size() == 0) i++;
    if (i >= N) break;
    j2 = A[i][k];
    //cout << "i=" << (uint32_t)i << " k=" << (uint32_t)k << " j2=" << (uint32_t)j2 << endl;
    if (j < j2) {
      if (! fcov[j]) {
        //cout << "gap k=" << (uint32_t)k << " A.size=" << (uint32_t)A[i].size() << endl;
        if ((k > 0) && (k < A[i].size()))
          op_seq.push_back( { OP_CONT_GAP, 0, j } );
        else
          op_seq.push_back( { OP_GAP, 0, j } );
        gaps.insert(j);
      }

      bool any_intervening = false;
      for (posn t=j; t<j2; t++)
        if (fcov[t]) {
          any_intervening = true;
          break;
        }

      if ((j == Z) || !any_intervening)
        j = j2;
      else {
        op_seq.push_back( { OP_JUMP_E, 0, j } );
        j = Z;
      }
    }

    if (j2 < j) {
      if ((j < Z) && !fcov[j]) {
        if ((k > 0) && (k < A[i].size()))
          op_seq.push_back( { OP_CONT_GAP, 0, j } );
        else
          op_seq.push_back( { OP_GAP, 0, j } );
        gaps.insert(j);
      }

      posn W = 0;
      bool found = false;
      posn foundPos = 0;
      for (auto &pos : gaps) {   // assuming this goes in sorted order!
        //cout << "auto pos: pos="<<(short)pos<<" desired="<<(short)A[i][k]<<" found="<<found<<" W="<<(short)W<<endl;
        if (pos == A[i][k]) {
          found = true;
          foundPos = pos;
          continue;
        }
        if (pos <= A[i][k]) {
          bool any_middle_covered = false;
          if (! found)  // if we've already found one, this one will be fine too
            for (posn m=pos; m<=A[i][k]; ++m)
              if (fcov[m]) {
                any_middle_covered = true;
                break;
              }
          if (! any_middle_covered) {
            found = true;
            foundPos = pos;
          }
          continue;
        }
        if (found)
          W++;
      }
      assert(found);
      //cout << "OP_JUMP_B: j="<<(short)j<<" i="<<(short)i<<" k="<<(short)k<<" len(A[i])="<<A[i].size()<<" A[i][k]="<<(short)A[i][k]<<", W="<<(short)W<<" found="<<found<<" foundPos="<<(short)foundPos<<endl;
      if (j < Z)
        op_seq.push_back( { OP_JUMP_E, 0, j } );

      gaps.erase(A[i][k]);
      op_seq.push_back( { OP_JUMP_B, W, j } );
      j = foundPos;
    }

    if (j < j2) {
      if ((k > 0) && (k < A[i].size()))
        op_seq.push_back( { OP_CONT_GAP, 2, j } );
      else
        op_seq.push_back( { OP_GAP, 2, j } );
      gaps.insert(j);
      j = j2;
    }
    if (k == 0) {
      if (dict == NULL)
        op_seq.push_back( { OP_GEN_ST, i, j } );
      else {
        lexeme identity = 0;
        mtu_item * found_mtu;
        assert(find_mtu_in_dict(dict, E[i], f, A[i], identity, &found_mtu));
        //cout << "GEN_ST i="<<i<<" identity="<<identity<<" and f[j]="<<f[j]<<endl;
        if ((A[i].size() == 1) && (E[i].size() == 1) && (found_mtu->tr_freq <= 1) &&
            (E[i][0] > 0) && (get_copy_id(vocab_match, f[A[i][0]] ) == E[i][0])) {
          //cerr << "op_copy @ f["<<A[i][0]<<"]="<<f[A[i][0]]<<" e="<<E[i][0]<<endl;
          op_seq.push_back( { OP_GEN_ST, (mtuid)-1, j });
        } else {
          if (keep_mtus != NULL) keep_mtus->insert(identity);
          op_seq.push_back( { OP_GEN_ST, identity, j } );
        }
      }
      fcov[j] = true;
    } else {
      //cout<<"OP_CONT_WORD with k="<<k<<" i="<<i<<" len(A[i])="<<A[i].size()<<endl;
      op_seq.push_back( { OP_CONT_WORD, 0, j } );
      fcov[j] = true;
    }
    j ++;
    k ++;

    while ((j < f.size()) && is_unaligned(A, j)) {
      op_seq.push_back( { OP_GEN_S, f[j], j } );
      fcov[j] = true;
      j++;
    }
    if (Z < j)
      Z = j;
    if (k == A[i].size()) {
      i ++;
      k = 0;
    }
  }
  if (j < Z) {
    op_seq.push_back( { OP_JUMP_E, 0, j } );
    j = Z;
  }

  return op_seq;
}

template<class T>
bool empty_intersection(set<T> bigger, set<T> smaller) {
  auto e = bigger.end();
  for (T x : smaller)
    if (bigger.find(x) != e)
      return false;
  return true;
}

bool is_sorted(vector<posn> al) {
  for (posn i=1; i<al.size(); ++i)
    if (al[i-1] > al[i])
      return false;
  return true;
}

bool read_mtu_item_from_file(FILE* fd, mtu_item& mtu) {
  char line[255];
  size_t nr;

  nr = fscanf(fd, "%d\t", &mtu.ident);
  //cerr<<"ident="<<mtu.ident<<", nr="<<nr<<endl;
  if ((nr == 0) || (feof(fd))) return true;

  nr = fscanf(fd, "%zd %zd\t", &mtu.tr_doc_freq, &mtu.tr_freq);
  assert(nr == 2);

  mtu.gap_option = 0;
  posn i=0;
  while (true) {
    nr = fscanf(fd, "%[^| ] ", line);
    //cerr << "first: line='"<<line<<"', nr="<<nr<<", gap_option="<<mtu.gap_option<<endl;
    if (nr == 0) break;
    if ((line[0] == '_') && (line[1] == 0))
      allow_gap_after(mtu.gap_option, i-1);
    else {
      mtu.src[i] = atoi(line);
      i++;
    }
  }
  mtu.src_len = i;
  if (feof(fd)) return true;
  nr = fscanf(fd, "|");
  if (feof(fd)) return true;

  i=0;
  while (true) {
    nr = fscanf(fd, "%d%[ \n]", &mtu.tgt[i], line);
    //cerr<<"tgt["<<i<<"]="<<mtu.tgt[i]<<", nr="<<nr<<endl;
    i++;
    if ((line[0]=='\n') || (nr < 2) || (feof(fd))) break;
  }
  mtu.tgt_len = i;
  if (i == 0) return true;

  return false;
}

bool gt_mtu_termfreq(mtu_item *a, mtu_item *b) {
  return a->tr_freq > b->tr_freq;
}

mtu_item_dict read_mtu_item_dict(FILE *fd) {
  mtu_item_dict dict;
  bool done = false;
  while (! done) {
    mtu_item * mtu = (mtu_item*) calloc(1, sizeof(mtu_item));
    done = read_mtu_item_from_file(fd, *mtu);
    if (!done) 
      mtu_add_unit(dict, mtu);
    else
      free(mtu);
  }

  for (auto &word_list : dict)
    sort(word_list.second.begin(), word_list.second.end(), gt_mtu_termfreq);

  return dict;
}


void set_info_from_string(char*p, char*otherstring, translation_info *info, bool warnOnUnknown=false) {
  int nr = 0;
  // TODO: operation_allowed
  if (sscanf(p, "pruning_coefficient=%g", &info->pruning_coefficient)) return;
  if (sscanf(p, "max_bucket_size=%zd", &info->max_bucket_size)) return;
  if (sscanf(p, "max_gaps=%zd", &info->max_gaps)) return;
  if (sscanf(p, "max_gap_width=%zd", &info->max_gap_width)) return;
  if (sscanf(p, "max_phrase_len=%zd", &info->max_phrase_len)) return;
  if (sscanf(p, "num_kbest_predictions=%zd", &info->num_kbest_predictions)) return;
  if (sscanf(p, "max_mtus_per_token=%zd", &info->max_mtus_per_token)) return;
  if (sscanf(p, "allow_copy=%d", &nr) > 0) { info->allow_copy = nr; return; }
  if (sscanf(p, "debug_level=%d", &nr) > 0) { info->debug_level = nr; return; }
  if (sscanf(p, "train_laso=%d",&nr) > 0) { info->train_laso = nr; return; }
  if (strcmp(p, "forced_decode")==0) { info->forced_decode = true; return; }

  if (sscanf(p, "w_ALL=%g,%g,%g,%g,%g,%g", &info->W[0], &info->W[1], &info->W[2], &info->W[3], &info->W[4], &info->W[5])) return;  // TODO: make this work with arbitrary # weight ids

  nr = sscanf(p, "vocab_match=%a[^ \t\n]", &otherstring);
  if (nr > 0) {
    info->vocab_match.clear();
    cerr << "reading vocab_match from " << otherstring << endl;
    info->vocab_match = read_vocab_match(otherstring);
    return;
  }

  if (sscanf(p, "w_LM %g", &info->W[W_LM])) return;
  if (sscanf(p, "w_TM %g", &info->W[W_TM])) return;
  if (sscanf(p, "w_GEN_S %g", &info->W[W_GEN_S])) return;
  if (sscanf(p, "w_GAP %g", &info->W[W_GAP])) return;
  if (sscanf(p, "w_BREV %g", &info->W[W_BREV])) return;
  if (sscanf(p, "w_COPY %g", &info->W[W_COPY])) return;

  if (sscanf(p, "w_LM=%g", &info->W[W_LM])) return;
  if (sscanf(p, "w_TM=%g", &info->W[W_TM])) return;
  if (sscanf(p, "w_GEN_S=%g", &info->W[W_GEN_S])) return;
  if (sscanf(p, "w_GAP=%g", &info->W[W_GAP])) return;
  if (sscanf(p, "w_BREV=%g", &info->W[W_BREV])) return;
  if (sscanf(p, "w_COPY=%g", &info->W[W_COPY])) return;

  nr = sscanf(p, "lm_tgt_bin=%a[^ \t\n]", &otherstring);
  if (nr > 0) { 
    if (info->language_model != NULL) delete info->language_model;
    cerr << "reading language model from " << otherstring << endl;
    //info->language_model = new lm::ngram::Model(otherstring);
    info->language_model = new lm::ngram::ProbingModel(otherstring);
    return;
  }

  nr = sscanf(p, "tm_bin=%a[^ \t\n]", &otherstring);
  if (nr > 0) { 
    if (info->opseq_model != NULL) delete info->opseq_model;
    cerr << "reading opseq model from " << otherstring << endl;
    //info->opseq_model = new lm::ngram::Model(otherstring);
    info->opseq_model = new lm::ngram::ProbingModel(otherstring);
    return;
  }
    
  nr = sscanf(p, "mtus=%a[^ \t\n]", &otherstring);
  if (nr > 0) { 
    cerr << "reading mtus from " << otherstring << endl;
    FILE *fd = fopen(otherstring, "r");
    if (fd == 0) { cerr << "error: cannot open mtu file for reading: '" << otherstring << "'" << endl; throw exception(); }
    info->mtu_dict = read_mtu_item_dict(fd);
    fclose(fd);
    return;
  }

  if (warnOnUnknown)
    cerr << "warning: unknown parameter setting '" << p << "'" << endl;
}

void read_ini_file(char* fname, translation_info *info) {
  FILE*f = fopen(fname, "r");
  char *inbuf, *otherstring;
  size_t size = 1024;

  inbuf = (char*)calloc(size, sizeof(char));
  otherstring = (char*)calloc(size, sizeof(char));

  while (!feof(f)) {
    if (getline(&inbuf, &size, f) < 0) break;
    if ((strlen(inbuf)==0) || (inbuf[0] == '#')) continue;
    char*p = inbuf;
    while ((*p == ' ') || (*p == '\t')) { ++p; }
    if (*p == 0) continue;

    set_info_from_string(p, otherstring, info);
  }

  free(inbuf);
  free(otherstring);

  fclose(f);
}


void collect_mtus(size_t max_phrase_len, bool max_discontig, aligned_sentence_pair spair, unordered_map< mtu_item, mtu_item_info > &cur_mtus, size_t &skipped_for_len, size_t &skipped_for_discontig) {
  auto E = spair.E;
  auto F = spair.F;
  auto A = spair.A;

  //cout << "F ="; for (auto j : F) cout << " " << (uint32_t)j; cout << endl;

  assert(E.size() == A.size());
  for (posn i=0; i<E.size(); ++i) {
    vector<posn> al = A[i];
    vector<lexeme> ephr = E[i];

    //cout << "al ="; for (auto j : al) cout << " " << (uint32_t)j; cout << endl;

    assert(is_sorted(al));

    if (ephr.size() > max_phrase_len) { skipped_for_len++; continue; }
    if (al.size()   > max_phrase_len) { skipped_for_len++; continue; }

    for (posn j=0; j<al.size()-1; ++j)
      if (al[j+1] > al[j]+1 + max_discontig) {
        skipped_for_discontig++;
        continue;
      }

    mtu_item mtu;
    memset(&mtu, 0, sizeof(mtu));
    mtu.tgt_len = ephr.size();
    for (posn j=0; j<ephr.size(); ++j)
      mtu.tgt[j] = ephr[j];

    mtu.src_len = al.size();
    uint32_t my_gaps = 0;
    for (posn j=0; j<al.size(); ++j) {
      mtu.src[j] = F[al[j]];

      if ((j < al.size()-1) && (al[j] != al[j+1]-1)) {
        //cout << "sgo " << (uint32_t)j << endl;
        allow_gap_after(my_gaps, j);
      }
    }
    mtu.ident = 0;

    //cout << "extracting: ";print_mtu_half(mtu.src, mtu.src_len, 0); cout << " | "; print_mtu_half(mtu.tgt, mtu.tgt_len, 0); cout << ", my_gaps=" << my_gaps << ", "; print_mtu_half(mtu.src, mtu.src_len, my_gaps); cout << endl;
    //cout << "  al ="; for (auto l : al) cout << " " << (uint32_t)l; cout << endl;
    
    auto it = cur_mtus.find(mtu);
    if (it == cur_mtus.end())
      cur_mtus.insert( { mtu, { 1, 1, my_gaps } } );
    else {
      mtu_item_info & old_val = cur_mtus[mtu];
      old_val.token_freq ++;
      old_val.gap |= my_gaps;
    }
  }
}

bool read_next_aligned_sentence(translation_info*info, FILE *fd, aligned_sentence_pair &ret, bool &is_new_document) {
  lexeme cnt, w,x;
  size_t nr;

  is_new_document = false;

 NEXT_LINE:

  if (feof(fd)) return false;

  nr = fscanf(fd, "%d", &cnt);
  assert(nr == 1);

  if (cnt == 0) {
    // this is a marker for a new document (just a single line containing "0")
    is_new_document = true;
    goto NEXT_LINE;
  }

  bool skip_me = false;

  if (cnt > MAX_SENTENCE_LENGTH) skip_me = true;
  vector<lexeme> F(cnt);
  for (posn i=0; i<cnt; ++i) nr = fscanf(fd, " %d", &F[i]);

  nr = fscanf(fd, "\t%d", &cnt);
  if (cnt > MAX_SENTENCE_LENGTH) skip_me = true;
  vector<lexeme> e(cnt);
  for (posn i=0; i<cnt; ++i) nr = fscanf(fd, " %d", &e[i]);

  nr = fscanf(fd, "\t%d", &cnt);
  vector< set<posn> > al;  // maps from english id to (set of) french ids
  for (posn i=0; i<e.size(); ++i)
    al.push_back(set<posn>());
  for (posn i=0; i<cnt; ++i) {
    nr = fscanf(fd, " %d-%d", &w, &x);
    al[x].insert(w);
  }
  nr = fscanf(fd, "\n");

  if (skip_me)
    goto NEXT_LINE;

  compute_bleu_stats(e, &info->bleu_total_stats);
  for (posn i=0; i<4; ++i)
    info->bleu_ref_counts[i] += info->bleu_total_stats.ng_counts[i];

  vector< vector<lexeme> > E;
  vector< vector<posn> > A;

  posn i = 0;
  while (i < e.size()) {
    // the english phrase starts at i -- find it's end
    set<posn> curF;
    curF.insert( al[i].begin(), al[i].end() );    // all french ids to which this phrase is aligned
    vector<lexeme> thisP;
    thisP.push_back(e[i]);
    posn j = i + 1;
    while (j < e.size()) {
      if (empty_intersection(curF, al[j]))
        break;
      curF.insert( al[j].begin(), al[j].end() );
      thisP.push_back(e[j]);
      j++;
    }
    
    E.push_back(thisP);
    A.push_back(vector<posn>( curF.begin(), curF.end() ));
    i = j;
  }

  //cout << "F ="; for (auto f : F) cout << " " << (uint32_t)f; cout << endl;
  //cout << "E ="; for (auto vec : E) { for (auto w : vec) cout << " " << w; cout << " |"; } cout << endl;
  //cout << "A ="; for (auto vec : A) { for (auto p : vec) cout << " " << (uint32_t)p; cout << " |"; } cout << endl;

  info->total_sentence_count++;
  info->total_word_count += F.size();

  if ((DEBUG >= 1) || info->total_sentence_count == info->next_sentence_print) { 
    cerr << "processing sentence pair " << info->total_sentence_count << " (" << info->total_word_count << " words)" << endl; 
    info->next_sentence_print *= 2; 
  }

  ret.F = F; ret.E = E; ret.A = A;
  return true;
}


void test_align() {
  vector< lexeme >         F = { 'D', 'H', 'E', 'I', 'B', 'G' };
  vector< vector<lexeme> > E = { { 't' }, { 'h' }, { 'r' }, { 'a' }, { 'b' } };
  vector< vector<posn  > > A = { { 0 }, { 2 }, { 1, 5 }, { 3 }, { 4 } };
  unordered_map<lexeme,lexeme> vocab_match;

  auto op_seq = get_operation_sequence( vocab_match, {F, E, A}, NULL, NULL );

  for (auto &op : op_seq)
    cout << "op=" << OP_NAMES[(uint32_t)op.op] << "\targ=(" << op.arg1 << ", " << op.arg2 << ")" << endl;

}

void test_decode() {
  mtu_item_dict dict;
  mtu_add_item_string(dict, 0, "A_B", "ab");
  mtu_add_item_string(dict, 1, "A_C", "ac");
  mtu_add_item_string(dict, 2, "A" , "a");
  mtu_add_item_string(dict, 3, "B" , "b");
  mtu_add_item_string(dict, 4, "B_C", "bc");
  mtu_add_item_string(dict, 5, "C" , "c");

  translation_info info;
  memset(&info, 0, sizeof(translation_info));
  initialize_translation_info(info);
  info.mtu_dict = dict;

  info.N       = 6;
  info.sent[0] = (uint32_t)'A';
  info.sent[1] = (uint32_t)'B';
  info.sent[2] = (uint32_t)'C';
  info.sent[3] = (uint32_t)'A';
  info.sent[4] = (uint32_t)'B';
  info.sent[5] = (uint32_t)'C';
  build_sentence_mtus(&info);
  info.compute_cost = simple_compute_cost;  // TODO: this doesn't actually do anything!
  info.language_model = NULL; // new lm::ngram::Model((char*)"file.arpa-bin");
  info.max_gaps = 5;
  info.max_gap_width = 5;
  info.max_phrase_len = 5;
  info.num_kbest_predictions = 100;

  info.operation_allowed = (1 << OP_MAXIMUM) - 1;  // all operations

  info.pruning_coefficient = 0.;
  //delete info.recomb_buckets; info.recomb_buckets = NULL;
  vector<operation> op_seq;

  for (size_t rep = 0; rep < 1 + 0*999; ++rep) {
    //cerr<<".";
    info.hyp_ring = initialize_ring<hypothesis>(INIT_HYPOTHESIS_RING_SIZE);

    vector<hypothesis*> Goals = 
      stack_generic_search(&info, 
                           [](hypothesis* hyp) { 
                             return (size_t)hyp->cov_vec_count * 2 + ((hyp->n != hyp->Z) || (!is_covered(hyp, hyp->n)));
                             //return (size_t)hyp->cov_vec_count; 
                           },
                           info.N*2);


    vector<astar_result> results = astar_kbest_search(&info, Goals);
    // for (auto pair : results) {
    //   cout << pair.first << "\t[";
    //   for (lexeme w : pair.second) {
    //     cout << w << " ";
    //     //cout << OP_CHAR[(short)op.op] << op.arg1 << " ";
    //   }
    //   cout << "]"<<endl;
    // }

    /*
    for (auto &hyp : GoalsVisited.first) {
      vector<lexeme> trans = get_translation(&info, hyp);

      cout<<hyp->cost<<"\t";
      for (auto &w : get_translation(&info, hyp))
        cout<<" "<<w;
      cout<<endl;
    }
    */

    free_ring(info.hyp_ring, free_one_hypothesis);
    free_ring(info.lm_state_ring);
    free_ring(info.tm_state_ring);
  }
  cerr<<endl;
  if (info.language_model != NULL) delete info.language_model;
  if (info.recomb_buckets) delete info.recomb_buckets;
  //if (info.vocab_dictionary) delete info.vocab_dictionary;
  free_sentence_mtus(info.mtus_at);
  free_dict(dict);
}

void test_lm() {
  ng::Model model("file.arpa");
  ng::State state(model.BeginSentenceState()), out_state;
  const ng::Vocabulary &vocab = model.GetVocabulary();
  std::string word;
  while (std::cin >> word) {
    std::cout << model.Score(state, vocab.Index(word), out_state) << '\n';
    state = out_state;
  }
}

void generate_output(translation_info*info, FILE*out, vector<hypothesis*> Goals) {
  if (Goals.size() == 0) {
    if (info->num_kbest_predictions <= 1)
      fprintf(out, "FAIL!\n");
    return;
  }

  // find the best translation
  size_t best_hyp_id = 0;
  for (size_t id=1; id<Goals.size(); ++id) {
    if (Goals[id]->cost < Goals[best_hyp_id]->cost)
      best_hyp_id = id;
  }
  hypothesis*hyp = Goals[best_hyp_id];
  vector<lexeme> trans = get_translation(info, hyp);
  update_bleu_stats(info, trans);
  info->total_output_cost += hyp->cost;

  if (info->num_kbest_predictions <= 1) {  // one-best output
    // print it ... TODO add options to get statistics
    for (size_t i=0; i<trans.size(); i++) {
      if (i > 0) fprintf(out, " ");
      fprintf(out, "%u", trans[i]);
    }
    fprintf(out, "\n");

    if (DEBUG >= 4) {
      for (hypothesis*hyp0 : Goals) {
        vector<lexeme> tt = get_translation(info,hyp0);
        cerr << "t=";
        for (lexeme l : tt) cerr << " " << l;
        cerr<<endl;
        float sumlm = 0;
        for (hypothesis*me=hyp0; me!=NULL; me=me->prev) {
          cerr << "from: ";
          sumlm += me->W[W_LM];
          cerr << "[lm=" << sumlm << "] ";
          print_hypothesis(info, me);
        }
        cerr << endl;
      }
    }
  } else {  // k-best output, in zmert format
    vector<astar_result> results = astar_kbest_search(info, Goals);
    for (astar_result res : results) {
      fprintf(out, "%zd |||", info->total_sentence_count-1);
      for (lexeme w : res.trans)
        fprintf(out, " %u", w);
      fprintf(out, " |||");
      for (size_t i=0; i<W_MAX_ID; i++)
        fprintf(out, " %g", res.W[i]);
      fprintf(out, " ||| %g\n", res.cost);
    }
    
  }
}
  
void add_all_mtus(unordered_map<mtu_item, mtu_item_info> &dst, 
                  unordered_map<mtu_item, mtu_item_info>  src) {
  for (auto item_pair : src) {
    mtu_item mtu = item_pair.first;
    size_t   cnt = item_pair.second.token_freq;
    gap_op_t gap = item_pair.second.gap;

    auto it = dst.find(mtu);
    if (it == dst.end())
      dst.insert( { mtu, { 1, cnt, gap } } );
    else {
      mtu_item_info& old_val = dst[mtu];
      old_val.doc_freq++;
      old_val.token_freq += cnt;
      old_val.gap |= gap;
    }
  }
}


void extract(translation_info *info, int argc, char*argv[]) {
  size_t max_phrase_len = info->max_phrase_len;
  size_t max_discontig  = info->max_gap_width;

  if ((argc != 1) || (!strcmp(argv[0], "-h")) || (!strcmp(argv[0], "-help")) || (!strcmp(argv[0], "--help"))) {
    cout << "usage: ngdec extract [ngdec file] > [mtu file]" << endl;
    exit(-1);
  }

  FILE *fd = fopen(argv[0], "r");
  if (fd == 0) { cerr << "error: cannot open file for reading: '" << argv[0] << "'" << endl; throw exception(); }

  unordered_map<mtu_item, mtu_item_info> all_mtus;
  unordered_map<mtu_item, mtu_item_info> cur_doc_mtus;

  size_t skipped_for_len = 0, skipped_for_discontig = 0;
  while (!feof(fd)) {
    bool is_new_document = false;
    aligned_sentence_pair spair;
    if (!read_next_aligned_sentence(info, fd, spair, is_new_document)) break;
    if (is_new_document) {
      add_all_mtus(all_mtus, cur_doc_mtus);
      cur_doc_mtus.clear();
    }
    collect_mtus(max_phrase_len, max_discontig, spair, cur_doc_mtus, skipped_for_len, skipped_for_discontig);
  }
  add_all_mtus(all_mtus, cur_doc_mtus);
  fclose(fd);

  cerr << "collected " << all_mtus.size() << " mtus" << endl;
  cerr << "skipped " << skipped_for_len       << " for length (>"<<max_phrase_len<<")" << endl;
  cerr << "skipped " << skipped_for_discontig << " for gap size (>"<<max_discontig<<")" << endl;

  print_mtu_set(all_mtus, true);
}

void oracle(translation_info*info, int argc, char*argv[]) {
  if ((argc != 3) || (!strcmp(argv[0], "-h")) || (!strcmp(argv[0], "-help")) || (!strcmp(argv[0], "--help"))) {
    cout << "usage: ngdec oracle [mtu file] [vocab_match file] [ngdec file] > [opseq file]" << endl;
    exit(-1);
  }

  FILE *fd = fopen(argv[0], "r");
  if (fd == 0) { cerr << "error: cannot open mtu file for reading: '" << argv[0] << "'" << endl; throw exception(); }
  mtu_item_dict dict = read_mtu_item_dict(fd);
  fclose(fd);

  info->vocab_match = read_vocab_match(argv[1]);
  
  fd = fopen(argv[2], "r");
  if (fd == 0) { cerr << "error: cannot open ngdec file for reading: '" << argv[2] << "'" << endl; throw exception(); }

  while (!feof(fd)) {
    bool is_new_document = false;
    aligned_sentence_pair spair;
    if (!read_next_aligned_sentence(info, fd, spair, is_new_document)) break;
    if (is_new_document) cout << endl;
    vector<operation> op_seq;

    op_seq = get_operation_sequence(info->vocab_match, spair, &dict, NULL);
    pretty_print_op_seq(op_seq);
  }

  fclose(fd);
  free_dict(dict);
}

void predict_lm(translation_info *info, int argc, char*argv[]) {
  if ((argc < 3) || (!strcmp(argv[0], "-h")) || (!strcmp(argv[0], "-help")) || (!strcmp(argv[0], "--help"))) {
    cout << "usage: ngdec predict-lm [ini file] [ngdec file|-] [output|-] (...override options)" << endl;
    exit(-1);
  }

  char* iniFile = argv[0];
  char* ngdecFile = argv[1];
  char* outputFile = argv[2];

  read_ini_file(iniFile, info);

  char*otherstring = (char*)calloc(1024, sizeof(char));
  for (int i=3; i<argc; ++i) {
    if ((strlen(argv[i]) < 3)  || (argv[i][0] != '-') || (argv[i][1] != '-'))
      { cerr << "error: unexpected option '" << argv[i] << "'" << endl; throw exception(); }
    set_info_from_string(argv[i] + 2, otherstring, info, true);
  }
  free(otherstring);

  if (need_op_sequence(info)) {
    info->max_gaps = MAX_SENTENCE_LENGTH;
    info->max_gap_width = MAX_SENTENCE_LENGTH;
  }

  FILE*in,*out;
  if (strcmp(ngdecFile, "-")==0) in = stdin;
  else {
    in = fopen(ngdecFile, "r");
    if (in == 0) { cerr << "error: cannot open " << ngdecFile << " for reading" << endl; throw exception(); }
  }

  if (info->train_laso > 1) { // make sure we can seek
    int res = fseek(in, 0, SEEK_SET);
    if (res < 0) {
      cerr << "error seeking in input stream for LaSO!" << endl;
      exit(-1);
    }
  }

  if (strcmp(outputFile, "-")==0) out = stdout;
  else {
    out = fopen(outputFile, "w");
    if (out == 0) { cerr << "error: cannot open " << outputFile << " for writing" << endl; throw exception(); }
  }

  size_t num_passes = 1;
  if (info->train_laso > 0) num_passes = info->train_laso;

  for (size_t pass=1; pass<=num_passes; pass++) {
    cerr << "translating..." << endl;
    while (!feof(in)) {
      bool is_new_document = false;
      aligned_sentence_pair spair;
      if (!read_next_aligned_sentence(info, in, spair, is_new_document)) break;
      if (is_new_document) cout << endl;

      if (need_op_sequence(info)) {
        info->forced_op_seq.clear();
        info->forced_keep_mtus.clear();
        info->forced_op_seq = get_operation_sequence(info->vocab_match, spair, &info->mtu_dict, &info->forced_keep_mtus);
        if (DEBUG >= 3) pretty_print_op_seq(info->forced_op_seq);
        info->forced_op_posn = 0;
      }

      info->N = spair.F.size();
      for (posn i=0; i<info->N; ++i)
        info->sent[i] = spair.F[i];
      build_sentence_mtus(info);

      info->hyp_ring = initialize_ring<hypothesis>(INIT_HYPOTHESIS_RING_SIZE);
      info->lm_state_ring = initialize_ring<lm::ngram::State>(INIT_HYPOTHESIS_RING_SIZE);
      info->tm_state_ring = initialize_ring<lm::ngram::State>(INIT_HYPOTHESIS_RING_SIZE);

      vector<hypothesis*> Goals =
        stack_generic_search(info, [](hypothesis* hyp) { return (size_t)hyp->cov_vec_count; }, info->N*2);

      generate_output(info, out, Goals);
    
      free_sentence_mtus(info->mtus_at);
      free_ring(info->hyp_ring, free_one_hypothesis);
      free_ring(info->lm_state_ring);
      free_ring(info->tm_state_ring);
    }

    if (info->train_laso>0) {
      cerr << "total laso updates: " << info->num_laso_updates << endl;
      cerr << "current weight vector: ";
      for (size_t i=0; i<W_MAX_ID; i++) {
        if (i > 0) cerr << ",";
        cerr << info->W[i];
      }
      cerr << endl;

      int res = fseek(in, 0, SEEK_SET);
      if (res < 0) {
        cerr << "error seeking in input stream for LaSO!" << endl;
        exit(-1);
      }
    }
  }


  if (strcmp(ngdecFile,  "-")!=0) fclose(in);
  if (strcmp(outputFile, "-")!=0) fclose(out);

  cerr << "total cost=" << info->total_output_cost << "\tbleu=";
  compute_overall_bleu(info, true);

  if (info->language_model != NULL) delete info->language_model;
  if (info->opseq_model    != NULL) delete info->opseq_model;
  if (info->recomb_buckets != NULL) delete info->recomb_buckets;
  //if (info->vocab_dictionary) delete info->vocab_dictionary;
  free_dict(info->mtu_dict);
}


void usage() {
  cout << "usage: ngdec [command] (options...)" << endl
       << endl
       << "valid commands are:" << endl
       << endl
       << "    extract          given a .ngdec file (source, target and alignments)," << endl
       << "                     generate a dictionary of MTUs" << endl
       << endl
       << "    oracle           given a .ngdec file and a dictionary, generate sequence" << endl
       << "                     of operations to match the given translation/alignment" << endl
       << "                     as well as possible" << endl
       << endl
       << "    predict-forced   perform \"force decoding\" on an aligned corpus" << endl
       << endl
       << "    predict-lm       given a trained op-seq language model, predict" << endl
       << "                     translations for 'test' sentences" << endl
       << endl;
  exit(-1);
}

namespace StolenFromKenLM {
  const char* skip_spaces(const char *at) {
    for (; *at == ' ' || *at == '\t'; ++at) {}
    return at;
  }

  float FloatSec(const struct timespec &tv) {
    return static_cast<float>(tv.tv_sec) + (static_cast<float>(tv.tv_nsec) / 1000000000.0);
  }
  float FloatSec(const struct timeval &tv) {
    return static_cast<float>(tv.tv_sec) + (static_cast<float>(tv.tv_usec) / 1000000.0);
  }
  void print_system_usage(translation_info *info, timespec started_timespec) {
    std::ifstream status("/proc/self/status", std::ios::in);
    string header, value;
    cerr<<"memory:";
    while ((status >> header) && getline(status, value)){
      if ((header == "VmPeak:") || (header == "VmRSS:"))
        cerr << "\t" << header << " " << skip_spaces(value.c_str());
    }
    cerr << endl;

    cerr << "time  :";

    struct timespec current_timespec;
    clock_gettime(CLOCK_MONOTONIC, &current_timespec);
    float total = FloatSec(current_timespec) - FloatSec(started_timespec);
    float sent_per_sec = static_cast<float>(info->total_sentence_count) / total;
    float word_per_sec = static_cast<float>(info->total_word_count) / total;
    cerr << "\ttotal: " << total << "s" << "\tsent/s: " << sent_per_sec << "\tword/s: " << word_per_sec << endl;
  }
}
  

int main(int argc, char*argv[]) {
  //read_ini_file("hansard/out/model.ini", &info);
  //  test_decode();  return 0;
  if (argc < 2) usage();

  timespec started_timespec;
  clock_gettime(CLOCK_MONOTONIC, &started_timespec);

  void (*cmd)(translation_info*, int, char*[]);

  if      (!strcmp(argv[1], "extract"       )) { cmd = extract; }
  else if (!strcmp(argv[1], "oracle"        )) { cmd = oracle; }
  //else if (!strcmp(argv[1], "predict-forced")) { cmd = predict_forced; }
  else if (!strcmp(argv[1], "predict-lm"    )) { cmd = predict_lm; }
  else { usage(); }
  argc -= 2;  argv += 2;

  translation_info info;
  memset(&info, 0, sizeof(translation_info));
  initialize_translation_info(info);

  cmd(&info, argc, argv);

  StolenFromKenLM::print_system_usage(&info, started_timespec);

  return 0;
}


  //test_align();
  //test_lm();
  //test_big_decode(argv[1], false);
  //main_collect_mtus(argv[1]);

  
