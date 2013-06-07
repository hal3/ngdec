#include <iostream>
#include <functional>
#include <assert.h>
#include <float.h>
#include <stack>
#include <unordered_set>
#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include "ngdec.h"
#include "string.h"
#include "lm/model.hh"

namespace ng=lm::ngram;

#define MAX(a,b) (((a)>(b))?(a):(b))

namespace std {
  template <> struct hash<mtu_item> : public unary_function<mtu_item, size_t> {
    size_t operator()(const mtu_item& v) const {
      return util::MurmurHashNative(&v, sizeof(mtu_item));
    }
  };
}

bool gap_allowed_after(gap_op_t gap_option, posn i) {
  assert(i >= 0); assert(i < 31);
  return (gap_option & (1 << i)) != 0;
}

void allow_gap_after(gap_op_t &gap_option, posn i) {
  assert(i >= 0); assert(i < 31);
  gap_option |= (1 << i);
}

void get_op_string(char op_op, mtuid op_arg1, stringstream&ss) {
  switch (op_op) {
  case OP_UNKNOWN:      ss<<'?';           break;
  case OP_INIT:         ss<<'0';           break;
  case OP_GEN_ST:       ss<<'G'<<op_arg1;  break;
  case OP_CONT_WORD:    ss<<'w';           break;
  case OP_CONT_GAP:     ss<<'g';           break;
  case OP_GEN_S:        ss<<'S'<<op_arg1;  break;
  case OP_GEN_T:        ss<<'T'<<op_arg1;  break;
  case OP_GAP:          ss<<'_';           break;
  case OP_JUMP_B:       ss<<'J'<<op_arg1;  break;
  case OP_JUMP_E:       ss<<'E';           break;
  default:              assert(false);
  }
}

void pretty_print_op_seq(vector<operation> ops) {
  bool first = true;
  stringstream ss;
  for (auto op : ops) {
    if (! first) ss<<" ";
    first = false;
    get_op_string(op.op, op.arg1, ss);
  }
  cout<<ss<<endl;
}

void print_coverage(bitset<MAX_SENTENCE_LENGTH> cov, posn N, posn cursor) {
  for (posn i=0; i<N; i++) {
    if (cov[i])
      cout << "X";
    else if (i < cursor)
      cout << "_";
    else
      cout << ".";
  }
}


void print_mtu_half(const lexeme*d, posn len, gap_op_t gap_option=0) {
  for (posn i=0; i<len; i++) {
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
      id++;
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
        len ++;
      else  // GEN_ST
        len += h->cur_mtu->mtu->tgt_len;
    }
    else if (h->last_op == OP_GEN_T) // || h->last_op == OP_COPY)
      len += 1;
    h = h->prev;
  }

  return len;
}

vector<lexeme> get_translation(translation_info *info, hypothesis *h) {
  posn len = get_translation_length(h);
  vector<lexeme> trans(len);

  posn m = len-1;
  while (h != NULL) {
    if (h->last_op == OP_GEN_ST) {
      if (h->cur_mtu->mtu->tgt_len == 0) { // COPY
        trans[m] = info->sent[h->n];
        m--;
      } else { // GEN_ST
        for (posn i=0; i<h->cur_mtu->mtu->tgt_len; i++) {
          trans[m] = h->cur_mtu->mtu->tgt[h->cur_mtu->mtu->tgt_len - 1 - i];
          m--;
        }
      }
    } else if (h->last_op == OP_GEN_T) {
      trans[m] = h->cur_mtu->mtu->tgt[0];
      m--;
    } /* else if (h->last_op == OP_COPY) {
      trans[m] = info->sent[h->n];
      m--;
      } */

    h = h->prev;
  }

  return trans;
}


void print_hypothesis(translation_info* info, hypothesis *h) {
  cerr << h;
  cerr << "\top="<<OP_NAMES[(size_t)h->last_op];
  cerr << " ["<<h->op_argument<<"]";
  cerr << "\tmtu=";
  //if (((h->last_op == OP_GEN_ST) || (h->last_op == OP_CONT_WORD) || (h->last_op == OP_CONT_GAP) /* || (h->last_op == OP_CONT_SKIP) */ ) && (h->cur_mtu != NULL)) print_mtu(*h->cur_mtu->mtu); else
  if (h->cur_mtu != NULL) print_mtu(*h->cur_mtu->mtu, true); else
  cerr<<"___";
  cerr << "\tqhd="<<(uint32_t)h->queue_head;
  cerr << "\tn="<<(uint32_t)h->n;
  cerr << "\tZ="<<(uint32_t)h->Z;
  cerr << "\tc "<<h->cov_vec<<" "<<(uint32_t)h->cov_vec_count<<"="; 
  if (info == NULL) print_coverage(*h->cov_vec, h->Z, h->Z);
  else print_coverage(*h->cov_vec, info->N, h->Z);
  cerr << "\t#g="<<h->gaps->size();
  cerr << "\tcst="<<h->cost;
  cerr << "\tprv="<<h->prev;
  cerr << endl;
}

bool is_covered(hypothesis *h, posn n) {
  return (*(h->cov_vec))[n];
}

void set_covered(hypothesis *h, posn n) {
  assert(! ((*(h->cov_vec))[n]) );
  if (! h->cov_vec_alloc) {
    assert( h->prev != NULL );
    if( h->prev->cov_vec_count != h->prev->cov_vec->count() ) {
      for (hypothesis*me=h; me!=NULL; me=me->prev) {
        print_hypothesis(NULL, me);
        cerr << me->cov_vec->to_string(' ', '*') << endl;
      }
      assert(false);
    }
    h->cov_vec = new bitset<MAX_SENTENCE_LENGTH>(*h->prev->cov_vec);

    (*h->cov_vec) ^= (*h->prev->cov_vec);
    bool cov_vec_eq = ! h->cov_vec->any();
    (*h->cov_vec) ^= (*h->prev->cov_vec);
    assert(cov_vec_eq);

    h->cov_vec_alloc = true;
  }

  if (h->cov_vec->count() != h->cov_vec_count) {
    for (hypothesis*me=h; me!=NULL; me=me->prev) {
      print_hypothesis(NULL, me);
      cerr << me->cov_vec->to_string(' ', '*') << endl;
    }
  }
  (*(h->cov_vec))[n] = true;
  h->cov_vec_hash += n * 38904831901;
  h->cov_vec_count++;
  assert( h->cov_vec_count == h->cov_vec->count() );
  assert( h->cov_vec->count() == h->prev->cov_vec->count()+1);
}

void free_hypothesis_ring(hypothesis_ring*r) {
  vector<hypothesis_ring*> to_free;
  while (r != NULL) {
    for (size_t i=0; i<r->my_size; i++) 
      if (r->my_hypotheses[i].tm_context != NULL)
        delete r->my_hypotheses[i].tm_context;
      //free(r->my_hypotheses[i].tm_context);

    for (size_t i=0; i<r->next_hypothesis; i++) {
      hypothesis *h = r->my_hypotheses + i;
      if (h->lm_context_alloc) delete h->lm_context;
      if (h->gaps_alloc)       delete h->gaps;
      if (h->cov_vec_alloc)    delete h->cov_vec;
    }
    free(r->my_hypotheses);
    to_free.push_back(r);
    r = r->previous_ring;
  }
  for (auto r : to_free)
    free(r);
}

hypothesis_ring* initialize_hypothesis_ring(translation_info*info, size_t desired_size) {
  hypothesis_ring *r = (hypothesis_ring*) calloc(1, sizeof(hypothesis_ring));
  r->my_hypotheses = (hypothesis*) calloc(desired_size, sizeof(hypothesis));
  //for (size_t i=0; i<desired_size; i++)
  //  r->my_hypotheses[i].tm_context = (mtuid*) calloc(info->tm_context_len, sizeof(mtuid));
  r->next_hypothesis = 0;
  r->my_size = desired_size;
  r->previous_ring = NULL;
  return r;
}

hypothesis* get_next_hyp_ring_element(translation_info *info) {
  if (info->hyp_ring->next_hypothesis >= info->hyp_ring->my_size) {
    // need to make more hypotheses!
    size_t new_size = info->hyp_ring->my_size * 2;
    if (new_size < info->hyp_ring->my_size)
      new_size = info->hyp_ring->my_size;
    hypothesis_ring * new_ring = initialize_hypothesis_ring(info, new_size);
    new_ring->previous_ring = info->hyp_ring;
    info->hyp_ring = new_ring;
  }

  info->hyp_ring->next_hypothesis ++;
  return info->hyp_ring->my_hypotheses + (info->hyp_ring->next_hypothesis-1);
}


hypothesis* next_hypothesis(translation_info*info, hypothesis *h) {
  hypothesis *h2 = get_next_hyp_ring_element(info);

  if (h == NULL) {  // asking for initial hypothesis
    h2->last_op = OP_INIT;
    h2->op_argument = 0;
    h2->cur_mtu = NULL;
    h2->queue_head = 0;
    h2->cov_vec = new bitset<MAX_SENTENCE_LENGTH>(); // new vector<bool>(info->N);
    h2->cov_vec_count = 0;
    h2->cov_vec_alloc = true;
    h2->cov_vec_hash = 0;
    h2->n = 0;
    h2->Z = 0;
    h2->gaps = new set<posn>();
    h2->gaps_alloc = true;
    if (info->language_model == NULL) {
      h2->lm_context = NULL;
      h2->lm_context_alloc = false;
      h2->lm_context_hash = 0;
    } else {
      h2->lm_context = new ng::State(info->language_model->BeginSentenceState());
      h2->lm_context_alloc = true;
      h2->lm_context_hash = ng::hash_value(*h2->lm_context);
    }
    if (info->opseq_model == NULL) {
      h2->tm_context = NULL;
      h2->tm_context_hash = 0;
    } else {
      h2->tm_context = new ng::State(info->opseq_model->BeginSentenceState());
      h2->tm_context_hash = ng::hash_value(*h2->tm_context);
    }
    //memset(h2->tm_context, OP_INIT, info->tm_context_len);
    //h2->tm_context_hash = util::MurmurHashNative(h2->tm_context, info->tm_context_len * sizeof(mtuid), 0);
    h2->skippable = false;
    h2->cost = 1.;
    h2->prev = NULL;
  } else {
    memcpy(h2, h, sizeof(hypothesis));
    h2->last_op = OP_UNKNOWN;
    h2->op_argument = 0;
    //h2->cov_vec = new bitset<MAX_SENTENCE_LENGTH>(*h->cov_vec);
    //h2->cov_vec_alloc = true;
    h2->cov_vec_alloc = false;
    h2->gaps_alloc = false;
      h2->lm_context = new ng::State(info->language_model->BeginSentenceState());
      h2->lm_context_alloc = true;
      h2->lm_context_hash = ng::hash_value(*h2->lm_context);
    //h2->lm_context_alloc = false;
    h2->tm_context = new ng::State(info->opseq_model->BeginSentenceState());
    h2->tm_context_hash = ng::hash_value(*h2->tm_context);
      //    h2->tm_context = NULL;
    h2->skippable = false;
    h2->prev = h;
  }

  return h2;
}

void free_sentence_mtus(vector< vector<mtu_for_sent*> > sent_mtus) {
  for (auto &it1 : sent_mtus)
    for (auto &it2 : it1)
      free(it2);
}

void build_sentence_mtus(translation_info *info, mtu_item_dict dict) {
  posn N = info->N;
  vector< vector<mtu_for_sent*> > mtus_at(N);  

  for (posn n=0; n<N; n++) {
    auto dict_entry = dict.find(info->sent[n]);
    if (dict_entry == dict.end()) continue;

    vector<mtu_item*> mtus = dict_entry->second;
    for (auto mtu_iter=mtus.begin(); mtu_iter!=mtus.end(); mtu_iter++) {
      mtu_for_sent* mtu = (mtu_for_sent*)calloc(1, sizeof(mtu_for_sent));
      mtu->mtu = *mtu_iter;

      for (posn m=0; m<info->max_phrase_len; m++)
        for (posn i=0; i<NUM_MTU_OPTS; i++)
          mtu->found_at[m][i] = MAX_SENTENCE_LENGTH+1;

      mtu->found_at[0][0] = n;
      posn last_position = n;
      bool success = true;
      for (posn i=1; i<mtu->mtu->src_len; i++) {
        posn num_found = 0;
        for (posn n2=last_position; n2<N; n2++) {
          if (info->sent[n2] == mtu->mtu->src[i]) {
            mtu->found_at[i][num_found] = n2;
            num_found++;
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

      if (success) {
        mtus_at[n].push_back(mtu);
      } else {
        free(mtu);
      }
    }
  }

  info->mtus_at = mtus_at;
}

bool is_ok_lex_position(translation_info *info, hypothesis* h, posn n, size_t i, posn here) {
  const mtu_for_sent * mtu = h->cur_mtu;
  posn queue_head = h->queue_head+1;
  mtu_item *it = mtu->mtu;

  //cout << "is_ok_lex_position: here="<<(short)here<<" n="<<(short)n<<" is_covered="<<is_covered(h,here)<<endl;

  if (here > MAX_SENTENCE_LENGTH)
    return false;

  if ((here >= n) && (! is_covered(h, here))) {
    bool can_add_here = true;
    // check to make sure it's (likely to be) "completeable"
    
    if (true) {
      posn min_pos = here;
      for (posn q_ptr=queue_head+1; q_ptr < it->src_len; q_ptr++) {
        bool found = false;
        for (size_t j=0; j<NUM_MTU_OPTS; j++) {
          posn there = mtu->found_at[q_ptr][i];
          if (there > MAX_SENTENCE_LENGTH)
            break;
          if ((there > min_pos)  && (! is_covered(h, there))) {
            found = true;
            min_pos = there;
            break;
          }
        }
        if (!found) {
          //cout << "is_ok_lex_position not found at "<<q_ptr<<" with queue_head="<<queue_head<<endl;
          can_add_here = false;
          break;
        }
      }
    }
    
    return can_add_here;
  }
  return false;
}

vector<posn> get_lex_positions(translation_info *info, hypothesis* h, posn n) {
  // mtu->mtu->src[queue_head] is a word
  // we want to know all "future" positions where this word occurs
  vector<posn> posns;
  for (size_t i=0; i<NUM_MTU_OPTS; i++) {
    posn here = h->cur_mtu->found_at[h->queue_head+1][i];
    if (is_ok_lex_position(info, h, n, i, here))
      posns.push_back(here);
  }
  return posns;
}

// returns TRUE if should be added
bool prepare_for_add(translation_info*info, hypothesis*h) {
  if ((info->operation_allowed & (1 << h->last_op)) > 0) {
    h->Z = MAX(h->Z, h->n);
    //h->cost = info->compute_cost(info, h);


    /*
    mtuid tm = h->last_op;
    switch (h->last_op) {
    case OP_JUMP_B:
      tm = OP_MAXIMUM + (mtuid)((size_t) h->cur_mtu);
      break;
      
    case OP_GEN_S:
      tm = OP_MAXIMUM + info->max_gaps + info->sent[h->n - 1];
      break;

    case OP_GEN_T:
      tm = OP_MAXIMUM + info->max_gaps + MAX_VOCAB_SIZE + 0; // TODO: target lexeme;
      break;

    case OP_GEN_ST:
      tm = OP_MAXIMUM + info->max_gaps + 2 * MAX_VOCAB_SIZE + h->cur_mtu->mtu->ident;
      break;
    }

    memmove(h->tm_context, h->tm_context+1, (info->tm_context_len-1) * sizeof(mtuid));
    h->tm_context[info->tm_context_len-1]= tm;
    h->tm_context_hash = util::MurmurHashNative(h->tm_context, info->tm_context_len * sizeof(mtuid), 0);
    */

    if (info->opseq_model != NULL) {
      // TODO: can this be made faster???
      stringstream ss;
      get_op_string(h->last_op, h->op_argument, ss);
      size_t this_op = info->opseq_model->GetVocabulary().Index(ss.str());

      //if (h->tm_context == NULL)
        //h->tm_context = new ng::State();
      float log_prob = info->opseq_model->Score(h->prev->tm_context, this_op, h->tm_context);
      h->tm_context_hash = ng::hash_value(*h->tm_context);
      h->cost -= log_prob;
    }

    return true;
  } else {
    return false;
  }
}

float shift_lm_context(translation_info* info, hypothesis *h, posn M, lexeme*tgt) {
  float log_prob = 0.;
  if (info->language_model == NULL)
    return log_prob;

  if (!h->lm_context_alloc) {
    h->lm_context = new ng::State(*(h->lm_context));
    h->lm_context_alloc = true;
  }
  ng::State in_state = *(h->lm_context);

  in_state = *(h->lm_context);
  for (posn m=0; m<M; m++) {
    log_prob += info->language_model->Score(in_state, tgt[m], *(h->lm_context));
    in_state = *(h->lm_context);
  }
  h->lm_context_hash = ng::hash_value(*h->lm_context);

  return log_prob;
}

template<class T> bool contains(set<T>* S, T t) {
  return S->find(t) != S->end();
}

void expand_one_step(translation_info *info, hypothesis *h, function<void(hypothesis*)> add_operation) {
  // first check to see if the queue is empty
  bool queue_empty = true;
  if (//((h->last_op == OP_GEN_ST) || (h->last_op == OP_CONT_WORD) || (h->last_op == OP_CONT_GAP) /* || (h->last_op == OP_CONT_SKIP) */ ) &&
      (h->cur_mtu != NULL) && 
      (h->queue_head < h->cur_mtu->mtu->src_len))
    queue_empty = false;
      
  posn N = info->N;
  posn n = h->n;
  bool last_op_was_gap = h->last_op == OP_CONT_GAP || h->last_op == OP_GAP;
  char last_last_op = ( h->prev == NULL ) ? OP_UNKNOWN : h->prev->last_op;
  //posn num_uncovered = N - h->cov_vec_count;

  if ( (!queue_empty) && (n < N) ) {   // CONTINUE
    assert(h->last_op != OP_GAP);

    if (h->last_op == OP_CONT_GAP) {
      for (size_t idx=0; idx<NUM_MTU_OPTS; idx++) {
        posn here = h->cur_mtu->found_at[h->queue_head][idx];
        if (here > MAX_SENTENCE_LENGTH) break;
        if (here >= h->n + info->max_gap_width) break;

        if (is_ok_lex_position(info, h, n, idx, here)) { //             (h->gaps->size() < info->max_gaps)) { // TODO: figure out why I wrote the following: && (h->gaps->size() < num_uncovered)) {

          hypothesis *h2 = next_hypothesis(info, h);
          h2->last_op = OP_CONT_WORD;
          h2->queue_head++;
          set_covered(h2, here);
          h2->n = here+1;
          if (prepare_for_add(info, h2)) add_operation(h2);
        }
      }
    } else { // last op was NOT a gap
      mtu_item *mtu = h->cur_mtu->mtu;
      lexeme lex = mtu->src[ h->queue_head ];

      //cout << "n="<<n<<" info->sent[n]=" << info->sent[n] << ", lex="<<lex<<endl;
      if ((info->sent[n] == lex) && (!is_covered(h, n))) {  // generate the word!
        hypothesis *h2 = next_hypothesis(info, h);
        h2->last_op = OP_CONT_WORD;
        h2->queue_head++;
        set_covered(h2, n);
        h2->n = n+1;
        if (prepare_for_add(info, h2)) add_operation(h2);
      }

      if ((h->queue_head > 0) && gap_allowed_after(h->cur_mtu->mtu->gap_option, h->queue_head-1) && (h->gaps->size() < info->max_gaps) && (!last_op_was_gap) && (!contains(h->gaps,n))) {
        hypothesis *h2 = next_hypothesis(info, h);
        h2->last_op = OP_CONT_GAP;
        if (! h2->gaps_alloc) {
          h2->gaps = new set<posn>( *h->gaps );
          h2->gaps_alloc = true;
        }
        h2->gaps->insert(n);
        h2->n = n+1;
        if (prepare_for_add(info, h2)) add_operation(h2);
      }
    }
  }

  if (queue_empty && (n < N)) { // try generating a new cept
    posn top = (last_op_was_gap) ? (N) : (n+1);
    assert(top <= N);
    for (posn m = n; m < top; m++) {
      if (m >= h->n + info->max_gap_width) break;
      if (is_covered(h, m)) continue;
      vector<mtu_for_sent*> mtus = info->mtus_at[m];
      for (auto &mtu : mtus) {
        //cout << "trying @m="<<m<<" lex="<<info->sent[m]<<": "; print_mtu(*mtu->mtu);cout<<endl;
        hypothesis *h2 = next_hypothesis(info, h);
        h2->last_op = OP_GEN_ST;
        h2->op_argument = mtu->mtu->ident;
        set_covered(h2, m);
        h2->n = m+1;
        h2->cur_mtu = mtu;
        h2->queue_head = 1;
        h2->cost -= shift_lm_context(info, h2, mtu->mtu->tgt_len, mtu->mtu->tgt);
        if (prepare_for_add(info, h2)) add_operation(h2);
      }
    }
  }

  if ((h->gaps->size() < info->max_gaps) && (!last_op_was_gap) && (n<N) && (!is_covered(h,n)) && (!contains(h->gaps,n)) && queue_empty) {
    //        (h->gaps->size() < num_uncovered) &&   // TODO: be sure this was actually a bad idea
    //cout << "inserting gap at n=" << h->n <<" Z="<<h->Z<<endl;
    hypothesis *h2 = next_hypothesis(info, h);
    h2->last_op = OP_GAP;
    if (! h2->gaps_alloc) {
      h2->gaps = new set<posn>( *h->gaps );
      h2->gaps_alloc = true;
    }
    h2->gaps->insert(h->n);
    h2->cost += info->gap_cost;
    h2->n++;
    if (prepare_for_add(info, h2)) add_operation(h2);
  }

  { // TODO: GEN_T

  }

  if (h->gaps->size() > 0) {   // was: n == h->Z
    size_t gap_id = h->gaps->size();
    for (auto &gap_pos : *h->gaps) {
      assert(gap_id > 0);
      gap_id--;
      if (gap_pos+1 != h->n) {
        hypothesis *h2 = next_hypothesis(info, h);
        h2->last_op = OP_JUMP_B;
        h2->op_argument = (size_t)gap_id;
        h2->n = gap_pos;
        if (! h2->gaps_alloc) {
          h2->gaps = new set<posn>( *h->gaps );
          h2->gaps_alloc = true;
        }
        h2->gaps->erase(gap_pos);
        if (prepare_for_add(info, h2)) add_operation(h2);
      }
    }
  }

  if ((n < N) && (!is_covered(h, n)) && (!last_op_was_gap)) { // generate just S
    hypothesis *h2 = next_hypothesis(info, h);
    h2->last_op = OP_GEN_S;
    h2->op_argument = (size_t)info->sent[n];
    set_covered(h2, n);
    h2->n++;
    h2->cost += info->gen_s_cost;
    if (prepare_for_add(info, h2)) add_operation(h2);
  }

    
  if ( (h->n < h->Z) && (h->last_op != OP_JUMP_B) && 
       ( ! ((last_last_op == OP_JUMP_B) && last_op_was_gap) ) ) {  // don't allow JUMP_B - GAP - JUMP_E
    hypothesis *h2 = next_hypothesis(info, h);
    h2->last_op = OP_JUMP_E;
    h2->n = h2->Z;
    if (prepare_for_add(info, h2)) add_operation(h2);
  }
}

bool is_final_hypothesis(translation_info *info, hypothesis *h) {
  posn N = info->N;
  if (h->n < N) return false;
  if (h->Z < N) return false;
  for (posn n=0; n<N; n++)
    if (! is_covered(h, n))
      return false;
  return true;
}

void expand_to_generation(translation_info *info, hypothesis *h, function<void(hypothesis*)> add_operation) {
  stack<hypothesis*> my_stack;
  my_stack.push(h);
  
  while (! my_stack.empty()) {
    //cerr << "|my_stack| = " << my_stack.size() << endl;
    hypothesis*cur = my_stack.top(); my_stack.pop();
    expand_one_step(info, cur, [info,add_operation,my_stack](hypothesis*next) mutable -> void {
        if (is_final_hypothesis(info, next) ||
            (next->last_op == OP_GEN_ST) ||
            (next->last_op == OP_CONT_WORD) ||
            (next->last_op == OP_GEN_S))
          add_operation(next);
        else
          my_stack.push(next);
      });
  }
}

void set_pruning_threshold(translation_info *info, hyp_stack* stack) {
  if (info->pruning_coefficient < 1)
    return;

  stack->prune_if_gt = stack->lowest_cost * info->pruning_coefficient;
}

size_t bucket_contains_equiv(translation_info*info, vector<hypothesis*> bucket, hypothesis *h) {
  for (size_t pos=0; pos<bucket.size(); pos++) {
    hypothesis *h2 = bucket[pos];
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

    (*h->cov_vec) ^= (*h2->cov_vec);
    bool cov_vec_eq = ! h->cov_vec->any();
    (*h->cov_vec) ^= (*h2->cov_vec);

    if (! cov_vec_eq)
        continue;

    return pos;
  }
  return (size_t)-1;
}

void recombine_stack(translation_info *info, hyp_stack* stack) {
  recombination_data * buckets = info->recomb_buckets;
  size_t mod = buckets->size();

  for (auto vec = buckets->begin(); vec != buckets->end(); vec++)
    (*vec).clear();

  for (auto h : stack->Stack) {
    if (h->skippable) continue;
    size_t id = (h->lm_context_hash * 3481183 +
                 h->tm_context_hash * 8942137 +
                 h->cov_vec_hash    * 9138921) % mod;

    size_t equiv_pos = bucket_contains_equiv(info, (*buckets)[id], h);

    if (equiv_pos == (size_t)-1) {  // there was NOT an equivalent hyp
      (*buckets)[id].push_back(h);
    } else {  // there was at position equiv_pos      
      if (h->cost < (*buckets)[id][equiv_pos]->cost) {
        (*buckets)[id][equiv_pos]->skippable = true;
        (*buckets)[id][equiv_pos] = h;
      } else {
        h->skippable = true;
      }
    }
  }
}
/*
void add_to_stack(translation_info *info, vector<hypothesis*> &S, hypothesis*h) {
  if (false) {
    S.push_back(h);
    return;
  }
  recombination_data *buckets = (info->recomb_buckets);
  size_t mod = buckets->size();

  if (h->skippable) return;
  size_t id = (h->lm_context_hash * 3481183 +
               h->tm_context_hash * 8942137 +
               h->cov_vec_hash    * 9138921) % mod;
  
  size_t equiv_pos = bucket_contains_equiv(info, (*buckets)[id], h);
  if (equiv_pos == (size_t)-1) {  // there was NOT an equivalent hyp
    S.push_back(h);
  } else {  // there was at position equiv_pos      
    if (h->cost < (*buckets)[id][equiv_pos]->cost) {
      S.push_back(h);
    } else {
      h->skippable = true;
    }
  }
}
*/
void initialize_hyp_stack(hyp_stack* stack) {
  stack->lowest_cost  = FLT_MAX;
  stack->highest_cost = FLT_MIN;
  stack->prune_if_gt  = FLT_MAX;
}

void add_to_hyp_stack(translation_info*info, hyp_stack* stack, hypothesis* h, bool try_recombination) {
  if (h->skippable)
    return;

  if (h->cost > stack->prune_if_gt)
    return;
 
  if ((stack->Stack.size() >= info->max_bucket_size) &&
      (h->cost >= stack->highest_cost))
    return;

  if (false && try_recombination) {
    recombination_data *buckets = info->recomb_buckets;
    size_t mod = buckets->size();

    size_t id = (h->lm_context_hash * 3481183 +
                 h->tm_context_hash * 8942137 +
                 h->cov_vec_hash    * 9138921) % mod;
  
    size_t equiv_pos = bucket_contains_equiv(info, (*buckets)[id], h);
    if (equiv_pos == (size_t)-1) {  // there was NOT an equivalent hyp
      stack->Stack.push_back(h);
      if (h->cost < stack->lowest_cost ) stack->lowest_cost  = h->cost;
      if (h->cost > stack->highest_cost) stack->highest_cost = h->cost;
    } else {  // there was at position equiv_pos      
      if (h->cost < (*buckets)[id][equiv_pos]->cost) {
        stack->Stack.push_back(h);
        if (h->cost < stack->lowest_cost ) stack->lowest_cost  = h->cost;
        if (h->cost > stack->highest_cost) stack->highest_cost = h->cost;
      } else {
        h->skippable = true;
      }
    }
  } else {
    stack->Stack.push_back(h);
    if (h->cost < stack->lowest_cost ) stack->lowest_cost  = h->cost;
    if (h->cost > stack->highest_cost) stack->highest_cost = h->cost;
  }
}

bool sort_hyp_by_cost(hypothesis*a, hypothesis*b) { return a->cost < b->cost; }

void shrink_hyp_stack(translation_info*info, hyp_stack* stack, bool force_shrink=false) {
  if (stack->Stack.size() <= info->max_bucket_size)
    return;

  if ((!force_shrink) && (stack->Stack.size() <= 2*info->max_bucket_size))
    return;

  auto begin  = stack->Stack.begin();
  auto end    = stack->Stack.end();
  auto middle = begin + info->max_bucket_size;
  partial_sort(begin, middle, end, sort_hyp_by_cost);

  middle = stack->Stack.begin() + info->max_bucket_size;
  end    = stack->Stack.end();
  stack->Stack.erase(middle, end);
}

pair< vector<hypothesis*>, vector<hypothesis*> > stack_generic_search(translation_info *info, size_t (*get_stack_id)(hypothesis*), size_t num_stacks_reserve=0) {
  unordered_map< size_t, hyp_stack* > Stacks;
  stack< size_t > NextStacks;
  vector< hypothesis* > visited;
  vector< hypothesis* > Goals;

  if (num_stacks_reserve > 0) {
    Stacks.reserve(num_stacks_reserve);
  }
  assert(Stacks[5] == NULL); assert(Stacks[57] == NULL);

  hypothesis *h0 = next_hypothesis(info, NULL);
  size_t stack0 = get_stack_id(h0);
  NextStacks.push(stack0);
  Stacks[stack0] = new hyp_stack();
  initialize_hyp_stack(Stacks[stack0]);

  add_to_hyp_stack(info, Stacks[stack0], h0, false);

  while (! NextStacks.empty()) {
    size_t cur_stack = NextStacks.top(); NextStacks.pop();
    //cout << "cur_stack="<<cur_stack<<endl;
    if (Stacks[cur_stack] == NULL) continue;
    //cerr<<"["<<Stacks[cur_stack]->Stack.size();

    set_pruning_threshold(info, Stacks[cur_stack]);
    recombine_stack(info, Stacks[cur_stack]);
    //cerr<<":"<<Stacks[cur_stack]->Stack.size();
    shrink_hyp_stack(info, Stacks[cur_stack], true);
    // cerr<<":"<<Stacks[cur_stack]->Stack.size()<<"]";
    // cerr<<"\t/"<<NextStacks.size()<<endl;

    auto it = Stacks[cur_stack]->Stack.begin();
    while (it != Stacks[cur_stack]->Stack.end()) {
      hypothesis *h = *it;
      if (h->cost > Stacks[cur_stack]->prune_if_gt) { it++; continue; }
      if (h->skippable) { it++; continue; }

      size_t diff = it - Stacks[cur_stack]->Stack.begin();
      //cerr << endl<<"expanding: "; print_hypothesis(info, h);
      //for (hypothesis* par=h->prev; par!=NULL; par=par->prev) { cerr <<"     from: "; print_hypothesis(info, par); } cout<<endl;

      expand_to_generation(info, h, [info,&Goals,cur_stack,&Stacks,&NextStacks,get_stack_id](hypothesis* next) mutable -> void {
          //cerr << "     next: "; print_hypothesis(info, next);
          // TODO: only keep around the k-best goals?
          if (is_final_hypothesis(info, next))
            Goals.push_back(next);
          else { // not final
            size_t next_stack = get_stack_id(next);
            assert(next_stack >= cur_stack);
            if (next_stack == cur_stack) {
              add_to_hyp_stack(info, Stacks[cur_stack], next, true);
            } else { // different stack
              if (Stacks[next_stack] == NULL) {
                Stacks[next_stack] = new hyp_stack();
                initialize_hyp_stack(Stacks[next_stack]);
                NextStacks.push(next_stack);
              }
              add_to_hyp_stack(info, Stacks[next_stack], next, false);
              shrink_hyp_stack(info, Stacks[next_stack], false);
            }
          }
        });
      it = Stacks[cur_stack]->Stack.begin() + diff;
      it++;
    }
    Stacks[cur_stack]->Stack.clear();
  }

  for (auto it : Stacks) delete it.second;

  return { Goals, visited };
}

bool force_decode_allowed(translation_info *info, vector<operation> op_seq, hypothesis* h, size_t i) {
  assert(i >= 0); assert(i < op_seq.size());
  char id = op_seq[i].op;
  lexeme arg1 = op_seq[i].arg1;
  posn   arg2 = op_seq[i].arg2;  // this is always the position BEFORE the operation

  if (h->last_op != id)
    return false;

  switch(id) {
  case OP_GEN_S:    // arg1 = the src word to gen, op_argument = src word
    if ((arg1 != h->op_argument) || (arg2 != h->n-1))
      return false;
    break;

  case OP_CONT_GAP:   // no arguments
    if (arg2 != h->n-1)
      return false;
    break;

  case OP_GAP:      // no arguments
    if (arg2 != h->n-1)
      return false;
    break;

  case OP_JUMP_E:   // no args
    break;

  case OP_JUMP_B:   // arg1 = gap id (counting from right), op_argument = gap id
    if (arg1 != h->op_argument)
      return false;
    break;

  case OP_GEN_ST:   // arg1 = mtu id, h->cur_mtu->mtu->ident = mtu id
    if ((h->cur_mtu == NULL) || (h->cur_mtu->mtu == NULL) || 
        (arg1 != h->cur_mtu->mtu->ident) ||
        (arg2 != h->n-1))
      return false;
    break;

  case OP_CONT_WORD:   // no args
    if (arg2 != h->n-1)
      return false;
    break;

  default:
    break;
  }

  return true;
}

pair< vector<hypothesis*>, vector<hypothesis*> > stack_search(translation_info *info, vector<operation> *force_decode_op_seq = NULL) {
  hypothesis *h0 = next_hypothesis(info, NULL);

  stack<hypothesis*> S;
  vector<hypothesis*> visited;
  vector<hypothesis*> Goals;
  size_t decode_step = 0;

  S.push(h0);
  
  while (!S.empty()) {
    if (force_decode_op_seq)
      assert(S.size() == 1);

    hypothesis *h = S.top(); S.pop();
    //cout<<endl<< "pop\t";
    //cout << "op=" << OP_NAMES[(uint32_t)((*force_decode_op_seq)[decode_step]).first] << "\targ=(" << ((*force_decode_op_seq)[decode_step]).second.first << ", " << ((*force_decode_op_seq)[decode_step]).second.second << ")" << endl;
    //print_hypothesis(info, h);

    expand_one_step(info, h, [info, &Goals, &S, force_decode_op_seq, decode_step](hypothesis*next) mutable -> void {
        if ((force_decode_op_seq == NULL) ||
            (force_decode_allowed(info, *force_decode_op_seq, next, decode_step))) {
          if (is_final_hypothesis(info, next))
            Goals.push_back(next);
          else
            S.push(next);
          //cout << "   added: "; print_hypothesis(info, next);
        } else {
          //cout << "rejected: "; print_hypothesis(info, next);
        }
      });
    visited.push_back(h);
    decode_step++;
  }

  visited.insert( visited.end(), Goals.begin(), Goals.end() );
  return {Goals, visited};
}


void mtu_add_unit(mtu_item_dict &dict, mtu_item*mtu) {
  dict[mtu->src[0]].push_back(mtu);
}

void mtu_add_item_string(mtu_item_dict &dict, mtuid ident, string src, string tgt) {
  mtu_item *mtu = (mtu_item*)calloc(1, sizeof(mtu_item));
  mtu->src_len = src.length();
  posn j = 0;
  for (uint32_t n=0; n<mtu->src_len; n++) {
    if (src[n] == '_')
      allow_gap_after(mtu->gap_option, j-1);
    else {
      mtu->src[j] = (uint32_t)src[n];
      j++;
    }
  }

  mtu->tgt_len = tgt.length();
  for (uint32_t n=0; n<mtu->tgt_len; n++) {
    mtu->tgt[n] = (uint32_t)tgt[n];
  }

  mtu->ident = ident;

  mtu_add_unit(dict, mtu);
}

void free_dict(mtu_item_dict dict) {
  for (auto it=dict.begin(); it!=dict.end(); it++) {
    vector<mtu_item*> mtus = (*it).second;
    for (auto mtu_it=mtus.begin(); mtu_it!=mtus.end(); mtu_it++)
      free(*mtu_it);
  }
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

bool find_mtu_in_dict(mtu_item_dict *dict, vector<lexeme> e, vector<lexeme> f, vector<posn> a, lexeme &identity) {
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
    for (posn i=1; i<f_len; i++)
      if (f[a[i]] != mtu->src[i]) {
        ok = false;
        break;
      }
    if (!ok) continue;

    for (posn i=0; i<e_len; i++)
      if (e[i] != mtu->tgt[i]) {
        ok = false;
        break;
      }
    if (!ok) continue;

    for (posn i=0; i<f_len-1; i++) {
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
    return true;
  }

  //cout << "    FAIL" << endl;
  return false;
}

posn filter_gap_width(posn j, bitset<MAX_SENTENCE_LENGTH> fcov, posn init_gap_width) {
  for (posn real_gap_width=1; real_gap_width<=init_gap_width; real_gap_width++) {
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

vector<operation> get_operation_sequence(aligned_sentence_pair data, mtu_item_dict *dict) {
  auto f = data.F;
  auto E = data.E;
  auto A = data.A;
  vector<operation> op_seq;
  set<posn> gaps;

  if (dict != NULL) { // remove any alignments that are not mtus in the dictionary
    for (posn i=0; i<A.size(); i++) {
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
      if (j == Z) 
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
            for (posn m=pos; m<=A[i][k]; m++)
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
      //if (j < Z)
      //op_seq.push_back( { OP_JUMP_E, 0, j } );

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
        assert(find_mtu_in_dict(dict, E[i], f, A[i], identity));
        //cout << "GEN_ST i="<<i<<" identity="<<identity<<" and f[j]="<<f[j]<<endl;
        op_seq.push_back( { OP_GEN_ST, identity, j } );
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
  for (posn i=1; i<al.size(); i++)
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

  nr = fscanf(fd, "%d %d\t", &mtu.tr_doc_freq, &mtu.tr_freq);
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
  return dict;
}


void collect_mtus(size_t max_phrase_len, aligned_sentence_pair spair, unordered_map< mtu_item, mtu_item_info > &cur_mtus, size_t &skipped_for_len) {
  auto E = spair.E;
  auto F = spair.F;
  auto A = spair.A;

  //cout << "F ="; for (auto j : F) cout << " " << (uint32_t)j; cout << endl;

  assert(E.size() == A.size());
  for (posn i=0; i<E.size(); i++) {
    vector<posn> al = A[i];
    vector<lexeme> ephr = E[i];

    //cout << "al ="; for (auto j : al) cout << " " << (uint32_t)j; cout << endl;

    assert(is_sorted(al));

    if (ephr.size() > max_phrase_len) { skipped_for_len++; continue; }

    mtu_item mtu;
    memset(&mtu, 0, sizeof(mtu));
    mtu.tgt_len = ephr.size();
    for (posn j=0; j<ephr.size(); j++)
      mtu.tgt[j] = ephr[j];

    if (al.size() > max_phrase_len) { skipped_for_len++; continue; }

    mtu.src_len = al.size();
    uint32_t my_gaps = 0;
    for (posn j=0; j<al.size(); j++) {
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

aligned_sentence_pair read_next_aligned_sentence(FILE *fd, bool &is_new_document) {
  assert(! feof(fd));

  lexeme cnt, w,x;
  size_t nr;

  is_new_document = false;

  nr = fscanf(fd, "%d", &cnt);
  assert(nr == 1);

  if (cnt == 0) {
    // this is a marker for a new document (just a single line containing "0")
    is_new_document = true;
    nr = fscanf(fd, "%d", &cnt);
    assert(nr == 1);
  }
  assert(cnt <= MAX_SENTENCE_LENGTH);
  vector<lexeme> F(cnt);
  for (posn i=0; i<cnt; i++) nr = fscanf(fd, " %d", &F[i]);

  nr = fscanf(fd, "\t%d", &cnt);
  assert(cnt <= MAX_SENTENCE_LENGTH);
  vector<lexeme> e(cnt);
  for (posn i=0; i<cnt; i++) nr = fscanf(fd, " %d", &e[i]);

  nr = fscanf(fd, "\t%d", &cnt);
  vector< set<posn> > al;  // maps from english id to (set of) french ids
  for (posn i=0; i<e.size(); i++)
    al.push_back(set<posn>());
  for (posn i=0; i<cnt; i++) {
    nr = fscanf(fd, " %d-%d", &w, &x);
    al[x].insert(w);
  }
  nr = fscanf(fd, "\n");

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

  return { F, E, A };
}


void test_align() {
  vector< lexeme >         F = { 'D', 'H', 'E', 'I', 'B', 'G' };
  vector< vector<lexeme> > E = { { 't' }, { 'h' }, { 'r' }, { 'a' }, { 'b' } };
  vector< vector<posn  > > A = { { 0 }, { 2 }, { 1, 5 }, { 3 }, { 4 } };

  auto op_seq = get_operation_sequence( {F, E, A}, NULL );

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
  info.N       = 3;
  info.sent[0] = (uint32_t)'A';
  info.sent[1] = (uint32_t)'B';
  info.sent[2] = (uint32_t)'C';
  info.sent[3] = (uint32_t)'A';
  info.sent[4] = (uint32_t)'B';
  info.sent[5] = (uint32_t)'C';
  build_sentence_mtus(&info, dict);
  info.compute_cost = simple_compute_cost;
  info.language_model = NULL; // new lm::ngram::Model((char*)"file.arpa-bin");
  info.max_gaps = 5;
  info.tm_context_len = 3;
  info.max_gap_width = 1;
  info.max_phrase_len = 5;

  info.operation_allowed = (1 << OP_MAXIMUM) - 1;  // all operations
    // (1 << OP_INIT  ) |
    // (1 << OP_GEN_ST) |
    // (1 << OP_CONT_WORD) |
    // (1 << OP_CONT_GAP) |
    // (1 << OP_GEN_S ) |
    // (1 << OP_GEN_T ) |
    // (1 << OP_GAP   ) |
    // (1 << OP_JUMP_B) |
    // (1 << OP_JUMP_E) |
    // 0;

  info.pruning_coefficient = 0.;
  info.recomb_buckets = new recombination_data(10231);

  for (size_t rep = 0; rep < 1 + 0*999; rep++) {
    cerr<<".";
    info.hyp_ring = initialize_hypothesis_ring(&info, INIT_HYPOTHESIS_RING_SIZE);

    pair< vector<hypothesis*>, vector<hypothesis*> > GoalsVisited = 
      // for search based on amount of coverage
      stack_generic_search(&info, [](hypothesis* hyp) { return (size_t)hyp->cov_vec_count; }, info.N*2);
      // for search based on (hash of) coverage vector
      //stack_generic_search(&info, [](hypothesis* hyp) { return (size_t)hyp->cov_vec_hash; }, info.N*100);



    for (auto &hyp : GoalsVisited.first) {
      vector<lexeme> trans = get_translation(&info, hyp);

      cout<<hyp->cost<<"\t";
      for (auto &w : get_translation(&info, hyp))
        cout<<" "<<(size_t)w;
      cout<<endl;
    }

    free_hypothesis_ring(info.hyp_ring);
  }
  cerr<<endl;
  if (info.language_model != NULL) delete info.language_model;
  delete info.recomb_buckets;
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


void extract(int argc, char*argv[]) {  // options start at argv[2]
  size_t max_phrase_len = 5;

  if ((argc != 3) || (!strcmp(argv[2], "-h")) || (!strcmp(argv[2], "-help")) || (!strcmp(argv[2], "--help"))) {
    cout << "usage: ngdec extract [ngdec file] > [mtu file]" << endl;
    exit(-1);
  }

  FILE *fd = fopen(argv[2], "r");
  if (fd == 0) { cerr << "error: cannot open file for reading: '" << argv[2] << "'" << endl; throw exception(); }

  size_t sent_id = 0;
  size_t next_print = 100;
  unordered_map<mtu_item, mtu_item_info> all_mtus;
  unordered_map<mtu_item, mtu_item_info> cur_doc_mtus;

  size_t skipped_for_len = 0;
  while (!feof(fd)) {
    sent_id++; if (sent_id == next_print) { cerr << "reading sentence pair " << sent_id << endl; next_print *= 2; }

    bool is_new_document = false;
    aligned_sentence_pair spair = read_next_aligned_sentence(fd, is_new_document);
    if (is_new_document) {
      add_all_mtus(all_mtus, cur_doc_mtus);
      cur_doc_mtus.clear();
    }
    collect_mtus(max_phrase_len, spair, cur_doc_mtus, skipped_for_len);
  }
  add_all_mtus(all_mtus, cur_doc_mtus);
  fclose(fd);

  cerr << "collected " << all_mtus.size() << " mtus" << endl;
  cerr << "skipped " << skipped_for_len << " for length" << endl;

  print_mtu_set(all_mtus, true);
}

void oracle(int argc, char*argv[]) {  // options start at argv[2]
  if ((argc != 4) || (!strcmp(argv[2], "-h")) || (!strcmp(argv[2], "-help")) || (!strcmp(argv[2], "--help"))) {
    cout << "usage: ngdec oracle [mtu file] [ngdec file] > [opseq file]" << endl;
    exit(-1);
  }

  FILE *fd = fopen(argv[2], "r");
  if (fd == 0) { cerr << "error: cannot open mtu file for reading: '" << argv[2] << "'" << endl; throw exception(); }
  mtu_item_dict dict = read_mtu_item_dict(fd);
  fclose(fd);
  
  fd = fopen(argv[3], "r");
  if (fd == 0) { cerr << "error: cannot open ngdec file for reading: '" << argv[3] << "'" << endl; throw exception(); }

  size_t sent_id = 0;
  size_t next_print = 100;

  while (!feof(fd)) {
    sent_id++; if (sent_id == next_print) { cerr << "reading sentence pair " << sent_id << endl; next_print *= 2; }

    bool is_new_document = false;
    aligned_sentence_pair spair = read_next_aligned_sentence(fd, is_new_document);
    if (is_new_document) cout << endl;
    vector<operation> op_seq;

    op_seq = get_operation_sequence(spair, &dict);
    pretty_print_op_seq(op_seq);
  }

  fclose(fd);
  free_dict(dict);
}

void predict_forced(int argc, char*argv[]) {  // options start at argv[2]
  if ((argc != 4) || (!strcmp(argv[2], "-h")) || (!strcmp(argv[2], "-help")) || (!strcmp(argv[2], "--help"))) {
    cout << "usage: ngdec predict-forced [mtu file] [ngdec file] > [predictions]" << endl;
    exit(-1);
  }

  FILE *fd = fopen(argv[2], "r");
  if (fd == 0) { cerr << "error: cannot open mtu file for reading: '" << argv[2] << "'" << endl; throw exception(); }
  mtu_item_dict dict = read_mtu_item_dict(fd);
  fclose(fd);
  
  fd = fopen(argv[3], "r");
  if (fd == 0) { cerr << "error: cannot open ngdec file for reading: '" << argv[3] << "'" << endl; throw exception(); }

  size_t sent_id = 0;
  size_t next_print = 100;

  translation_info info;
  info.compute_cost = simple_compute_cost;
  info.language_model = NULL;
  info.operation_allowed = (1 << OP_MAXIMUM) - 1;  // all operations
  info.pruning_coefficient = 0.;
  info.recomb_buckets = new recombination_data(10231);
  info.max_gaps = 5;
  info.tm_context_len = 3;
  info.max_gap_width = 1;
  info.max_phrase_len = 5;

  while (!feof(fd)) {
    sent_id++; if (sent_id == next_print) { cerr << "reading sentence pair " << sent_id << endl; next_print *= 2; }

    bool is_new_document = false;
    aligned_sentence_pair spair = read_next_aligned_sentence(fd, is_new_document);
    if (is_new_document) cout << endl;
    vector<operation> op_seq;

    op_seq = get_operation_sequence(spair, &dict);

    info.N = spair.F.size();
    for (posn i=0; i<info.N; i++)
      info.sent[i] = spair.F[i];
    build_sentence_mtus(&info, dict);
    info.hyp_ring = initialize_hypothesis_ring(&info, INIT_HYPOTHESIS_RING_SIZE);

    pair< vector<hypothesis*>, vector<hypothesis*> > GoalsVisited = stack_search(&info, &op_seq);

    if (GoalsVisited.first.size() == 0)
      cout<<"FAIL"<<endl;
    else
      for (auto &hyp : GoalsVisited.first) {
        vector<lexeme> trans = get_translation(&info, hyp);
        cout<<hyp->cost<<"\t";
        for (auto &w : get_translation(&info, hyp))
          cout<<" "<<w;
        cout<<endl;
        break;
      }
    
    free_sentence_mtus(info.mtus_at);
    free_hypothesis_ring(info.hyp_ring);
  }
  fclose(fd);

  if (info.language_model != NULL) delete info.language_model;
  delete info.recomb_buckets;
  free_dict(dict);
}

template<size_t N>
size_t bitset_hash(bitset<N> bs) {
 std::hash<std::bitset<N>> hash_fn;
 return hash_fn(bs);
  /*
  size_t hash = 84032481;
  for (size_t j=0; j<N; j++) {
    hash = 483921053 * hash + 389015 * bs[j];
  }
  return hash;
  */
}


void predict_lm(int argc, char*argv[]) {  // options start at argv[2]
  if ((argc != 6) || (!strcmp(argv[2], "-h")) || (!strcmp(argv[2], "-help")) || (!strcmp(argv[2], "--help"))) {
    cout << "usage: ngdec predict-lm [en lm file] [opseq lm file] [mtu file] [ngdec file] > [predictions]" << endl;
    exit(-1);
  }

  FILE *fd = fopen(argv[4], "r");
  if (fd == 0) { cerr << "error: cannot open mtu file for reading: '" << argv[4] << "'" << endl; throw exception(); }
  mtu_item_dict dict = read_mtu_item_dict(fd);
  fclose(fd);
  
  fd = fopen(argv[5], "r");
  if (fd == 0) { cerr << "error: cannot open ngdec file for reading: '" << argv[5] << "'" << endl; throw exception(); }

  size_t sent_id = 0;
  size_t next_print = 100;

  translation_info info;
  info.compute_cost = simple_compute_cost;
  info.language_model = new lm::ngram::Model(argv[2]);
  info.opseq_model    = new lm::ngram::Model(argv[3]);
  info.operation_allowed = (1 << OP_MAXIMUM) - 1;  // all operations
  //info.operation_allowed &= ~(1 << OP_GEN_S); // turn off OP_GEN_S
  info.gen_s_cost = 10.;
  info.gap_cost   = 10.;
  info.pruning_coefficient = 10.;  // prune anything more than X* as bad as the best thing
  info.max_bucket_size = 500;
  info.recomb_buckets = new recombination_data(10231);
  info.max_gaps = 10;
  info.max_gap_width = 16;
  info.max_phrase_len = 5;

  while (!feof(fd)) {
    sent_id++; if (sent_id == next_print) { cerr << "reading sentence pair " << sent_id << endl; next_print *= 2; }

    bool is_new_document = false;
    aligned_sentence_pair spair = read_next_aligned_sentence(fd, is_new_document);
    if (is_new_document) cout << endl;

    info.N = spair.F.size();
    for (posn i=0; i<info.N; i++)
      info.sent[i] = spair.F[i];
    build_sentence_mtus(&info, dict);
    info.hyp_ring = initialize_hypothesis_ring(&info, INIT_HYPOTHESIS_RING_SIZE);

    pair< vector<hypothesis*>, vector<hypothesis*> > GoalsVisited = 
      stack_generic_search(&info, [](hypothesis* hyp) { 
          return (size_t)hyp->cov_vec_count * 2 + ((hyp->n != hyp->Z) || (!is_covered(hyp, hyp->n)));
          //return bitset_hash(*hyp->cov_vec);
        }, info.N*2);

    if (GoalsVisited.first.size() == 0)
      cout<<"FAIL"<<endl;
    else {
      size_t best_hyp_id = 0;
      for (size_t id=0; id<GoalsVisited.first.size(); id++) {
        if (GoalsVisited.first[id]->cost < GoalsVisited.first[best_hyp_id]->cost)
          best_hyp_id = id;
      }

      hypothesis*hyp = GoalsVisited.first[best_hyp_id];
      vector<lexeme> trans = get_translation(&info, hyp);
      cout<<hyp->cost<<"\t";
      for (auto &w : get_translation(&info, hyp))
        cout<<" "<<w;
      cout<<endl;

      //for (hypothesis* par=hyp; par!=NULL; par=par->prev) { cerr <<"     from: "; print_hypothesis(&info, par); } cerr<<endl;

    }
    
    free_sentence_mtus(info.mtus_at);
    free_hypothesis_ring(info.hyp_ring);
  }
  fclose(fd);

  if (info.language_model != NULL) delete info.language_model;
  if (info.opseq_model != NULL) delete info.opseq_model;
  delete info.recomb_buckets;
  free_dict(dict);
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
  void print_system_usage(timespec started_timespec) {
    std::ifstream status("/proc/self/status", std::ios::in);
    string header, value;
    while ((status >> header) && getline(status, value)){
      if ((header == "VmPeak:") || (header == "VmRSS:"))
        cerr << header << skip_spaces(value.c_str()) << "\t";
    }
    /*
    struct rusage usage;
    if (getrusage(RUSAGE_CHILDREN, &usage))
      throw exception();
    cerr << "RSSMax:" << usage.ru_maxrss<< " kB\t";
    cerr << "user:" << FloatSec(usage.ru_utime) << "\tsys:" << FloatSec(usage.ru_stime) << '\t';
    cerr << "CPU:" << (FloatSec(usage.ru_utime) + FloatSec(usage.ru_stime));
    cerr << "\t";
    */

    struct timespec current_timespec;
    clock_gettime(CLOCK_MONOTONIC, &current_timespec);
    cerr << "time:" << (FloatSec(current_timespec) - FloatSec(started_timespec)) << "s" << endl;
  }
}
  

int main(int argc, char*argv[]) {
  if (argc < 2) usage();

  timespec started_timespec;
  clock_gettime(CLOCK_MONOTONIC, &started_timespec);

  if      (!strcmp(argv[1], "extract"       )) { extract       (argc, argv); }
  else if (!strcmp(argv[1], "oracle"        )) { oracle        (argc, argv); }
  else if (!strcmp(argv[1], "predict-forced")) { predict_forced(argc, argv); }
  else if (!strcmp(argv[1], "predict-lm"    )) { predict_lm    (argc, argv); }
  else { usage(); }
  //test_align();
  //test_decode();
  //test_lm();
  //test_big_decode(argv[1], false);
  //main_collect_mtus(argv[1]);

  StolenFromKenLM::print_system_usage(started_timespec);

  return 0;
}

  
