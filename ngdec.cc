#include <iostream>
#include <functional>
#include <assert.h>
#include <float.h>
#include <stack>
#include <unordered_set>
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


bool is_covered(hypothesis *h, posn n) {
  return (*(h->cov_vec))[n];
}

void set_covered(hypothesis *h, posn n) {
  assert(! ((*(h->cov_vec))[n]) );
  if (! h->cov_vec_alloc) {
    assert( h->prev != NULL );
    h->cov_vec = new bitset<MAX_SENTENCE_LENGTH>(*h->prev->cov_vec);
    h->cov_vec_alloc = true;
  }
  (*(h->cov_vec))[n] = true;
  h->cov_vec_hash += n * 38904831901;
  h->cov_vec_count++;
  assert( h->cov_vec_count == h->cov_vec->count() );
}

void free_hypothesis_ring(hypothesis_ring*r) {
  vector<hypothesis_ring*> to_free;
  while (r != NULL) {
    for (size_t i=0; i<r->my_size; i++) 
      free(r->my_hypotheses[i].tm_context);

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

hypothesis_ring* initialize_hypothesis_ring(size_t desired_size) {
  hypothesis_ring *r = (hypothesis_ring*) calloc(1, sizeof(hypothesis_ring));
  r->my_hypotheses = (hypothesis*) calloc(desired_size, sizeof(hypothesis));
  for (size_t i=0; i<desired_size; i++)
    r->my_hypotheses[i].tm_context = (mtuid*) calloc(TM_CONTEXT_LEN, sizeof(mtuid));
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
    hypothesis_ring * new_ring = initialize_hypothesis_ring(new_size);
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
    memset(h2->tm_context, OP_INIT, TM_CONTEXT_LEN);
    h2->tm_context_hash = util::MurmurHashNative(h2->tm_context, TM_CONTEXT_LEN * sizeof(mtuid), 0);
    h2->skippable = false;
    h2->cost = 1.;
    h2->prev = NULL;
  } else {
    mtuid* old_tm_context = h2->tm_context;
    memcpy(h2, h, sizeof(hypothesis));
    h2->last_op = OP_UNKNOWN;
    h2->cov_vec_alloc = false;
    h2->gaps_alloc = false;
    h2->lm_context_alloc = false;
    h2->tm_context = old_tm_context;
    memcpy(h2->tm_context, h->tm_context, TM_CONTEXT_LEN-1);
    h2->skippable = false;
    h2->prev = h;
  }

  return h2;
}

/*
void print_hypothesis(translation_info* info, hypothesis *h) {
  cout << h;
  cout << "\tlast_op="<<OP_NAMES[(size_t)h->last_op];
  cout << "\tcur_mtu="<<h->cur_mtu<<"=";
  if (((h->last_op == OP_GEN_ST) || (h->last_op == OP_CONT_W) || (h->last_op == OP_CONT_G) || (h->last_op == OP_CONT_SKIP)) && (h->cur_mtu != NULL)) print_mtu(h->cur_mtu->mtu);
  else cout<<"___\t";
  cout << "\tqueue_head="<<(uint32_t)h->queue_head;
  cout << "\tn="<<(uint32_t)h->n;
  cout << "\tZ="<<(uint32_t)h->Z;
  cout << "\tcov "<<(uint32_t)h->cov_vec_count<<"="; print_coverage(*h->cov_vec, info->N, h->Z);
  cout << "\t#gaps="<<h->gaps->size();
  cout << "\tcost="<<h->cost;
  cout << "\tprev="<<h->prev;
  cout << endl;
}
*/

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

      for (posn m=0; m<MAX_PHRASE_LEN; m++)
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

  if (here > MAX_SENTENCE_LENGTH)
    return false;

  if ((here > n) && (! is_covered(h, here))) {
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
    h->cost = info->compute_cost(info, h);

    mtuid tm = h->last_op;
    switch (h->last_op) {
    case OP_JUMP_B:
      tm = OP_MAXIMUM + (mtuid)((size_t) h->cur_mtu);
      break;
      
    case OP_GEN_S:
      tm = OP_MAXIMUM + MAX_GAPS + info->sent[h->n - 1];
      break;

    case OP_GEN_T:
      tm = OP_MAXIMUM + MAX_GAPS + MAX_VOCAB_SIZE + 0; // TODO: target lexeme;
      break;

    case OP_GEN_ST:
      tm = OP_MAXIMUM + MAX_GAPS + 2 * MAX_VOCAB_SIZE + h->cur_mtu->mtu->ident;
      break;
    }

    memmove(h->tm_context, h->tm_context+1, (TM_CONTEXT_LEN-1) * sizeof(mtuid));
    h->tm_context[TM_CONTEXT_LEN-1]= tm;

    h->tm_context_hash = util::MurmurHashNative(h->tm_context, TM_CONTEXT_LEN * sizeof(mtuid), 0);

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

bool is_gap_option(gap_op_t gap_option, posn i) {
  if ((i < 0) || (i >= 31)) return false;
  return (gap_option & (1 << i)) != 0;
}

void set_gap_option(gap_op_t &gap_option, posn i) {
  if ((i < 0) || (i >= 31)) return;
  gap_option |= (1 << i);
}

void expand(translation_info *info, hypothesis *h, function<void(hypothesis*)> add_operation) {

  // first check to see if the queue is empty
  bool queue_empty = true;
  if (((h->last_op == OP_GEN_ST) || (h->last_op == OP_CONT_W) || (h->last_op == OP_CONT_G) || (h->last_op == OP_CONT_SKIP)) &&
      (h->cur_mtu != NULL) && 
      (h->queue_head < h->cur_mtu->mtu->src_len))
    queue_empty = false;
      
  posn N = info->N;
  posn n = h->n;
  posn num_uncovered = N - h->cov_vec_count;

  if (! queue_empty) { // then all we can do is CONTINUE
    if (n < N && !is_covered(h, n)) {
      mtu_item *mtu = h->cur_mtu->mtu;
      lexeme lex = mtu->src[ h->queue_head ];

      if (! is_gap_option(h->cur_mtu->mtu->gap_option, h->queue_head) ) {  // no GAP option
        if (info->sent[n] == lex) {
          hypothesis *h2 = next_hypothesis(info, h);
          h2->last_op = OP_CONT_W;
          h2->queue_head++;
          set_covered(h2, n);
          h2->n++;
          if (prepare_for_add(info, h2)) add_operation(h2);
        }
      } else { // we can have a gap BEFORE the current word (== h->queue_head)
        for (size_t idx=0; idx<NUM_MTU_OPTS; idx++) {
          posn here = h->cur_mtu->found_at[h->queue_head][idx];

          bool okay_for_gap = is_ok_lex_position(info, h, n+1, idx, here);
          bool okay_for_skip = okay_for_gap;
          if (! okay_for_skip)
            okay_for_skip = is_ok_lex_position(info, h, n, idx, here);

          if (okay_for_gap &&
              (h->gaps->size() < MAX_GAPS) && (h->gaps->size() < num_uncovered)) {
            // here is okay for BOTH skip and GAP because it passed with n+1
            hypothesis *h2 = next_hypothesis(info, h);
            h2->last_op = OP_CONT_G;
            if (! h2->gaps_alloc) {
              h2->gaps = new set<posn>( *h->gaps );
              h2->gaps_alloc = true;
            }
            h2->gaps->insert(h->n);
            h2->queue_head ++;
            h2->n = here;
            set_covered(h2, here);
            h2->n++;
            if (prepare_for_add(info, h2)) add_operation(h2);
          }
          if (okay_for_skip) {
            hypothesis *h2 = next_hypothesis(info, h);
            h2->last_op = OP_CONT_SKIP;
            h2->queue_head += 2;
            h2->n = here;
            set_covered(h2, here);
            h2->n++;
            if (prepare_for_add(info, h2)) add_operation(h2);
          }
        }
      }
    }
  } else { // the queue IS empty
    if (n < N && !is_covered(h, n)) { // try generating a new cept
      vector<mtu_for_sent*> mtus = info->mtus_at[n];
      for (auto &mtu : mtus) {
        hypothesis *h2 = next_hypothesis(info, h);
        h2->last_op = OP_GEN_ST;
        set_covered(h2, n);
        h2->n++;
        h2->cur_mtu = mtu;
        h2->queue_head = 1;
        shift_lm_context(info, h2, mtu->mtu->tgt_len, mtu->mtu->tgt);
        if (prepare_for_add(info, h2)) add_operation(h2);
      }
    }

    if ((n < N) && 
        (!is_covered(h, n)) && 
        ((h->last_op != OP_GAP))) { // generate just S
      hypothesis *h2 = next_hypothesis(info, h);
      h2->last_op = OP_GEN_S;
      set_covered(h2, n);
      h2->n++;
      h2->cur_mtu = NULL;
      h2->queue_head = 1;
      if (prepare_for_add(info, h2)) add_operation(h2);
    }

    if ((h->last_op != OP_JUMP_B) &&
        (h->gaps->size() < MAX_GAPS) &&
        (h->gaps->size() < num_uncovered) &&
        (!is_covered(h, n))) {
      hypothesis *h2 = next_hypothesis(info, h);
      h2->cur_mtu = NULL;
      h2->last_op = OP_GAP;
      if (! h2->gaps_alloc) {
        h2->gaps = new set<posn>( *h->gaps );
        h2->gaps_alloc = true;
      }
      h2->gaps->insert(h->n);
      h2->n++;
      if (prepare_for_add(info, h2)) add_operation(h2);
    }

    { // TODO: GEN_T

    }

    if ((h->n == h->Z) &&
        (h->last_op != OP_GAP)) {
      size_t gap_id = 0;
      for (auto &gap_pos : *h->gaps) {
        gap_id++;
        if (gap_pos+1 != h->n) {
          hypothesis *h2 = next_hypothesis(info, h);
          h2->cur_mtu = (mtu_for_sent*)gap_id;
          h2->last_op = OP_JUMP_B;
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
    
    if ( (h->n < h->Z) && (h->last_op != OP_JUMP_B) ) {
      hypothesis *h2 = next_hypothesis(info, h);
      h2->cur_mtu = NULL;
      h2->last_op = OP_JUMP_E;
      h2->n = h2->Z;
      if (prepare_for_add(info, h2)) add_operation(h2);
    }
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

float get_pruning_threshold(translation_info *info, vector<hypothesis*> stack) {
  float min_cost = FLT_MAX;
  if (info->pruning_coefficient >= 1.) {
    for (auto &h : stack)
      if (h->cost < min_cost)
        min_cost = h->cost;
    min_cost *= info->pruning_coefficient;
  }
  return min_cost;
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
      if (memcmp(h->tm_context, h2->tm_context, TM_CONTEXT_LEN * sizeof(mtuid)) != 0)
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

void recombine_stack(translation_info *info, vector<hypothesis*> stack) {
  recombination_data * buckets = info->recomb_buckets;
  size_t mod = buckets->size();

  for (auto vec = buckets->begin(); vec != buckets->end(); vec++)
    (*vec).clear();

  for (auto h : stack) {
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

pair< vector<hypothesis*>, vector<hypothesis*> > stack_generic_search(translation_info *info, size_t (*get_stack_id)(hypothesis*), size_t num_stacks_reserve=0) {
  unordered_map< size_t, vector<hypothesis*>* > Stacks;
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
  Stacks[stack0] = new vector<hypothesis*>();
  Stacks[stack0]->push_back(h0);

  while (! NextStacks.empty()) {
    size_t cur_stack = NextStacks.top(); NextStacks.pop();
    if (Stacks[cur_stack] == NULL) continue;

    float prune_if_gt = get_pruning_threshold(info, *(Stacks[cur_stack]));
    recombine_stack(info, Stacks[cur_stack][0]);
    
    auto it = Stacks[cur_stack]->begin();
    while (it != Stacks[cur_stack]->end()) {
      hypothesis *h = *it;
      if (h->cost > prune_if_gt) { it++; continue; }
      if (h->skippable) { it++; continue; }

      size_t diff = it - Stacks[cur_stack]->begin();
      expand(info, h, [&](hypothesis* next) mutable -> void {
          if (is_final_hypothesis(info, next))
            Goals.push_back(next);
          else { // not final
            size_t next_stack = get_stack_id(next);
            if (next_stack == cur_stack)
              add_to_stack(info, *Stacks[cur_stack], next);
            else { // different stack
              if (Stacks[next_stack] == NULL) {
                Stacks[next_stack] = new vector<hypothesis*>();
                NextStacks.push(next_stack);
              }
              Stacks[next_stack]->push_back(next);
            }
          }
        });
      it = Stacks[cur_stack]->begin() + diff;
      it++;
    }
  }

  for (auto it : Stacks) delete it.second;

  return { Goals, visited };
}

pair< vector<hypothesis*>, vector<hypothesis*> > stack_search(translation_info *info) {
  hypothesis *h0 = next_hypothesis(info, NULL);

  stack<hypothesis*> S;
  vector<hypothesis*> visited;
  vector<hypothesis*> Goals;

  S.push(h0);
  
  while (!S.empty()) {
    hypothesis *h = S.top(); S.pop();
    expand(info, h, [info, &Goals, &S](hypothesis*next) mutable -> void {    
        if (is_final_hypothesis(info, next))
          Goals.push_back(next);
        else
          S.push(next);
      });
    visited.push_back(h);
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
      set_gap_option(mtu->gap_option, j);
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

void get_operation_sequence(aligned_sentence_pair data) {
  auto f = data.F;
  auto E = data.E;
  auto A = data.A;
  vector< pair<char, pair<lexeme,lexeme> > > op_seq;
  set<posn> gaps;

  posn i,j,j2,k,N,Z;
  N = E.size();
  i = 0; j = 0; k = 0;
  
  bitset<MAX_SENTENCE_LENGTH> fcov;

  while ((j < f.size()) && is_unaligned(A, j)) {
    op_seq.push_back( { OP_GEN_S, { f[j], 0 } } );
    fcov[j] = true;
    j++;
  }
  Z = j;

  while (i < N) {
    j2 = A[i][k];
    cout << "i=" << (uint32_t)i << " k=" << (uint32_t)k << " j2=" << (uint32_t)j2 << endl;
    if (j < j2) {
      if (! fcov[j]) {
        cout << "gap k=" << (uint32_t)k << " A.size=" << (uint32_t)A[i].size() << endl;
        if ((k > 0) && (k < A[i].size()))
          op_seq.push_back( { OP_CONT_G, { j, 0 } } );
        else
          op_seq.push_back( { OP_GAP, { j, 0 } } );
        gaps.insert(j);
      }
      if (j == Z) 
        j = j2;
      else {
        op_seq.push_back( { OP_JUMP_E, { 0, 0 } } );
        j = Z;
      }
    }

    if (j2 < j) {
      if ((j < Z) && !fcov[j]) {
        if ((k > 0) && (k < A[i].size()))
          op_seq.push_back( { OP_CONT_G, { j, 0 } } );
        else
          op_seq.push_back( { OP_GAP, { j, 0 } } );
        gaps.insert(j);
      }

      posn W = 0;
      bool found = true;
      for (auto &pos : gaps) {
        if (pos == A[i][k])
          found = true;
        else if (pos > A[i][k])
          W++;
      }
      assert(found);
      gaps.erase(A[i][k]);
      op_seq.push_back( { OP_JUMP_B, { W, 0 } } );
      j = A[i][k];
    }

    if (j < j2) {
      if ((k > 0) && (k < A[i].size()))
        op_seq.push_back( { OP_CONT_G, { j, 0 } } );
      else
        op_seq.push_back( { OP_GAP, { j, 0 } } );
      gaps.insert(j);
      j = j2;
    }
    if (k == 0) {
      op_seq.push_back( { OP_GEN_ST, { i, k } } );
      fcov[j] = true;
    } else {
      op_seq.push_back( { OP_CONT_W, { i, k } } );
      fcov[j] = true;
    }
    j ++;
    k ++;

    while ((j < f.size()) && is_unaligned(A, j)) {
      bool in_gap = (k > 0) && (k < A[i].size());
      op_seq.push_back( { OP_GEN_S, { f[j], in_gap } } );
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
    op_seq.push_back( { OP_JUMP_E, { 0, 0 } } );
    j = Z;
  }

  for (auto &op : op_seq)
    cout << "op=" << OP_NAMES[(uint32_t)op.first] << "\targ=(" << op.second.first << ", " << op.second.second << ")" << endl;
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

void print_mtu_half(const lexeme*d, posn len, gap_op_t gap_option=0) {
  for (posn i=0; i<len; i++) {
    if (i > 0) cout << " ";
    cout << d[i];
    if (is_gap_option(gap_option, i))
        cout << " _";
  }
}

void print_mtu_set(unordered_map<mtu_item,uint32_t> mtus, bool renumber_mtus=false) {
  size_t id = 0;
  for (auto mtu_gap : mtus) {
    if (renumber_mtus) {
      cout << id << "\t";
      id++;
    } else
      cout << mtu_gap.first.ident << "\t";

    print_mtu_half(mtu_gap.first.src, mtu_gap.first.src_len, mtu_gap.second);
    cout << " | ";
    print_mtu_half(mtu_gap.first.tgt, mtu_gap.first.tgt_len);
    cout << endl;
  }
}


bool read_mtu_item_from_file(FILE* fd, mtu_item& mtu) {
  char line[255];
  size_t nr;

  nr = fscanf(fd, "%d\t", &mtu.ident);
  //cerr<<"ident="<<mtu.ident<<", nr="<<nr<<endl;
  if ((nr == 0) || (feof(fd))) return true;
  mtu.gap_option = 0;
  posn i=0;
  while (true) {
    nr = fscanf(fd, "%[^| ] ", line);
    //cerr << "first: line='"<<line<<"', nr="<<nr<<", gap_option="<<mtu.gap_option<<endl;
    if (nr == 0) break;
    if ((line[0] == '_') && (line[1] == 0))
      set_gap_option(mtu.gap_option, i-1);
    else
      mtu.src[i++] = atoi(line);
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
  }
  return dict;
}


void collect_mtus(aligned_sentence_pair spair, unordered_map< mtu_item, uint32_t > &cur_mtus, size_t &skipped_for_len) {
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

    if (ephr.size() > MAX_PHRASE_LEN) { skipped_for_len++; continue; }

    mtu_item mtu;
    memset(&mtu, 0, sizeof(mtu));
    mtu.tgt_len = ephr.size();
    for (posn j=0; j<ephr.size(); j++)
      mtu.tgt[j] = ephr[j];

    if (al.size() > MAX_PHRASE_LEN) { skipped_for_len++; continue; }

    mtu.src_len = al.size();
    uint32_t my_gaps = 0;
    for (posn j=0; j<al.size(); j++) {
      mtu.src[j] = F[al[j]];

      if ((j < al.size()-1) && (al[j] != al[j+1]-1))
        set_gap_option(my_gaps, j);
    }
    mtu.ident = 0;
    
    auto it = cur_mtus.find(mtu);
    if (it == cur_mtus.end())
      cur_mtus.insert( { mtu, my_gaps } );
    else {
      uint32_t& old_val = cur_mtus[mtu];
      old_val |= my_gaps;
    }
  }
}

aligned_sentence_pair read_next_aligned_sentence(FILE *fd) {
  assert(! feof(fd));

  lexeme cnt, w,x;
  size_t nr;

  nr = fscanf(fd, "%d", &cnt);
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

  get_operation_sequence( {F, E, A} );
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
  info.N       = 6;
  info.sent[0] = (uint32_t)'A';
  info.sent[1] = (uint32_t)'B';
  info.sent[2] = (uint32_t)'C';
  info.sent[3] = (uint32_t)'A';
  info.sent[4] = (uint32_t)'B';
  info.sent[5] = (uint32_t)'C';
  build_sentence_mtus(&info, dict);
  info.compute_cost = simple_compute_cost;
  info.language_model = NULL; // new lm::ngram::Model((char*)"file.arpa-bin");

  info.operation_allowed =
    (1 << OP_INIT  ) |
    (1 << OP_GEN_ST) |
    (1 << OP_CONT_W) |
    (1 << OP_CONT_G) |
    (1 << OP_GEN_S ) |
    (1 << OP_GEN_T ) |
    (1 << OP_GAP   ) |
    (1 << OP_JUMP_B) |
    (1 << OP_JUMP_E) |
    (1 << OP_CONT_SKIP) |
    0;

  info.pruning_coefficient = 0.;

  //pair< vector<hypothesis*>, vector<hypothesis*> > GoalsVisited = greedy_search(&info);

  info.recomb_buckets = new recombination_data(10231);

  for (size_t rep = 0; rep < 1 + 0*999; rep++) {
    cerr<<".";
    info.hyp_ring = initialize_hypothesis_ring(INIT_HYPOTHESIS_RING_SIZE);

    pair< vector<hypothesis*>, vector<hypothesis*> > GoalsVisited = 
      // for search based on amount of coverage
      stack_generic_search(&info, [](hypothesis* hyp) { return (size_t)hyp->cov_vec_count; }, info.N*2);
      // for search based on (hash of) coverage vector
      //stack_generic_search(&info, [](hypothesis* hyp) { return (size_t)hyp->cov_vec_hash; }, info.N*100);



    for (auto &hyp : GoalsVisited.first) {
      vector<lexeme> trans = get_translation(&info, hyp);

      cout<<hyp->cost<<"\t";
      for (auto &w : get_translation(&info, hyp))
        cout<<" "<<(char)w;
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

void main_collect_mtus(char* filename) {
  FILE *fd = fopen(filename, "r");
  assert(fd != 0);

  size_t sent_id = 0;
  size_t next_print = 100;
  unordered_map<mtu_item, gap_op_t > cur_mtus;
  size_t skipped_for_len = 0;
  while (!feof(fd)) {
    sent_id++;
    if (sent_id == next_print) {
      cerr << "reading sentence pair " << sent_id << endl;
      next_print *= 2;
    }

    aligned_sentence_pair spair = read_next_aligned_sentence(fd);
    collect_mtus(spair, cur_mtus, skipped_for_len);
  }
  fclose(fd);

  cerr << "collected " << cur_mtus.size() << " mtus" << endl;
  cerr << "skipped " << skipped_for_len << " for length" << endl;

  print_mtu_set(cur_mtus, true);
}

int main(int argc, char*argv[]) {
  //test_align();
  //test_decode();
  //test_lm();
  //main_collect_mtus(argv[1]);
  FILE *fd = fopen("test/test.ngdec.mtus", "r");
  mtu_item_dict dict = read_mtu_item_dict(fd);
  fclose(fd);
  
  fd = fopen("test/test.ngdec", "r");
  while (!feof(fd)) {
    aligned_sentence_pair spair = read_next_aligned_sentence(fd);
    get_operation_sequence( spair );
  }
  
  //print_mtu_set(dict, false);
}

  
