#include <iostream>
#include <functional>
#include <assert.h>
#include <float.h>
#include <stack>
#include "ngdec.h"
#include "string.h"
#include "lm/model.hh"

namespace ng=lm::ngram;

#define MAX(a,b) (((a)>(b))?(a):(b))

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

void print_mtu(const mtu_item*mtu) {
  if (mtu == NULL)
    cout << "?";
  else
    for (uint32_t m=0; m<mtu->tgt_len; m++)
      cout << (char)mtu->tgt[m];
}

void print_mtus(vector<const mtu_item*> mtus) {
  for (auto it = mtus.begin(); it != mtus.end(); it++) {
    if (it != mtus.begin())
      cout << " ";
    print_mtu(*it);
  }
  cout << "\n";
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
    for (size_t i=0; i<r->next_hypothesis; i++) {
      hypothesis *h = r->my_hypotheses + i;
      free(h->tm_context);
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

template<class T>
uint32_t hash_context(T *context, size_t length) {
  uint32_t hash = 13984321;
  for (size_t i=0; i<length; i++)
    hash = (hash * 3910489) + context[i];
  return hash;
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
    h2->lm_context = new ng::State(info->language_model.BeginSentenceState());
    h2->lm_context_alloc = true;
    h2->lm_context_hash = ng::hash_value(*h2->lm_context);
    h2->tm_context = (mtuid*) calloc(TM_CONTEXT_LEN, sizeof(mtuid));
    memset(h2->tm_context, OP_INIT, TM_CONTEXT_LEN);
    h2->tm_context_hash = hash_context<mtuid>(h2->tm_context, TM_CONTEXT_LEN);
    h2->skippable = false;
    h2->cost = 1.;
    h2->prev = NULL;
  } else {
    memcpy(h2, h, sizeof(hypothesis));
    h2->last_op = OP_UNKNOWN;
    h2->cov_vec_alloc = false;
    h2->gaps_alloc = false;
    h2->lm_context_alloc = false;

    h2->tm_context = (mtuid*) calloc(TM_CONTEXT_LEN, sizeof(mtuid));
    memcpy(h2->tm_context, h->tm_context, TM_CONTEXT_LEN-1);
    h2->tm_context_hash = hash_context<mtuid>(h2->tm_context, TM_CONTEXT_LEN);  // TODO: could be faster

    h2->skippable = false;  // TODO: maybe can get rid of this???
    h2->prev = h;


    /*
    h2->cur_mtu = h->cur_mtu;
    h2->queue_head = h->queue_head;
    h2->cov_vec = h->cov_vec;
    h2->cov_vec_count = h->cov_vec_count;
    h2->n = h->n;
    h2->Z = h->Z;
    h2->gaps = h->gaps;
    h2->cost = h->cost;
    */
  }

  return h2;
}

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

    vector<mtu_item*> *mtus = dict_entry->second;
    for (auto mtu_iter=mtus->begin(); mtu_iter!=mtus->end(); mtu_iter++) {
      mtu_for_sent* mtu = (mtu_for_sent*)calloc(1, sizeof(mtu_for_sent));
      mtu->mtu = *mtu_iter;

      for (posn m=0; m<MAX_PHRASE_LEN; m++)
        for (posn i=0; i<NUM_MTU_OPTS; i++)
          mtu->found_at[m][i] = MAX_SENTENCE_LENGTH+1;

      mtu->found_at[0][0] = n;
      posn last_position = n;
      bool success = true;
      for (posn i=1; i<mtu->mtu->src_len; i++) {
        if (mtu->mtu->src[i] == GAP_LEX) {
          last_position++;
        } else {
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
          if (mtu->mtu->src[q_ptr] == GAP_LEX)
            continue;
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
    h->tm_context_hash = hash_context<mtuid>(h->tm_context, TM_CONTEXT_LEN);

    return true;
  } else {
    return false;
  }
}

float shift_lm_context(translation_info* info, hypothesis *h, posn M, lexeme*tgt) {
  float log_prob = 0.;

  if (!h->lm_context_alloc) {
    h->lm_context = new ng::State(*(h->lm_context));
    h->lm_context_alloc = true;
  }
  ng::State in_state = *(h->lm_context);

  in_state = *(h->lm_context);
  for (posn m=0; m<M; m++) {
    log_prob += info->language_model.Score(in_state, tgt[m], *(h->lm_context));
    in_state = *(h->lm_context);
  }
  h->lm_context_hash = ng::hash_value(*h->lm_context);

  return log_prob;
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
      if (lex != GAP_LEX) {  // this is not a GAP
        if (info->sent[n] == lex) {
          hypothesis *h2 = next_hypothesis(info, h);
          h2->last_op = OP_CONT_W;
          h2->queue_head++;
          set_covered(h2, n);
          h2->n++;
          if (prepare_for_add(info, h2)) add_operation(h2);
        }
      } else {
        for (size_t idx=0; idx<NUM_MTU_OPTS; idx++) {
          posn here = h->cur_mtu->found_at[h->queue_head+1][idx];
          lex = mtu->src[ h->queue_head+1 ]; // now this is the NEXT word

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
            h2->queue_head += 2;
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

pair< vector<hypothesis*>, vector<hypothesis*> > stack_covlen_search(translation_info *info) {
  posn N = info->N;
  hypothesis *h0 = next_hypothesis(info, NULL);

  vector< vector<hypothesis*> > Stacks(N+1);
  vector< hypothesis* > visited;
  vector< hypothesis* > Goals;

  Stacks[ h0->cov_vec_count ].push_back(h0);
  
  for (posn covered=0; covered<=N; covered++) {
    //cout<<"==== COVERED " << ((uint32_t)covered) << " ===="<<endl<<endl;;
    float prune_if_gt = get_pruning_threshold(info, Stacks[covered]);
    if (true) recombine_stack(info, Stacks[covered]);

    for (uint32_t id=0; id<Stacks[covered].size(); id++) {  // do it this way because Stacks[covered] might grow!
      hypothesis *h = Stacks[covered][id];
      if (h->cost > prune_if_gt) continue;
      if (h->skippable) continue;

      //cout<<"expand ("<<id<<"/"<<Stacks[covered].size()<<"): "; print_hypothesis(info, h);
      // hypothesis *me = h->prev;
      // while (me != NULL) {
      //   cout<<"     -> "; print_hypothesis(info, me);
      //   me = me->prev;
      // }

      expand(info, h, [info, &Goals, covered, prune_if_gt, &Stacks](hypothesis*next) mutable -> void {
          //cout<<"  next: "; print_hypothesis(info, next);
          if (is_final_hypothesis(info, next))
            Goals.push_back(next);
          else if (next->cov_vec_count == covered) {
            if ( next->cost <= prune_if_gt )
              //Stacks[ covered ].push_back(next);
              add_to_stack(info, Stacks[ covered ], next);
          } else
            Stacks[ next->cov_vec_count ].push_back(next);
          //else
          //  free_hypothesis(next);
        });
      //cout<<endl;

      visited.push_back(h);
    }
  }

  visited.insert( visited.end(), Goals.begin(), Goals.end() );
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


void mtu_add_unit(mtu_item_dict*dict, mtu_item*mtu) {
  auto it = dict->find(mtu->src[0]);
  if (it == dict->end()) {
    vector<mtu_item*> * mtus = new vector<mtu_item*>;
    mtus->push_back(mtu);
    dict->insert( { mtu->src[0], mtus } );
  } else {
    vector<mtu_item*> * mtus = it->second;
    mtus->push_back(mtu);
  }
}

void mtu_add_item_string(mtu_item_dict*dict, mtuid ident, string src, string tgt) {
  mtu_item *mtu = (mtu_item*)calloc(1, sizeof(mtu_item));
  mtu->src_len = src.length();
  for (uint32_t n=0; n<mtu->src_len; n++) {
    if (src[n] == '_')
      mtu->src[n] = GAP_LEX;
    else
      mtu->src[n] = (uint32_t)src[n];
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
    vector<mtu_item*> *mtus = (*it).second;
    for (auto mtu_it=mtus->begin(); mtu_it!=mtus->end(); mtu_it++)
      free(*mtu_it);
    delete mtus;
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

void get_operation_sequence(vector< vector<lexeme> > E, vector<lexeme> f, vector< vector<posn> > A) {
  vector< pair<char, pair<lexeme,lexeme> > > op_seq;
  set<posn> gaps;

  posn i,j,j2,k,N,Z;
  N = E.size();
  i = 0; j = 0; k = 0;
  
  bitset<MAX_SENTENCE_LENGTH> fcov;

  while ((j < f.size()) && is_unaligned(A, j)) {
    op_seq.push_back( { OP_GEN_S, { f[j], 0 } } );
    //cout << "gen_s " << f[j] << endl;
    fcov[j] = true;
    j++;
  }
  Z = j;

  while (i < N) {
    j2 = A[i][k];
    if (j < j2) {
      if (! fcov[j]) {
        op_seq.push_back( { OP_UNKNOWN, { j, 0 } } ); // "insert gap" -- need to fix later!
        //cout << "insert gap1" << endl;
        gaps.insert(j);
      }
      if (j == Z)
        j = j2;
      else {
        op_seq.push_back( { OP_JUMP_E, { 0, 0 } } );
        //cout << "jump end" << endl;
        j = Z;
      }
    }

    if (j2 < j) {
      if ((j < Z) && !fcov[j]) {
        op_seq.push_back( { OP_UNKNOWN, { j, 0 } } ); // "insert gap" -- need to fix later!
        //cout << "insert gap2" << endl;
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
      //cout << "jump back " << (uint32_t)W << endl;
      j = A[i][k];
    }

    if (j < j2) {
      op_seq.push_back( { OP_UNKNOWN, { j, 0 } } ); // "insert gap" -- need to fix later!
      //cout << "insert gap3" << endl;
      gaps.insert(j);
      j = j2;
    }
    if (k == 0) {
      //cout << "generate " << (char)f[A[i][k]] << " from " << (uint32_t)i << endl;
      op_seq.push_back( { OP_GEN_ST, { i, k } } );
      fcov[j] = true;
    } else {
      //cout << "continue " << (char)f[A[i][k]] << " from " << (uint32_t)i << endl;
      op_seq.push_back( { OP_CONT_W, { i, k } } );
      fcov[j] = true;
    }
    j ++;
    k ++;

    while ((j < f.size()) && is_unaligned(A, j)) {
      op_seq.push_back( { OP_GEN_S, { f[j], 0 } } );
      //cout << "gen_s " << f[j] << endl;
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
    //cout << "jump end" << endl;
    op_seq.push_back( { OP_JUMP_E, { 0, 0 } } );
    j = Z;
  }

  for (auto &op : op_seq)
    cout << "op=" << OP_NAMES[(uint32_t)op.first] << "\targ=(" << op.second.first << ", " << op.second.second << ")" << endl;

  // now, fix up the gaps (aka unknowns)
  vector< pair<char, pair<lexeme,lexeme> > > ops;
  for (size_t i=0; i<op_seq.size(); i++) {
    if (op_seq[i].first != OP_UNKNOWN)
      ops.push_back(op_seq[i]);
    else {
      assert(i+1 < op_seq.size());
      assert(op_seq[i+1].first == OP_CONT_W);
      op_seq[i+1].first = OP_CONT_G;
    }
  }
}

void test_align() {
  vector< lexeme >         F = { 'D', 'H', 'E', 'I', 'B', 'G' };
  vector< vector<lexeme> > E = { { 't' }, { 'h' }, { 'r' }, { 'a' }, { 'b' } };
  vector< vector<posn  > > A = { { 0 }, { 2 }, { 1, 5 }, { 3 }, { 4 } };

  get_operation_sequence(E, F, A);
}

void test_decode() {
  mtu_item_dict dict;
  mtu_add_item_string(&dict, 0, "A_B", "ab");
  mtu_add_item_string(&dict, 1, "A_C", "ac");
  mtu_add_item_string(&dict, 2, "A" , "a");
  mtu_add_item_string(&dict, 3, "B" , "b");
  mtu_add_item_string(&dict, 4, "B_C", "bc");
  mtu_add_item_string(&dict, 5, "C" , "c");

  translation_info info("file.arpa-bin");
  info.N       = 6;
  info.sent[0] = (uint32_t)'A';
  info.sent[1] = (uint32_t)'B';
  info.sent[2] = (uint32_t)'C';
  info.sent[3] = (uint32_t)'A';
  info.sent[4] = (uint32_t)'B';
  info.sent[5] = (uint32_t)'C';
  build_sentence_mtus(&info, dict);
  info.compute_cost = simple_compute_cost;

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

  for (size_t rep = 0; rep < 1 + 1*199; rep++) {
    cerr<<".";
    info.hyp_ring = initialize_hypothesis_ring(INIT_HYPOTHESIS_RING_SIZE);

    pair< vector<hypothesis*>, vector<hypothesis*> > GoalsVisited = stack_covlen_search(&info);


    for (auto &hyp : GoalsVisited.first) {
      vector<lexeme> trans = get_translation(&info, hyp);

      cout<<hyp->cost<<"\t";
      for (auto &w : get_translation(&info, hyp))
        cout<<" "<<(char)w;
      cout<<endl;
      // hypothesis *me = hyp;
      // while (me != NULL) {
      //   cout<<"\t"; print_hypothesis(info, me);
      //   me = me->prev;
      // }
      // cout<<endl; 
    }

    free_hypothesis_ring(info.hyp_ring);
  }
  cerr<<endl;
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

int main(int argc, char*argv[]) {
  //test_align();
  test_decode();
  //test_lm();
  return 0;
}

/*
KENLM:
  after running ./compile_query_only.sh
  create an archive:
    ar rvs kenlm.a util/double-conversion/bignum.o util/double-conversion/bignum-dtoa.o util/double-conversion/cached-powers.o util/double-conversion/diy-fp.o util/double-conversion/double-conversion.o util/double-conversion/fast-dtoa.o util/double-conversion/fixed-dtoa.o util/double-conversion/strtod.o util/bit_packing.o util/ersatz_progress.o util/exception.o util/file.o util/file_piece.o util/mmap.o util/murmur_hash.o util/pool.o util/read_compressed.o util/scoped.o util/string_piece.o util/usage.o lm/bhiksha.o lm/binary_format.o lm/config.o lm/lm_exception.o lm/model.o lm/quantize.o lm/read_arpa.o lm/search_hashed.o lm/search_trie.o lm/sizes.o lm/trie.o lm/trie_sort.o lm/value_build.o lm/virtual_interface.o lm/vocab.o
  copy it to the ngdec directory
*/

/*
with expand creating a temporary list, and using bitset

time ./ngdec  > /dev/null   (1000 reps)

real	3m55.024s
user	3m30.633s
sys	0m24.162s

switch to vector<bool>
real	4m36.946s
user	4m11.424s
sys	0m25.226s

switch to bitset<MAX_SENTENCE_LENGTH>
real	4m5.635s
user	3m40.790s
sys	0m24.590s

switch to functional expand
real	4m5.045s
user	3m38.974s
sys	0m25.790s

fixing "gap doesn't consume a position" bug
real	3m50.487s
user	3m25.809s
sys	0m24.438s

adding SKIP GAP
real	3m57.525s
user	3m32.305s
sys	0m24.946s

(switch to 200 reps)
real	0m48.737s
user	0m43.379s
sys	0m5.276s

refactoring skip/gap
real	0m48.483s
user	0m43.091s
sys	0m5.344s

added ngram contexts
real	0m53.590s
user	0m47.935s
sys	0m5.592s

added hypothesis ring (1024)
real	0m43.693s
user	0m38.338s
sys	0m5.300s

growable hypothesis ring
real	0m43.438s
user	0m37.966s
sys	0m5.428s

added history and hashes for covvec and lm and tm
real	1m15.180s
user	1m6.024s
sys	0m34.726s

added hypothesis recombination (5g LM, 9g TM)
real	0m3.510s
user	0m2.948s
sys	0m0.832s

checking recombination on insertion into current stack
real	0m3.158s
user	0m2.916s
sys	0m0.228s

switched to kenlm
real	0m1.992s
user	0m1.860s
sys	0m0.176s


*/
  
