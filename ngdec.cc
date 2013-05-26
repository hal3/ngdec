#include <iostream>
#include <assert.h>
#include <float.h>
#include <stack>
#include "ngdec.h"
#include "string.h"

//#include<boost/coroutine/generator.hpp>
//using namespace coro = boost::coroutines;


#define MAX(a,b) (((a)>(b))?(a):(b))

template <posn N>
void print_coverage(bitset<N> cov, posn cursor) {
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

template<posn N>
posn get_translation_length(hypothesis<N> *h) {
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

template<posn N>
vector<lexeme> get_translation(translation_info<N> *info, hypothesis<N> *h) {
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


template<posn N>
bool is_covered(hypothesis<N> *h, posn n) {
  return (*(h->cov_vec))[n];
}

template<posn N>
void set_covered(hypothesis<N> *h, posn n) {
  assert(! ((*(h->cov_vec))[n]) );
  if (! h->cov_vec_alloc) {
    assert( h->prev != NULL );
    h->cov_vec = new bitset<N>(*h->prev->cov_vec);
    h->cov_vec_alloc = true;
  }
  (*(h->cov_vec))[n] = true;
  h->cov_vec_count++;
}

template<posn N>
hypothesis<N>* next_hypothesis(hypothesis<N> *h) {
  hypothesis<N> *h2 = (hypothesis<N>*) calloc(1, sizeof(hypothesis<N>));

  if (h == NULL) {  // asking for initial hypothesis
    h2->last_op = OP_INIT;
    h2->cur_mtu = NULL;
    h2->queue_head = 0;
    h2->cov_vec = new bitset<N>();
    h2->cov_vec_count = 0;
    h2->cov_vec_alloc = true;
    h2->n = 0;
    h2->Z = 0;
    h2->gaps = new set<posn>();
    h2->gaps_alloc = true;
    h2->cost = 0.;
    h2->prev = NULL;
  } else {
    memcpy(h2, h, sizeof(hypothesis<N>));
    h2->last_op = OP_UNKNOWN;
    h2->cov_vec_alloc = false;
    h2->gaps_alloc = false;
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

template<posn N>
void print_hypothesis(hypothesis<N> *h) {
  cout << h;
  cout << "\tlast_op="<<OP_NAMES[(size_t)h->last_op];
  cout << "\tcur_mtu="<<h->cur_mtu<<"=";
  if (h->cur_mtu != NULL) print_mtu(h->cur_mtu->mtu);
  else cout<<"___\t";
  cout << "\tqueue_head="<<(uint32_t)h->queue_head;
  cout << "\tn="<<(uint32_t)h->n;
  cout << "\tZ="<<(uint32_t)h->Z;
  cout << "\tcov "<<(uint32_t)h->cov_vec_count<<"="; print_coverage<N>(*h->cov_vec, h->Z);
  cout << "\t#gaps="<<h->gaps.size();
  cout << "\tcost="<<h->cost;
  cout << "\tprev="<<h->prev;
  cout << endl;
}

template<posn N>
void free_hypothesis(hypothesis<N> *h) {
  if (h->gaps_alloc)
    delete h->gaps;
    
  if (h->cov_vec_alloc)
    delete h->cov_vec;

  free(h);
}

void free_sentence_mtus(vector< vector<mtu_for_sent*> > sent_mtus) {
  for (auto &it1 : sent_mtus)
    for (auto &it2 : it1)
      free(it2);
}

template<posn N>
void build_sentence_mtus(translation_info<N> *info, mtu_item_dict dict) {
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

template<posn N>
vector<posn> get_lex_positions(translation_info<N> *info, const mtu_for_sent *mtu, posn queue_head, posn n) {
  // mtu->mtu->src[queue_head] is a word
  // we want to know all "future" positions where this word occurs
  vector<posn> posns;
  for (size_t i=0; i<NUM_MTU_OPTS; i++) {
    posn here = mtu->found_at[queue_head][i];
    if (here > MAX_SENTENCE_LENGTH)
      break;
    if (here > n) {
      // TODO: check to make sure it's "completeable"
      posns.push_back(here);
    }
  }
  return posns;
}


template<posn N>
void add_operation(translation_info<N>*info, vector<hypothesis<N>*> &ret, hypothesis<N>*h) {
  if ((info->operation_allowed & (1 << h->last_op)) > 0)
    ret.push_back(h);
  else
    free_hypothesis(h);
}

template<posn N>
vector<hypothesis<N>*> expand(translation_info<N> *info, hypothesis<N> *h) {
  vector<hypothesis<N>*> ret;

  // first check to see if the queue is empty
  bool queue_empty = true;
  if ((h->cur_mtu != NULL) && (h->queue_head < h->cur_mtu->mtu->src_len))
    queue_empty = false;
      
  posn n = h->n;
  posn num_uncovered = N - h->cov_vec_count;

  if (! queue_empty) { // then all we can do is CONTINUE
    if (n < N && !is_covered(h, n)) {
      mtu_item *mtu = h->cur_mtu->mtu;
      lexeme lex = mtu->src[ h->queue_head ];
      if (lex != GAP_LEX) {  // this is not a GAP
        if (info->sent[n] == lex) {
          hypothesis<N> *h2 = next_hypothesis(h);
          h2->last_op = OP_CONT_W;
          h2->queue_head++;
          set_covered(h2, n);
          h2->n++;
          add_operation(info, ret, h2);
        }
      } else if ((h->gaps->size() < MAX_GAPS) &&
                 (h->gaps->size() < num_uncovered)) {
        lex = mtu->src[ h->queue_head+1 ]; // now this is the NEXT word
        vector<posn> lex_pos = get_lex_positions(info, h->cur_mtu, h->queue_head+1, n);
        for (auto &j : lex_pos) {
          if (!is_covered(h, j)) {
            hypothesis<N> *h2 = next_hypothesis(h);
            h2->last_op = OP_CONT_G;
            if (! h2->gaps_alloc) {
              h2->gaps = new set<posn>( *h->gaps );
              h2->gaps_alloc = true;
            }
            h2->gaps->insert(h->n);
            h2->queue_head += 2;
            h2->n = j;
            set_covered(h2, j);
            h2->n++;
            add_operation(info, ret, h2);
          }
        }
      }
    }
  } else { // the queue IS empty
    if (n < N && !is_covered(h, n)) { // try generating a new cept
      vector<mtu_for_sent*> mtus = info->mtus_at[n];
      for (auto &mtu : mtus) {
        hypothesis<N> *h2 = next_hypothesis(h);
        h2->last_op = OP_GEN_ST;
        set_covered(h2, n);
        h2->n++;
        h2->cur_mtu = mtu;
        h2->queue_head = 1;
        add_operation(info, ret, h2);
      }
    }

    if ((n < N) && 
        (!is_covered(h, n)) && 
        ((h->last_op != OP_GAP))) { // generate just S
      hypothesis<N> *h2 = next_hypothesis(h);
      h2->last_op = OP_GEN_S;
      set_covered(h2, n);
      h2->n++;
      h2->cur_mtu = NULL;
      h2->queue_head = 1;
      add_operation(info, ret, h2);
    }

    if ((h->last_op != OP_JUMP_B) &&
        (h->gaps->size() < MAX_GAPS) &&
        (h->gaps->size() < num_uncovered) &&
        (!is_covered(h, n))) {
      hypothesis<N> *h2 = next_hypothesis(h);
      h2->cur_mtu = NULL;
      h2->last_op = OP_GAP;
      if (! h2->gaps_alloc) {
        h2->gaps = new set<posn>( *h->gaps );
        h2->gaps_alloc = true;
      }
      h2->gaps->insert(h->n);
      h2->n++;
      add_operation(info, ret, h2);
    }

    { // TODO: GEN_T

    }

    if ((h->n == h->Z) &&
        (h->last_op != OP_GAP)) {
      for (auto &gap_pos : *h->gaps) {
        if (gap_pos+1 != h->n) {
          hypothesis<N> *h2 = next_hypothesis(h);
          h2->cur_mtu = NULL;
          h2->last_op = OP_JUMP_B;
          h2->n = gap_pos;
          if (! h2->gaps_alloc) {
            h2->gaps = new set<posn>( *h->gaps );
            h2->gaps_alloc = true;
          }
          h2->gaps->erase(gap_pos);
          add_operation(info, ret, h2);
        }
      }
    }
    
    if ( (h->n < h->Z) && (h->last_op != OP_JUMP_B) ) {
      hypothesis<N> *h2 = next_hypothesis(h);
      h2->cur_mtu = NULL;
      h2->last_op = OP_JUMP_E;
      h2->n = h2->Z;
      add_operation(info, ret, h2);
    }
  }

  for (auto &h : ret) {
    h->Z = MAX(h->Z, h->n);
    h->cost = info->compute_cost(info, h);
  }

  return ret;
}

template<posn N>
bool is_final_hypothesis(translation_info<N> *info, hypothesis<N> *h) {
  if (h->n < N) return false;
  if (h->Z < N) return false;
  for (posn n=0; n<N; n++)
    if (! is_covered(h, n))
      return false;
  return true;
}

template<posn N>
float get_pruning_threshold(translation_info<N> *info, vector<hypothesis<N>*> stack) {
  float min_cost = FLT_MAX;
  if (info->pruning_coefficient >= 1.) {
    for (auto &h : stack)
      if (h->cost < min_cost)
        min_cost = h->cost;
    min_cost *= info->pruning_coefficient;
  }
  return min_cost;
}

template<posn N>
pair< vector<hypothesis<N>*>, vector<hypothesis<N>*> > stack_covlen_search(translation_info<N> *info) {
  hypothesis<N> *h0 = next_hypothesis<N>(NULL);

  vector< vector<hypothesis<N>*> > Stacks(N+1);
  vector< hypothesis<N>* > visited;
  vector< hypothesis<N>* > Goals;

  Stacks[ h0->cov_vec_count ].push_back(h0);
  
  for (posn covered=0; covered<=N; covered++) {
    //cout<<"==== COVERED " << ((uint32_t)covered) << " ===="<<endl<<endl;;
    float prune_if_gt = get_pruning_threshold(info, Stacks[covered]);

    for (uint32_t id=0; id<Stacks[covered].size(); id++) {  // do it this way because Stacks[covered] might grow!
      hypothesis<N> *h = Stacks[covered][id];
      if (h->cost > prune_if_gt) continue;
      //cout<<"expand ("<<id<<"/"<<Stacks[covered].size()<<"): "; print_hypothesis(h);
      // hypothesis<N> *me = h->prev;
      // while (me != NULL) {
      //   cout<<"     -> "; print_hypothesis(me);
      //   me = me->prev;
      // }

      for (auto &next : expand(info, h)) {
        //cout<<"  next: "; print_hypothesis(next);
        if (is_final_hypothesis(info, next))
          Goals.push_back(next);
        else {
          if (! (( next->cov_vec_count == id ) && ( next->cost > prune_if_gt )) )
            Stacks[ next->cov_vec_count ].push_back(next);
        }
      }
      //cout<<endl;

      visited.push_back(h);
    }
  }

  visited.insert( visited.end(), Goals.begin(), Goals.end() );
  return { Goals, visited };
}


template<posn N>
pair< vector<hypothesis<N>*>, vector<hypothesis<N>*> > greedy_search(translation_info<N> *info) {
  hypothesis<N> *h0 = next_hypothesis<N>(NULL);

  stack<hypothesis<N>*> S;
  vector<hypothesis<N>*> visited;
  vector<hypothesis<N>*> Goals;

  S.push(h0);
  
  while (!S.empty()) {
    hypothesis<N> *h = S.top(); S.pop();
    for (auto &next : expand(info, h))
      if (is_final_hypothesis(info, next))
        Goals.push_back(next);
      else
        S.push(next);
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

void mtu_add_item_string(mtu_item_dict*dict, string src, string tgt) {
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

template<posn N>
float simple_compute_cost(void*info_, hypothesis<N>*h) {
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
  mtu_add_item_string(&dict, "A_B", "ab");
  mtu_add_item_string(&dict, "A_C", "ac");
  mtu_add_item_string(&dict, "A" , "a");
  mtu_add_item_string(&dict, "B" , "b");
  mtu_add_item_string(&dict, "B_C", "bc");
  mtu_add_item_string(&dict, "C" , "c");

  const posn N = 6;
  translation_info<N> info;
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
    0;

  info.pruning_coefficient = 0.;

  //pair< vector<hypothesis*>, vector<hypothesis*> > GoalsVisited = greedy_search(&info);

  for (size_t rep = 0; rep < 1000; rep++) {

  pair< vector<hypothesis<N>*>, vector<hypothesis<N>*> > GoalsVisited = stack_covlen_search(&info);


  for (auto &hyp : GoalsVisited.first) {
    vector<lexeme> trans = get_translation(&info, hyp);

    cout<<hyp->cost<<"\t";
    for (auto &w : get_translation(&info, hyp))
      cout<<" "<<(char)w;
    cout<<endl;
    // hypothesis<N> *me = hyp;
    // while (me != NULL) {
    //   cout<<"\t"; print_hypothesis(me);
    //   me = me->prev;
    // }
    // cout<<endl; 

  }
  for (auto &hyp : GoalsVisited.second)
    free_hypothesis(hyp);

  }

  free_sentence_mtus(info.mtus_at);
  free_dict(dict);
}

int main(int argc, char*argv[]) {
  //test_align();
  test_decode();
  return 0;
}

/*
with expand creating a temporary list, and using bitset

time ./ngdec  > /dev/null

real	3m55.024s
user	3m30.633s
sys	0m24.162s
*/
