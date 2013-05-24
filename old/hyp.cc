#include <assert.h>
#include <iostream>
#include <vector>
#include <stack>
#include "hyp.h"
#include <float.h>

bool mtu_matches(uint32_t* src, int n, const mtu* unit) {
  for (uint32_t m=0; m<unit->src_len; m++)
    if (src[n+m] != unit->src[m]) {
      //cerr<<"mismatch at "<<n<<"+"<<m<<": "<<src[n+m]<<" != "<<unit->src[m]<<endl;
      return false;
    }
  return true;
}

vector<hyp*> extend_hyp(tdata*data, hyp*h) {
  vector<hyp*> ret;

  // option 1: GEN_ST
  // can only do this in the following states
  //    GEN_ST or GEN_CONT and num_in_queue=0
  //    any other state
  if ( (h->num_in_queue == 0) ||
       ((h->op != OP_GEN_ST) && (h->op != OP_CONT)) ) {
    
    posn n = h->j;

    uint32_t src_hash = hash_initialize();
    for (posn m = 0; (m < MAX_PHRASE_LEN) && (n+m) < data->N; m++) {
      if (h->cov[n+m]) break;
      src_hash = hash_increment(src_hash, data->src[n+m]);

      auto iter = data->dict.find(src_hash);
      if (iter != data->dict.end()) {
        vector<mtu*> mtus = iter->second;
        for (auto mtu_iter = mtus.begin(); mtu_iter != mtus.end(); mtu_iter++) {
          if (mtu_matches(data->src, n, *mtu_iter)) {
            hyp*next = (hyp*)calloc(1, sizeof(hyp));

            next->op = OP_GEN_ST;
            next->cur_mtu = *mtu;
            next->num_in_queue = (*mtu)->src_len - 1;

            next->j = n + 1;
            next->Z = MAX( n+1, h->Z );

            next->cov = covvec(h->cov);
            next->cov[n]
            ret.push_back(next);
          }
        }
      }
    }
  }

  /*
  bool hit_any = false;

  for (uint32_t n=0; n<data->N; n++) {
    if (h->cov[n])
      continue;

    if (hit_any && !data->allow_reordering)
      break;

    hit_any = true;

    uint32_t src_hash = hash_initialize();
    for (uint32_t m = 0; (m < MAX_PHRASE_LEN) && (n+m) < data->N; m++) {
      if (h->cov[n+m]) break;
      src_hash = hash_increment(src_hash, data->src[n+m]);
      // src_hash is now a hash of src[n..n+m]

      //cerr<<"n="<<n<<" m="<<m<<endl;
      auto iter = data->dict.find(src_hash);
      if (iter != data->dict.end()) {
        //cerr<<"found!"<<endl;
        // we found potential mtus
        vector<mtu*> mtus = iter->second;
        for (auto mtu_iter = mtus.begin(); mtu_iter != mtus.end(); mtu_iter++) {
          //cerr<<"iter"<<endl;
          if (mtu_matches(data->src, n, *mtu_iter)) {
            //cerr<<"matches: tgt="<<(char)(*mtu_iter)->tgt[0]<<(char)(*mtu_iter)->tgt[1]<<endl;

            hyp*next = (hyp*)calloc(1, sizeof(hyp));
            next->m = *mtu_iter;
            next->cov = covvec(h->cov);
            for (uint32_t n2=n; n2<=n+m; n2++)
              next->cov[n2] = true;
            assert(h->num_uncov >= m+1);
            next->num_uncov = h->num_uncov - (m+1);
            next->prev = h;
            next->cost = data->compute_cost ? data->compute_cost(data, next) : (h->cost + 1.f);

            next->i  = h->i  + (*mtu_iter)->tgt_len;
            next->j  = h->j  + (*mtu_iter)->src_len;
            next->j2 = h->j2 + (*mtu_iter)->src_len;
            next->Z  = h->Z  + (*mtu_iter)->src_len;

            ret.push_back(next);
          }
        }
      }
    }
    
  }
  */

  //cerr<< "extensions:";
  for (auto it=ret.begin(); it!=ret.end(); it++) {
    //cerr<< " " << (char)(*it)->m->tgt[0] << (char)(*it)->m->tgt[1];
  }
  //cerr<<endl;

  return ret;
}

hyp* initial_hyp(tdata*data) {
  covvec zeroCov;
  hyp*h0 = (hyp*)calloc(1, sizeof(hyp));

  h0->op = OP_INIT;
  h0->cur_mtu = NULL;
  h0->num_in_queue = 0;

  h0->i = 0;
  h0->j = 0;
  h0->j2 = 0;
  h0->Z = 0;

  h0->cov = zeroCov;
  h0->num_uncov = data->N;
  h0->num_gaps = 0;
  memset(h0->gap_positions, 0, MAX_GAPS * sizeof(posn));
  
  h0->prev = NULL;
  h0->cost = 0;

  return h0;
}

pair< vector<hyp*>, vector<hyp*> > single_vector_decode(tdata*data) {
  hyp*h0 = initial_hyp(data);

  stack<hyp*> S;
  vector<hyp*> visited;
  vector<hyp*> Goals;

  S.push(h0);
  
  while (!S.empty()) {
    hyp *h = S.top();
    S.pop();

    vector<hyp*> nextHyp = extend_hyp(data, h);
    for (auto hyp_iter = nextHyp.begin(); hyp_iter != nextHyp.end(); hyp_iter++) {
      if ((*hyp_iter)->num_uncov == 0) {
        Goals.push_back(*hyp_iter);
        visited.push_back(*hyp_iter);
      } else
        S.push(*hyp_iter);
    }
    visited.push_back(h);
  }

  return {Goals, visited};
}

/*
pair< vector<hyp*>, vector<hyp*> > coverage_vector_decode(tdata*data) {
  hyp*h0 = initial_hyp(data);

  vector<vector<hyp*>> Stacks(data->N);
  vector<hyp*> visited;
  vector<hyp*> Goals;

  Stacks[h0->num_uncov-1].push_back(h0);

  for (uint32_t covered=0; covered<data->N; covered++) {
    uint32_t id=data->N-covered-1;
    //cerr<<"id="<<id<<endl;

    float prune_if_gt = FLT_MAX;
    if (data->pruning_coefficient >= 1.f) {
      float min_cost = FLT_MAX;
      for (auto hyp_iter=Stacks[id].begin(); hyp_iter!=Stacks[id].end(); hyp_iter++)
        if ((*hyp_iter)->cost < min_cost)
          min_cost = (*hyp_iter)->cost;
      prune_if_gt = min_cost * data->pruning_coefficient;
    }

    for (uint32_t pos = 0; pos < Stacks[id].size(); pos++) {
      hyp *h = Stacks[id][pos];

      if (h->cost > prune_if_gt) {
        //cerr<<"prune! cost="<<h->cost<<" prune_if_gt="<<prune_if_gt<<endl;
        free(h);
        continue;
      }

      vector<hyp*> nextHyp = extend_hyp(data, h);
      for (auto hyp_iter = nextHyp.begin(); hyp_iter != nextHyp.end(); hyp_iter++) {
        if ((*hyp_iter)->num_uncov == 0) {
          Goals.push_back(*hyp_iter);
          visited.push_back(*hyp_iter);
        } else {
          //cerr<<"  nu="<<(*hyp_iter)->num_uncov<<endl;
          Stacks[(*hyp_iter)->num_uncov-1].push_back(*hyp_iter);
        }
      }
      visited.push_back(h);
    }
  }

  return {Goals, visited};
}

uint32_t hypothesis_length(hyp*h) {
  uint32_t len = 0;
  while (h != NULL) {
    if (h->m != NULL)
      len++;
    h = h->prev;
  }
  return len;
}

vector<const mtu*> hypothesis_mtus(hyp*h) {
  uint32_t len = hypothesis_length(h);
  //cerr<<"len="<<len<<endl;
  vector<const mtu*> mtus(len);
  while (h != NULL) {
    if (h->m != NULL) {
      assert(len>0);
      len--;
      mtus[len] = h->m;
    }
    h = h->prev;
  }
  //cerr<<"done"<<endl;
  assert(len == 0);
  return mtus;
}
*/

