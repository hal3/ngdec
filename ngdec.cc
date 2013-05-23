#include <stdio.h>
#include <iostream>
#include <string>
#include <stack>
#include <vector>
#include "mtu.h"
#include "hyp.h"

using namespace std;

void mtu_add_unit(mtu_dict*dict, mtu*unit) {
  auto it = dict->find(unit->src_hash);
  if (it == dict->end()) {
    vector<mtu*> mtus;
    mtus.push_back(unit);
    dict->insert( { unit->src_hash, mtus } );
  } else {
    vector<mtu*> mtus = it->second;
    mtus.push_back(unit);
  }
}

void mtu_add_item_string(mtu_dict*dict, string src, string tgt) {
  mtu *unit = (mtu*)calloc(1, sizeof(mtu));
  unit->src_len = src.length();
  unit->src_hash = hash_initialize();
  for (uint32_t n=0; n<unit->src_len; n++) {
    unit->src[n] = (uint32_t)src[n];
    unit->src_hash = hash_increment(unit->src_hash, unit->src[n]);
  }

  unit->tgt_len = tgt.length();
  unit->tgt_hash = hash_initialize();
  for (uint32_t n=0; n<unit->tgt_len; n++) {
    unit->tgt[n] = (uint32_t)tgt[n];
    unit->tgt_hash = hash_increment(unit->tgt_hash, unit->tgt[n]);
  }

  mtu_add_unit(dict, unit);
}

void print_mtus(vector<const mtu*> mtus) {
  for (auto it = mtus.begin(); it != mtus.end(); it++) {
    if (it != mtus.begin())
      cout << " ";

    if ((*it) == NULL)
      cout << "?";
    else
      for (uint32_t m=0; m<(*it)->tgt_len; m++)
        cout << (char)(*it)->tgt[m];
  }
  cout << "\n";
}

void free_dict(mtu_dict dict) {
  for (auto it=dict.begin(); it!=dict.end(); it++) {
    vector<mtu*> mtus = (*it).second;
    for (auto mtu_it=mtus.begin(); mtu_it!=mtus.end(); mtu_it++)
      free(*mtu_it);
  }
}

float simple_compute_cost(void*data_, hyp*h) {
  tdata*data = (tdata*)data_;
  return h->prev->cost + 1.f;
}

int main(int argc, char*argv[]) {
  mtu_dict dict;
  mtu_add_item_string(&dict, "A" , "a");
  mtu_add_item_string(&dict, "AB", "ab");
  mtu_add_item_string(&dict, "B" , "b");
  mtu_add_item_string(&dict, "BC", "bc");
  mtu_add_item_string(&dict, "C" , "c");

  tdata data;
  data.N = 5;
  data.src[0] = (uint32_t)'A';
  data.src[1] = (uint32_t)'B';
  data.src[2] = (uint32_t)'C';
  data.src[3] = (uint32_t)'B';
  data.src[4] = (uint32_t)'C';
  data.dict = downselect_dict(dict, data.src, data.N);
  data.compute_cost = simple_compute_cost;

  data.allow_reordering = false;
  data.pruning_coefficient = 0.;

  //pair< vector<hyp*>, vector<hyp*> > GoalsVisited = single_vector_decode(&data);
  pair< vector<hyp*>, vector<hyp*> > GoalsVisited = coverage_vector_decode(&data);

  for (auto it=GoalsVisited.first.begin(); it!=GoalsVisited.first.end(); it++) {
    vector<const mtu*> mtus = hypothesis_mtus(*it);
    cout<<(*it)->cost<<"\t";
    print_mtus(mtus);
  }
  for (auto it=GoalsVisited.second.begin(); it!=GoalsVisited.second.end(); it++) {
    free(*it);
  }

  free_dict(dict);

  return 0;
}
