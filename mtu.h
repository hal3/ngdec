#ifndef mtu_h
#define mtu_h

#define MAX_PHRASE_LEN   3

#include <stdio.h>
#include <stdint.h>
#include <unordered_map>
#include <vector>

using namespace std;

struct mtu {
  uint32_t src_len;
  uint32_t tgt_len;

  uint32_t src[MAX_PHRASE_LEN];
  uint32_t tgt[MAX_PHRASE_LEN];

  uint32_t src_hash;
  uint32_t tgt_hash;

  bool operator==(const mtu&rhs) const {
    return false;
  }
};

typedef unordered_map< uint32_t, vector< mtu* > > mtu_dict;  // from src_hash to a set of mtus

uint32_t hash_initialize();
uint32_t hash_increment(uint32_t hash, uint32_t new_item);

mtu_dict downselect_dict(mtu_dict, uint32_t*, uint32_t);

#endif
