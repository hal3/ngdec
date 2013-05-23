#include "mtu.h"

mtu_dict downselect_dict(mtu_dict dict, uint32_t* src, uint32_t N) {
  mtu_dict ret;

  for (uint32_t n=0; n<N; n++) {
    uint32_t src_hash = hash_initialize();
    for (uint32_t m = 0; (m < MAX_PHRASE_LEN) && (n+m) < N; m++) {
      src_hash = hash_increment(src_hash, src[n+m]);
      // src_hash is now a hash of src[n..n+m]
      
      auto iter = dict.find(src_hash);
      if (iter != dict.end()) {
        // we found potential mtus
        vector<mtu*> mtus = iter->second;
        ret.insert({src_hash, mtus});
      }
    }
  }

  return ret;
}

uint32_t hash_initialize() { return 3289021; }
uint32_t hash_increment(uint32_t hash, uint32_t new_item) { return hash*930211 + new_item; }
