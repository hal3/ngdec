**********************************************************************
*************************** BUILDING NGDEC ***************************
**********************************************************************

-- Setup KenLM --

 1. Download and install the kenlm source somewhere
 2. Run his ./compile_query_only.sh
      (you might have to add -lrt to the linker steps)
 3. Create a kenlm archive by running:
    ar rvs kenlm.a util/double-conversion/bignum.o util/double-conversion/bignum-dtoa.o util/double-conversion/cached-powers.o util/double-conversion/diy-fp.o util/double-conversion/double-conversion.o util/double-conversion/fast-dtoa.o util/double-conversion/fixed-dtoa.o util/double-conversion/strtod.o util/bit_packing.o util/ersatz_progress.o util/exception.o util/file.o util/file_piece.o util/mmap.o util/murmur_hash.o util/pool.o util/read_compressed.o util/scoped.o util/string_piece.o util/usage.o lm/bhiksha.o lm/binary_format.o lm/config.o lm/lm_exception.o lm/model.o lm/quantize.o lm/read_arpa.o lm/search_hashed.o lm/search_trie.o lm/sizes.o lm/trie.o lm/trie_sort.o lm/value_build.o lm/virtual_interface.o lm/vocab.o
 4. Copy this archive (kenlm.a) to the ngdec directory

-- Edit our Makefile --

 1. Change KENLM_DIR to point to where you downloaded kenlm
      ($KENLM_DIR/lm must exist)
 2. Change anything else you want

-- Building and running ngdec --

 1. Run make
 2. Run ./ngdec

**********************************************************************
************************ RUNNING A TEST CASE *************************
**********************************************************************

There is some test data in the test/ subdirectory that you can play
with. There is a script there called prep_data.sh that you can use to
do all the data preprocessing. Note that we assume you already have
word alignments. This script assumes that both lmplz and build_binary
from kenlm are in your path.

The end result of this is a file, test.ngdec, that contains (in easy
to read integer format) all the relevant data (en text, fr text and
post-edited word alignments). There are also two binarized language
models, one for en (test.en.arpa-bin) and one for fr (obvious name).



