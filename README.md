**********************************************************************
*************************** BUILDING NGDEC ***************************
**********************************************************************

-- Setup KenLM --

 1. Download and install the kenlm source somewhere

 2a. (OPTIONAL) compile kenlm with larger LM limit:
                  --max-kenlm-order=9

 2b. (REQUIRED) Run his ./compile_query_only.sh ... you should 
     definitely switch the max order to at least 9 in the shell
     script by setting -DKENLM_MAX_ORDER=9
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

We can now instruct ngdec to extract minimal translation units (MTUs)
from the alignment (.ngdec) file:

% ./ngdec extract test/test.ngdec > test/test.mtus

This should give you a dictionary of 17733 MTUs, 226 skipped for
length. If you want longer ones, add "--max_phrase_length 10" (for
instance) after the "extract" command. We first prune it down:

% ./prune_mtu_dict.pl < test/test.mtus > test/test.mtus.pruned

Now that we have the dictionary, we can extract operation sequences as
follows:

% ./ngdec oracle test/test.mtus.pruned test/test.ngdec > test/test.opseq

As a sanity check, we can do "forced decoding" to see how close we can
get to the English reference translations:

% ./ngdec predict-forced test/test.mtus.pruned test/test.ngdec  | grep -n FAIL

You should see that it failed on sentence pairs 302 and 1390 (this is
because those two sentences require gaps larger than the model is
willing to consider).

We can (finally) train an operation sequence model on the opseq data
using kenlm. We'll build a 5-gram model:

% lmplz -o 5 -S 10% < test/test.opseq > test/test.opseq.arpa
% build_binary test/test.opseq.arpa test/test.opseq.arpa-bin

And no we can decode:

% ./ngdec predict-lm test/test.en.arpa-bin test/test.opseq.arpa-bin test/test.mtus.pruned test/test.ngdec
