#!/bin/bash

LM_ORDER=3

if [[ "`which lmplz`" == "" ]] ; then
  echo error: cannot find lmplz
  exit -1
fi
if [[ "`which build_binary`" == "" ]] ; then
  echo error: cannot find build_binary
  exit -1
fi

# clean up data
rm -f test.en.* test.fr.* 2>/dev/null

# monolingual stuff
echo =======================================================
echo ===== building English monolingual language model =====
echo =======================================================
lmplz -o $LM_ORDER -S 10% < test.en > test.en.arpa
build_binary test.en.arpa test.en.arpa-bin
echo

echo ======================================================
echo ===== building French monolingual language model =====
echo ======================================================
lmplz -o $LM_ORDER -S 10% < test.fr > test.fr.arpa
build_binary test.fr.arpa test.fr.arpa-bin
echo

echo =========================================
echo ===== mapping words to IDs: English =====
echo =========================================
./word_to_id.pl test.en.arpa < test.en > test.en.id
echo

echo ========================================
echo ===== mapping words to IDs: French =====
echo ========================================
./word_to_id.pl test.fr.arpa < test.fr > test.fr.id
echo

# post-edit alignments per Durrani, Schmid & Fraser ACL2011
echo ===================================
echo ===== post-editing alignments =====
echo ===================================
./postedit_alignments.pl test

echo
echo SANITY CHECK: the following unaligned tokens, nonconsec alns and reattached toks should be zero
echo
cp test.en test.sanity.en
cp test.al.pe test.sanity.al
./postedit_alignments.pl test.sanity
rm -f test.sanity* 2>/dev/null
echo

# munge everything together
echo ========================================================
echo ===== combining into a single, easy to C-read file =====
echo ========================================================
./combine_all.pl test.fr.id test.en.id test.al.pe > test.ngdec
echo

echo done!
