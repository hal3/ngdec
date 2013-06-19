#!/usr/bin/perl -w
use strict;
use ngdec;

my $USE_LM = '';
my $USE_LM_BIN = '';
my $LM_ORDER = 5;
my $TM_ORDER = 9;

my $USAGE = '' .
    "usage: train-model.pl [options] [par-src] [par-tgt] [par-aln] [outdir]\n" .
    "where [options] includes:\n" .
    "  --use_lm [file]          instead of building a language model from\n" .
    "                           [par-tgt], use the specified language model,\n" .
    "                           which must be in ARPA format\n" .
    "  --use_lm_bin [file]      use this file as a binary LM (requires --use_lm)\n" .
    "  --lm_order [N]           set LM order to N (def: $LM_ORDER)\n" .
    "  --tm_order [N]           set TM order to N (def: $TM_ORDER)\n" .
    "  --savev file             save the local vocab to this file\n" .
    '';

my $SRC;
my $SAVEV = '';
while (1) {
    my $tmp = shift or last;
    if    ($tmp eq '--use_lm'  ) { $USE_LM   = shift or die "$tmp requires an argument!"; }
    elsif ($tmp eq '--use_lm_bin'){$USE_LM_BIN=shift or die "$tmp requires an argument!"; }
    elsif ($tmp eq '--lm_order') { $LM_ORDER = shift or die "$tmp requires an argument!"; }
    elsif ($tmp eq '--tm_order') { $TM_ORDER = shift or die "$tmp requires an argument!"; }
    elsif ($tmp eq '--savev')    { $SAVEV    = shift or die "$tmp requires an argument!"; }
    elsif ($tmp =~ /^-/) { die "unknown option: '$tmp'"; }
    else { $SRC = $tmp; last; }
}
my $TGT = shift or die $USAGE;
my $ALN = shift or die $USAGE;
my $OUT = shift or die $USAGE;

if (($USE_LM_BIN ne '') && ($USE_LM eq '')) { die "--use_lm_bin requires --use_lm also"; }

my @requiredFiles = ($SRC, $TGT, $ALN);
if ($USE_LM ne '') { push @requiredFiles, $USE_LM; }
if ($USE_LM_BIN ne '') { push @requiredFiles, $USE_LM_BIN; }

foreach my $file (@requiredFiles) {
    if (! -e $file) { die "file does not exist '$file'"; }
}
if (! -d $OUT) { die "directory does not exist '$OUT'"; }


my $LMPLZ = `which lmplz`;        chomp $LMPLZ; if ($LMPLZ eq '') { die "error: cannot find lmplz"; }
my $B_BIN = `which build_binary`; chomp $B_BIN; if ($B_BIN eq '') { die "error: cannot find build_binary"; }

my $AnythingChanged = 0;

my $LM_TGT_ARPA = build_arpa_lm('tgt', $TGT, $LM_ORDER, $USE_LM);
my $LM_TGT_BIN  = build_binary_lm('tgt', $LM_TGT_ARPA, $USE_LM_BIN);

my $LM_SRC_ARPA = build_arpa_lm('src', $SRC, 2, '');
my $LM_SRC_BIN  = build_binary_lm('src', $LM_SRC_ARPA, '');

my $ALN_PE = postedit_alignments();

#my $TGT_ID = map_to_ids('tgt', $TGT, $LM_TGT_ARPA);
#my $SRC_ID = map_to_ids('src', $SRC, $LM_SRC_ARPA);

my $vocabFile;
if ($SAVEV ne '') {
    local *VOC;
    open VOC, "> $SAVEV" or die "cannot open $SAVEV for writing";
    $vocabFile = *VOC{IO};
}

my $NGDEC  = combine_all();

if (defined $vocabFile) { close $vocabFile; }

my $MTUS0 = run("$OUT/mtus", "./ngdec extract $NGDEC > {}");
my $MTUS  = run("$OUT/mtus-pruned", "./prune_mtu_dict.pl < $MTUS0 > {}");

my $OPSEQ = run("$OUT/opseq", "./ngdec oracle $MTUS $NGDEC > {}");

my $TM_ARPA = build_arpa_lm('ops', $OPSEQ, $TM_ORDER, '');
my $TM_BIN  = build_binary_lm('ops', $TM_ARPA, '');



print STDERR "writing to $OUT/model.ini\n";

my %opts = ( 'lm_tgt_arpa' => $LM_TGT_ARPA,
             'lm_tgt_bin'  => $LM_TGT_BIN,
             'lm_src_arpa' => $LM_SRC_ARPA,
             'lm_src_bin'  => $LM_SRC_BIN,
             'lm_order'    => $LM_ORDER,
             'tm_order'    => $TM_ORDER,
             'train_data'  => $NGDEC,
             'mtus'        => $MTUS,
             'train_ops'   => $OPSEQ,
             'tm_arpa'     => $TM_ARPA,
             'tm_bin'      => $TM_BIN );

open O, ">$OUT/model.ini" or die;
foreach my $k (sort keys %opts) { print O $k . "=" . $opts{$k} . "\n"; }
close O or die;

print STDERR "done!\n";

sub build_arpa_lm {
    my ($fnameStr, $inFile, $order, $uselm) = @_;
    if ($uselm ne '') {
        open F, mkfilename($uselm) or die "cannot read lm: $uselm";
        $_ = <F>;
        if (not defined $_) { die "cannot read from $uselm"; }
        if (/^\\data/) {  # it's arpa format
            return $uselm;
        }
        die "$uselm in unknown LM format";
    }

    # otherwise, make it ourselves
    return run("$OUT/$fnameStr.arpa", "$LMPLZ -o $order -S 10% < $inFile > {}");
}

sub build_binary_lm {
    my ($fnameStr, $inFile, $uselm) = @_;
    if ($uselm ne '') {
        return $uselm;
    } else {
        return run("$OUT/$fnameStr.arpa-bin", "$B_BIN $inFile {}");
    }
}

sub postedit_alignments {
    return run("$OUT/aln", "./postedit_alignments.pl $TGT $ALN {}");
}

sub combine_all {
    if ((-e "$OUT/train") && (not $AnythingChanged)) {
        print STDERR "skipping: $OUT/train // combine_all\n";
        return "$OUT/train";
    }
    $AnythingChanged = 1;

    print STDERR "running: combine_all > $OUT/train\n";

    my %srcV = read_lm_vocab($LM_SRC_ARPA);
    my %tgtV = read_lm_vocab($LM_TGT_ARPA);

    open O, "> $OUT/train" or die;

    open F, mkfilename($SRC) or die;
    open E, mkfilename($TGT) or die;
    open A, mkfilename($ALN_PE) or die;

    while (1) {
        my %localV = ();

        $_ = <F>; if (not defined $_) { last; }
        print O stringToIDs(\%srcV, \%localV, $_, 1, $vocabFile);

        $_ = <E>; if (not defined $_) { die "too few lines in $TGT"; }
        print O "\t" . stringToIDs(\%tgtV, \%localV, $_, 0, $vocabFile);

        $_ = <A>; if (not defined $_) { die "too few lines in $ALN_PE"; }
        s/\s+/ /g; s/^ //; s/ $//;
        my @t = split;
        print O "\t" . (scalar @t) . ' ' . $_ . "\n";
    }

    while (<E>) { die "too many lines in $TGT"; }
    while (<A>) { die "too many lines in $ALN_PE"; }

    close A;
    close E;
    close F;

    close O;

    return "$OUT/train";
}

sub run {
    my ($outFile, $cmd) = @_;

    if ((-e $outFile) && (not $AnythingChanged)) {
        print STDERR "skipping: $outFile // '$cmd'\n";
        return $outFile;
    }
    
    $cmd =~ s/\{\}/$outFile/g;
    print STDERR "running: '$cmd'\n";
    my $ret = system $cmd;

    if ($ret != 0) {
        if ($? == -1) {
            print "failed to execute: $!\n";
        } elsif ($? & 127) {
            printf "child died with signal %d, %s coredump\n", ($? & 127), ($? & 128) ? 'with' : 'without';
        } else {
            printf "child exited with value %d\n", $? >> 8;
        }
        exit(-1);
    }

    $AnythingChanged = 1;
    return $outFile;
}
