#!/usr/bin/perl -w
use strict;
use ngdec;

$|++;

my $USAGE = 
    "usage: prep-test.pl (options) [model.ini] < src-test-data\n" .
    "where [options] includes:\n" .
    "  --tgt file      use known translations in this file\n" .
    "  --aln file      use known alignments in this file (presupposes --tgt)\n" .
    "  --tagsep        read ALL from stdin, src\\ttgt\\taln\n" .
    "  --savev file    save the local vocab to this file\n";

my $INI=''; my $TGT=''; my $ALN=''; my $SRC=''; my $TAGSEP=0; my $SAVEV = '';
while (1) {
    my $tmp = shift or last;
    if    ($tmp eq '--tgt') { $TGT = shift or die "$tmp requires an arguments!"; }
    elsif ($tmp eq '--aln') { $ALN = shift or die "$tmp requires an arguments!"; }
    elsif ($tmp eq '--savev'){$SAVEV=shift or die "$tmp requires an arguments!"; }
    elsif ($tmp eq '--tagsep') { $TAGSEP = 1; }
    else { $INI = $tmp; last; }
}

if (($ALN ne '') && ($TGT eq '')) { die "--aln requires --tgt"; }
if ($TAGSEP && (($TGT ne '') || ($ALN ne ''))) { die "--tagsep must not be combined with --tgt or --aln"; }
if ((not defined $INI) || ($INI eq '')) { die $USAGE; }

my %opts = readINI($INI);

requireOpt(\%opts, 'lm_src_arpa');
requireOpt(\%opts, 'lm_tgt_arpa');

my %srcV = read_lm_vocab($opts{'lm_src_arpa'});
my %tgtV = ();
if (($TGT ne '') || ($TAGSEP)) { %tgtV = read_lm_vocab($opts{'lm_tgt_arpa'}); }

if ($ALN ne '') { open A, mkfilename($ALN); }
if ($TGT ne '') { open T, mkfilename($TGT); }

my $vocabFile;
if ($SAVEV ne '') {
    local *VOC;
    open VOC, "> $SAVEV" or die "cannot open $SAVEV for writing";
    $vocabFile = *VOC{IO};
}

my %localV = ();
while (my $inLine = <>) {
    chomp $inLine;

    my $src; my $tgt; my $aln;
    if ($TAGSEP) {
        if ($inLine =~ /^([^\t]+)\t([^\t]+)\t([^\t]+)$/) {
            $src = $1; $tgt = $2; $aln = $3;
        } else { die "malformed line in input: '$inLine'"; }
    } else { $src = $inLine; }

    print stringToIDs(\%srcV, \%localV, $src, 1, $vocabFile);

    if ($TGT ne '') {
        $tgt = <T>;
        if (not defined $tgt) { die "not enough lines in $TGT"; }
        chomp $tgt;
    }
    print "\t" . ((defined $tgt) ? stringToIDs(\%tgtV, \%localV, $tgt, 0, $vocabFile) : '0');

    if ($ALN ne '') {
        $aln = <A>;
        if (not defined $aln) { die "not enough lines in $ALN"; }
        chomp $aln;
    }
    if (defined $aln) {
        $aln =~ s/^\s+//; $aln =~ s/\s+$//;
        my @a = split /\s+/, $aln;
        print "\t" . (scalar @a) . $aln . "\n";
    } else {
        print "\t0\n";
    }
}

if ($ALN ne '') { close A; }
if ($TGT ne '') { close T; }

if (defined $vocabFile) { close $vocabFile; }

