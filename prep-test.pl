#!/usr/bin/perl -w
use strict;
use ngdec;

$|++;

my $USAGE = 
    "usage: prep-test.pl (options) [model.ini] < src-test-data\n" .
    "where [options] includes:\n" .
    "  --tgt file      use known translations in this file\n" .
    "  --aln file      use known alignments in this file (presupposes --tgt)\n" .
    "  --tabsep        read ALL from stdin, src\\ttgt\\taln\n" .
    "  --savev file    save the local vocab to this file\n" .
    "  --deaccent enc  override deaccent spec in model.ini (enc=none for none)\n";

my $INI=''; my $TGT=''; my $ALN=''; my $SRC=''; my $TABSEP=0; my $SAVEV = '';
my $overrideDeaccent = '';

while (1) {
    my $tmp = shift or last;
    if    ($tmp eq '--tgt') { $TGT = shift or die "$tmp requires an arguments!"; }
    elsif ($tmp eq '--aln') { $ALN = shift or die "$tmp requires an arguments!"; }
    elsif ($tmp eq '--savev'){$SAVEV=shift or die "$tmp requires an arguments!"; }
    elsif ($tmp eq '--deaccent'){$overrideDeaccent=shift or die "$tmp requires an arguments!"; }
    elsif ($tmp eq '--tabsep') { $TABSEP = 1; }
    else { $INI = $tmp; last; }
}

if (($ALN ne '') && ($TGT eq '')) { die "--aln requires --tgt"; }
if ($TABSEP && (($TGT ne '') || ($ALN ne ''))) { die "--tabsep must not be combined with --tgt or --aln"; }
if ((not defined $INI) || ($INI eq '')) { die $USAGE; }

my %opts = readINI($INI);

requireOpt(\%opts, 'lm_src_arpa');
requireOpt(\%opts, 'lm_tgt_arpa');

my %srcV = read_lm_vocab($opts{'lm_src_arpa'});
my %tgtV = ();
if (($TGT ne '') || ($TABSEP)) { %tgtV = read_lm_vocab($opts{'lm_tgt_arpa'}); }

my %localV = ();
if ((defined $opts{'extra_vocab'}) && (-e $opts{'extra_vocab'})) {
    %localV = read_local_vocab($opts{'extra_vocab'});
}

my $deaccent = '';
if (defined $opts{'deaccent'}) { $deaccent = $opts{'deaccent'}; }
if ($overrideDeaccent ne '') {
    if ($overrideDeaccent eq 'none') { $deaccent = ''; }
    else { $deaccent = $overrideDeaccent; }
}

if ($ALN ne '') { open A, mkfilename($ALN); }
if ($TGT ne '') { open T, mkfilename($TGT); }

my $vocabFile;
if ($SAVEV ne '') {
    local *VOC;
    open VOC, "> $SAVEV" or die "cannot open $SAVEV for writing";
    $vocabFile = *VOC{IO};
}

while (my $inLine = <>) {
    chomp $inLine;

    my $src; my $tgt; my $aln;
    if ($TABSEP) {
        if ($inLine =~ /^([^\t]+)\t([^\t]+)\t([^\t]+)$/) {
            $src = $1; $tgt = $2; $aln = $3;
        } elsif ($inLine =~ /^([^\t]+)\t([^\t]+)$/) {
            $src = $1; $tgt = $2;
        } else {
            $src = $inLine;
        }
    } else { $src = $inLine; }

    if ($TGT ne '') {
        $tgt = <T>;
        if (not defined $tgt) { die "not enough lines in $TGT"; }
        chomp $tgt;
    }

    # process target first because we want to look up unaccented source against target
    # (sigh, yes, this assumes we're translating into English...)
    my $tgtIds = ((defined $tgt) ? stringToIDs($deaccent, \%srcV, \%tgtV, \%localV, 0, $vocabFile, $tgt) : '0');
    my $srcIds = stringToIDs($deaccent, \%srcV, \%tgtV, \%localV, 1, $vocabFile, $src);

    print $srcIds . "\t" . $tgtIds;

    if ($ALN ne '') {
        $aln = <A>;
        if (not defined $aln) { die "not enough lines in $ALN"; }
        chomp $aln;
    }
    if (defined $aln) {
        $aln =~ s/^\s+//; $aln =~ s/\s+$//;
        my @a = split /\s+/, $aln;
        print "\t" . (scalar @a) . ' ' . $aln . "\n";
    } else {
        print "\t0\n";
    }
}

if ($ALN ne '') { close A; }
if ($TGT ne '') { close T; }

if (defined $vocabFile) { close $vocabFile; }
