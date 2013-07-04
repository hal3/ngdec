#!/usr/bin/perl -w
use strict;

my $USAGE = "usage: prep-test.pl (options) [model.ini] < src-test-data\nwhere [options] includes:\n  --tgt file      use known translations in this file\n  --aln file      use known alignments in this file (presupposes --tgt)\n";

my $INI=''; my $TGT=''; my $ALN='';
while (1) {
    my $tmp = shift or last;
    if    ($tmp eq '--tgt') { $TGT = shift or die "$tmp requires an arguments!"; }
    elsif ($tmp eq '--aln') { $ALN = shift or die "$tmp requires an arguments!"; }
    else { $INI = $tmp; last; }
}

if (($ALN ne '') && ($TGT eq '')) { die "--aln requires --tgt"; }

if (not defined $INI) { die $USAGE; }

my %opts = ();
open F, $INI or die "cannot read $INI";
while (<F>) {
    chomp;
    if (/^([^=]+)=(.+)$/) {
        $opts{$1} = $2;
    }
    elsif (/^\s*$/) { next; }
    elsif (/^\#/) { next; }
    else { die "malformed ini line: $_"; }
}
close F or die;

print STDERR "you can call ngdec with:\n% ngdec predict-lm " . (join ' ', ($opts{'lm_tgt_bin'}, $opts{'tm_bin'}, $opts{'mtus'}, 'test-file')) . "\n";

#requireOpt(\%opts, 'lm_src_arpa');

my %v = ();
my $V = 0;
my $Unk = 0;
open F, mkfilename( $opts{'lm_src_arpa'} );
my $inUnigrams = 0;
while (<F>) {
    chomp;
    if (/^\\1-grams:/) { 
        $inUnigrams = 1;
    } elsif ($inUnigrams) {
        if (/^\s*$/) { last; }
        if (/^\\2-grams:/) { last; }
        my ($prob, $word, $backoff) = split;
        if (not defined $word) { die "malformed LM"; }
        $v{$word} = $V;
        if ($word eq '<unk>') { $Unk = $V; }
        $V++;
    }
}
close F;

my %vt = ();
$V = 0;
if ($TGT ne '') {
    open F, mkfilename( $opts{'lm_tgt_arpa'} );
    $inUnigrams = 0;
    while (<F>) {
        chomp;
        if (/^\\1-grams:/) { 
            $inUnigrams = 1;
        } elsif ($inUnigrams) {
            if (/^\s*$/) { last; }
            if (/^\\2-grams:/) { last; }
            my ($prob, $word, $backoff) = split;
            if (not defined $word) { die "malformed LM"; }
            $vt{$word} = $V;
            $V++;
        }
    }
    close F;
}


if ($ALN ne '') { open A, mkfilename($ALN); }
if ($TGT ne '') { open T, mkfilename($TGT); }

while (<>) {
    chomp;
    my @w = split;
    print (scalar @w);
    for (my $i=0; $i<@w; $i++) {
        print ' ';
        if (defined $v{$w[$i]}) {
            print $v{$w[$i]};
        } else {
            print $Unk;
        }
    }

    if ($TGT eq '') { print "\t0\t0\n"; next; }

    $_ = <T>;
    if (not defined $_) { print STDERR "warning: not enough lines in $TGT\n"; print "\t0\t0\n"; next; }
    chomp;
    @w = split;
    print (scalar @w);
    for (my $i=0; $i<@w; $i++) {
        print ' ';
        if (defined $vt{$w[$i]}) {
            print $vt{$w[$i]};
        } else {
            print $Unk;
        }
    }

    if ($ALN eq '') { print "\t0\n"; next; }

    $_ = <A>;
    if (not defined $_) { print STDERR "warning: not enough lines in $ALN\n"; print "\t0\n"; next; }
    my @a = split;
    print (scalar @a);
    print $_ . "\n";
}
    

if ($ALN ne '') { close A; }
if ($TGT ne '') { close T; }


sub mkfilename {
    my ($fn) = @_;
    if ($fn =~ /\.gz$/) { return "zcat $fn |"; }
    else { return $fn; }
}
