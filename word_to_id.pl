#!/usr/bin/perl -w
use strict;

my $LMname = shift or die;

my %v = ();
my $V = 0;
my $Unk = 0;
open F, "$LMname" or die;
my $inUnigrams = 0;
while (<F>) {
    chomp;
    if (/^\\1-grams:/) { 
        $inUnigrams = 1;
    } elsif ($inUnigrams) {
        if (/^\s*$/) { last; }
        if (/^\\2-grams:/) { last; }
        my ($prob, $word, $backoff) = split;
        if (not defined $word) { die "malformed LM: $LMname"; }
        $v{$word} = $V;
        if ($word eq '<unk>') { $Unk = $V; }
        $V++;
    }
}
close F or die;

while (<>) {
    chomp;
    my @w = split;
    for (my $i=0; $i<@w; $i++) {
        if ($i > 0) { print ' '; }
        if (defined $v{$w[$i]}) {
            print $v{$w[$i]};
        } else {
            print $Unk;
        }
    }
    print "\n";
}

    
