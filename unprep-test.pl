#!/usr/bin/perl -w
use strict;

my $USAGE = "usage: unprep-test.pl [model.ini] < tgt-word-ids\n";
my $INI = shift or die $USAGE;

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

my %v = ();
my $V = 0;
open F, mkfilename( $opts{'lm_tgt_arpa'} );
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
        $v{$V} = $word;
        $V++;
    }
}
close F;

while (<>) {
    chomp;
    s/^[^\t]*\t//;
    my @w = split;
    for (my $i=0; $i<@w; $i++) {
        if ($i > 0) { print ' '; }
        if (defined $v{$w[$i]}) {
            print $v{$w[$i]};
        } else {
            print '*UNK*';
        }
    }
    print "\n";
}
    
sub mkfilename {
    my ($fn) = @_;
    if ($fn =~ /\.gz$/) { return "zcat $fn |"; }
    else { return $fn; }
}
