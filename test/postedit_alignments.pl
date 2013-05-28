#!/usr/bin/perl -w
use strict;

my $FNAME = shift or die;

# first, we need to find which target (EN) words are ever unaligned
# and compute their neighbor frequencies, and also get EN word
# frequencies

my %enFreq = ();
my %neighbors = ();
my $numUnaligned = 0;
my $enTokenCount = 0;
open EN, "$FNAME.en" or die;
open AL, "$FNAME.al" or die;
while (my $en = <EN>) {
    my $al = <AL>;
    if (not defined $al) { die "$FNAME.al has fewer lines than $FNAME.en"; }
    chomp $en; my @en = split /\s+/, $en;
    chomp $al; my @al = split /\s+/, $al;
    my %hit = ();
    foreach my $a (@al) {
        my ($f,$e) = split /-/, $a;
        $hit{$e} = 1;
        if ($e >= @en) { die "malformed alignments"; }
    }
    for (my $e=0; $e<@en; $e++) {
        $enFreq{$en[$e]}++;
        $enTokenCount++;
        if (not defined $hit{$e}) {
            $numUnaligned++;
            # this word is unaligned
            if ($e > 0) {
                $neighbors{$en[$e]}{$en[$e-1]}++;
            }
            if ($e < @en-1) {
                $neighbors{$en[$e]}{$en[$e+1]}++;
            }
        }
    }
}
while (<AL>) { die "$FNAME.al has more lines than $FNAME.en"; }
close EN or die;
close AL or die;

print STDERR "  english tokens: $enTokenCount (" . (scalar keys %enFreq) . " word types)\n";
print STDERR "unaligned tokens: $numUnaligned (" . (scalar keys %neighbors) . " word types)\n";


# now, we can do the post-editing
my $countConsecutive = 0;
my $countNonConsecutive = 0;
my $totalFixed = 0;
my $sentenceID = 0;
open EN, "$FNAME.en" or die;
open AL, "$FNAME.al" or die;
open O,  "> $FNAME.al.pe" or die;
while (my $en = <EN>) {
    my $al = <AL>;
    if (not defined $al) { die "$FNAME.al has fewer lines than $FNAME.en"; }
    chomp $en; my @en = split /\s+/, $en;
    chomp $al; my @al = split /\s+/, $al;
    $sentenceID++;

    my %al = ();
    foreach my $a (@al) {
        my ($f,$e) = split /-/, $a;
        $al{$f}{$e} = 1;
        if ($e >= @en) { die "malformed alignments"; }
    }

    # find any FR word that's aligned to multiple EN words that are
    # not consecutive
    foreach my $f (keys %al) {
        my @allE = sort { $a <=> $b } keys %{$al{$f}};
        my $consecutive = 1;
        for (my $i=1; $i<@allE; $i++) {
            if ($allE[$i]-1 != $allE[$i-1]) { 
                $consecutive = 0;
                last;
            }
        }
        $countConsecutive += $consecutive;
        $countNonConsecutive += (1 - $consecutive);
        if (not $consecutive) {
            #die "on sentence $sentenceID, f=$f, allE=" . (join ' ', @allE) . "\n";

            # find least frequent word position in allE
            my $lfwId = 0;
            for (my $i=1; $i<@allE; $i++) {
                if ($enFreq{$en[$allE[$lfwId]]} < $enFreq{$en[$allE[$i]]}) {
                    $lfwId = $i;
                }
            }

            # find start and end (INCLUSIVE) ids that are contiguous
            my $stID = $lfwId;
            my $enID = $lfwId;
            while ($stID >= 0) {
                if ($allE[$stID-1] == $allE[$stID]-1) {
                    $stID--;
                } else {
                    last;
                }
            }
            while ($enID < @allE-1) {
                if ($allE[$enID+1] == $allE[$enID]+1) {
                    $enID++;
                } else {
                    last;
                }
            }
            
            # now we're only aligned to things in this range
            %{$al{$f}} = ();
            for (my $id=$stID; $id<=$enID; $id++) {
                $al{$f}{$allE[$id]} = 1;
            }
        }
    }

    # now attach unaligned source words to their most frequent (aligned) neighbors
    while (1) {
        my %hit = ();
        foreach my $f (keys %al) {
            foreach my $e (keys %{$al{$f}}) {
                $hit{$e} = 1;
            }
        }
        if (scalar keys %hit == 0) {
            die "completely unaligned sentence???";
        }
        if (scalar keys %hit == scalar @en) {
            last; # everything is aligned
        }
        my $numFixed = 0;
        for (my $e=0; $e<@en; $e++) {
            if (defined $hit{$e}) { next; } # already aligned
            my $freqLeft = -1;
            my $freqRight = -1;
            if (($e > 0) && (defined $hit{$e-1})) {
                $freqLeft = $neighbors{$en[$e]}{$en[$e-1]} || 0;
            }
            if (($e < @en-1) && (defined $hit{$e+1})) {
                $freqRight = $neighbors{$en[$e]}{$en[$e+1]} || 0;
            }
            if (($freqLeft < 0) && ($freqRight < 0)) {
                next; # neither of my neighbors is aligned!
            }
            my $friend;
            if ($freqLeft < 0) { $friend = $e+1; }
            elsif ($freqRight < 0) { $friend = $e-1; }
            elsif ($freqLeft <= $freqRight) { $friend = $e-1; } # slight bias here to lump on to the word to our left
            else { $friend = $e+1; }

            my $friendF = -1;
            foreach my $f (keys %al) {
                if (defined $al{$f}{$friend}) { $friendF = $f; last; }
            }
            if ($friendF < 0) { 
                print STDERR "freqLeft=$freqLeft freqRight=$freqRight e=$e friend=$friend\n";
                die "could not find friend!"; 
            }

            $al{$friendF}{$e} = 1;
            $hit{$e} = 1;
            $numFixed ++;
            $totalFixed++;
        }
        if ($numFixed == 0) { die "could not repair this sentence"; }
    }
    
    # write the output
    my $first = 1;
    foreach my $f (sort { $a <=> $b } keys %al) {
        foreach my $e (sort { $a <=> $b } keys %{$al{$f}}) {
            if (not $first) { print O ' '; }
            print O "$f-$e";
            $first = 0;
        }
    }
    print O "\n";
}
while (<AL>) { die "$FNAME.al has more lines than $FNAME.en"; }
close EN or die;
close AL or die;
close O  or die;

print STDERR "     consec alns: $countConsecutive\n";
print STDERR "  nonconsec alns: $countNonConsecutive\n";
print STDERR " reattached toks: $totalFixed\n";
