#!/usr/bin/perl -w
use strict;

my $THRESHOLD = 0.95;
my $MAXCOUNT  = 10;

my %fr = ();
my %mtu = ();
my $N = 0;
while (<>) {
    chomp;
    my ($id,$freqs,$mtu) = split /\t/, $_;
    my ($df,$tf) = split /\s+/, $freqs;
    my ($fr,$en) = split / \| /, $mtu;
    $mtu{$fr}{$en} = $tf;
    my @fr = split /\s+/, $fr;
    $fr{$fr[0]}{$fr} = 1;
    $N += $tf;
}

=pod
foreach my $fr0 (keys %mtu) {
    my $fr = $fr0; $fr =~ s/ _//g;
    my @fr = split /\s+/, $fr;
    if (@fr < 2) { next; }

    foreach my $frL0 (keys %{$fr{$fr[0]}}) { # these are phrases that match the first word
        my $frL = $frL0; $frL =~ s/ _//g;
        my @frL = split /\s+/, $frL;
        if (@frL >= @fr) { next; }

        my $matchesFR = 1;
        for (my $i=1; $i<@frL; $i++) {
            if ($frL[$i] != $fr[$i]) { $matchesFR = 0; last; }
        }
        if (not $matchesFR) { next; }

        my $j = @frL;
        foreach my $frR0 (keys %{$fr{$fr[$j]}}) {
            my $frR = $frR0; $frR =~ s/ _//g;
            my @frR = split /\s+/, $frR;
            if ($j + @frR != @fr) { next; }
            $matchesFR = 1;
            for (my $i=$j; $i<@fr; $i++) {
                if ($fr[$i] != $frR[$i-$j]) { $matchesFR = 0; last; }
            }
            if (not $matchesFR) { next; }

            foreach my $en (keys %{$mtu{$fr0}}) {
                foreach my $enL (keys %{$mtu{$frL0}}) {
                    foreach my $enR (keys %{$mtu{$frR0}}) {
                        if (($en eq ($enL . ' ' . $enR)) || ($en eq ($enR . ' ' . $enL))) {
                            my $tf  = $mtu{$fr0}{$en} / $N;
                            my $tfL = $mtu{$frL0}{$enL} / $N;
                            my $tfR = $mtu{$frR0}{$enR} / $N;
                            my $prod = $tfL * $tfR;
                            my $lt = ($tf < $prod) ? '<' : (($tf == $prod) ? '=' : '>');
                            print "Success! $tf $lt $prod ($tfL $tfR) '$fr0'='$frL0 + $frR0' and '$en'='$enL + $enR'\n";
                        }
                    }
                }
            }
        }

    }
}
exit(-1);
=cut

my $id = 0;
foreach my $fr (keys %mtu) {
    my $total = 0;
    foreach my $v (values %{$mtu{$fr}}) { $total += $v; }
    my $running = 0;
    my $count = 0;
    foreach my $en (sort { $mtu{$fr}{$b} <=> $mtu{$fr}{$a} } keys %{$mtu{$fr}}) {
        $count++;
        if ($count > $MAXCOUNT) { last; }
        print $id . "\t1 " . $mtu{$fr}{$en} . "\t" . $fr . " | " . $en . "\n";
        $id++;
        $running += $mtu{$fr}{$en};
        if ($running >= $total * $THRESHOLD) {
            last;
        }
    }
}
