#!/usr/bin/perl -w
use strict;

my $THRESHOLD = 0.95;
my $MAXCOUNT  = 10;

my %mtu = ();
while (<>) {
    chomp;
    my ($id,$freqs,$mtu) = split /\t/, $_;
    my ($df,$tf) = split /\s+/, $freqs;
    my ($fr,$en) = split / \| /, $mtu;
    $mtu{$fr}{$en} = $tf;
}

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
