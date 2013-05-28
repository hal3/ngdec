#!/usr/bin/perl -w
use strict;

my $FRfile = shift or die;
my $ENfile = shift or die;
my $ALfile = shift or die;

open F, $FRfile or die;
open E, $ENfile or die;
open A, $ALfile or die;
while (1) {
    $_ = <F>;
    if (not defined $_) { last; }
    printIt($_);
    print "\t";

    $_ = <E>;
    if (not defined $_) { die; }
    printIt($_);
    print "\t";

    $_ = <A>;
    if (not defined $_) { die; }
    s/-/ /g;
    printIt($_);
    print "\n";
}
while (<E>) { die; }
while (<A>) { die; }

sub printIt {
    my ($t) = @_;
    chomp $t; 
    $t =~ s/\s+/ /g;
    $t =~ s/^ //; 
    $t =~ s/ $//;
    my @t = split /\s+/, $t;
    print '' . (scalar @t) . ' ' . $t;
}
