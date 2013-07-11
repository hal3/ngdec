use strict;
use Text::Unaccent;  # can be commented out if you'll never use --deaccent

require 5.005;

my $MAX_VOCAB_SIZE = 10000000;  # warning: MUST MATCH ngdec.h !!!


sub read_lm_vocab {
    my ($lmName) = @_;

    open F, mkfilename($lmName) or die;
    my %v = ();
    my $V = 0;
    my $inUnigrams = 0;
    while (<F>) {
        chomp;
        if (/^\\1-grams:/) { 
            $inUnigrams = 1;
        } elsif ($inUnigrams) {
            if (/^\s*$/) { last; }
            if (/^\\2-grams:/) { last; }
            my ($prob, $word, $backoff) = split;
            if (not defined $word) { die "malformed LM: $lmName"; }
            $v{$word} = $V;
            if (($word eq '<unk>') && ($V != 0)) { die "lm $lmName doesn't have <unk>=0"; }
            if (($word eq '<s>'  ) && ($V != 1)) { die "lm $lmName doesn't have <s>=1"; }
            if (($word eq '</s>' ) && ($V != 2)) { die "lm $lmName doesn't have </s>=2"; }
            $V++;
        }
    }
    close F;

    return (%v);
}


sub lookupVocab {
    my ($TryDeaccent, $srcV, $tgtV, $localV, $isSRC, $vocabFile, $w) = @_;

    if ($isSRC) {
        if (defined $srcV->{$w}) { return $srcV->{$w}; }
    } else {
        if (defined $tgtV->{$w}) { return $tgtV->{$w}; }
    }
    if (!defined $localV->{$w}) { 
        if ($isSRC && $TryDeaccent) {
            my $w2 = deaccent($TryDeaccent, $w);
            if (defined $localV->{$w2}) {
                return $MAX_VOCAB_SIZE + $localV->{$w2};
            }
        }
        $localV->{$w} = scalar keys %$localV;
        if (defined $vocabFile) {
            print $vocabFile ($MAX_VOCAB_SIZE + $localV->{$w}) . "\t" . $w . "\n";
        }
    }
    return $MAX_VOCAB_SIZE + $localV->{$w};
}

sub stringToIDs {
    my ($TryDeaccent, $srcV, $tgtV, $localV, $isSRC, $vocabFile, $txt) = @_;
    $txt =~ s/\s+/ /g; 
    $txt =~ s/^ //;
    $txt =~ s/ $//;
    my @t = split / /, $txt;
    my $ret = scalar @t;
    foreach my $t (@t) { 
        $ret .= ' ' . lookupVocab($TryDeaccent, $srcV, $tgtV, $localV, $isSRC, $vocabFile, $t);
    }
    return $ret;
}


sub mkfilename {
    my ($fn) = @_;
    if ($fn =~ /\.gz$/) { return "zcat $fn |"; }
    else { return $fn; }
}

sub readINI {
    my ($INI) = @_;
    my %opts = ();
    open F, $INI or die "cannot read $INI";
    while (<F>) {
        chomp;
        if (/^([^=]+)=(.+)$/) { $opts{$1} = $2; }
        elsif (/^([^\s]+)\s+([^\s]+)$/) { $opts{$1} = $2; }
        elsif (/^\s*$/) { next; }
        elsif (/^\#/) { next; }
        else { die "malformed ini line: $_"; }
    }
    close F or die;
    return (%opts);
}

sub requireOpt {
    my ($opts, $option) = @_;
    if (not defined $opts->{$option}) { die "ini file didn't specify required option $option"; }
}

sub deaccent {
    my ($Encoding, $txt) = @_;
    # simple
    $txt = unac_string($Encoding, $txt);
    # remove combining marks
    $txt =~ s/\pM//og;
    # in case of european style numbers:
    if ($txt =~ /^([0-9]+),([0-9]+)$/) {
        if (length($2) != 3) {
            $txt = $1 . '.' . $2;
        }
    }
    # done
    return $txt;
}

sub read_local_vocab {
    my ($fn) = @_;
    my %v = ();
    open F, mkfilename($fn) or die "cannot open $fn";
    while (<F>) {
        chomp;
        if (/^([0-9]+)\t(.+)$/) {
            $v{$2} = $1;
        } else {
            die "local vocab file $fn malformed: '$_'";
        }
    }
    close F;
    return %v;
}
