use strict;
use warnings 'all';
use List::Util qw( min max sum );
use List::MoreUtils 'all';
use Getopt::Long;
use Inline CPP => 'DATA';
use Inline CPP => config => ccflags => '-std=c++11';

our $processor = \&countUnigramFreqInWindow;
our $interestf1name;
our $corpus_dir;

our %options = ( mode => '2', nofilter => 0, c => 0, input => "", dir => "",
                 f1 => "top1grams.txt", f2 => "top2grams.txt",
				 minwords_sentence => 5, window => 3, dyn => 0, top1 => 15000, # '0' means to count all words
				 thres1 => '100,5e-8', thres2 => '0,0', # '5,0.0005'
			   );

$| = 1;

sub usage
{
	print <<USAGE;
Usage:
    $0 -d(--dir) corpus_directory [options]
    $0 -i(--input) corpus_file [options]

Options optional:
-c:                 Use C++ extensions to gain speedup
--nofilter:         Disable text filtering/formatting, to speed up processing
-m,--mode int:      '1' (unigram) or '2'(bigram) (default: $options{mode})
--f1 file:          List of words to write or read.
                    If the mode is '1', this is the filename to save.
                    If the mode is '2', this file is required to be read
                    as the counted word list
                    (default: "$options{f1}")
--f2 file:          File name of the bigrams to save (default: "$options{f2}")
-s,--suffix file:   Corpus text file suffix
--filecount int:    Number of files to process. Other files in the corpus dir
                    are ignored.
-n,--window int:    Size of the neighborhood for bigrams (default: $options{window})
--dyn:              Dynamic window size ( uniformly random in [ 1,--window ] )
--thres1 int,float: Threshods of unigrams to be considered.
                    Format: freq,prob. Max of the two is taken.
                    (default: $options{thres1})
--thres2 int,float: Threshods of bigrams to be considered.
                    Format: freq,prob. Max of the two is taken.
                    (default: $options{thres2})
--top1 int:         Top n words to be counted in the unigram word list file.
                    Only applicable in the bigram mode. (default: $options{top1})
-e,--extra file:    List of unigrams not in --top1 to count in the bigram mode
                    These words have to be present in --f1 file.
USAGE

	exit;
}


Getopt::Long::Configure("bundling", "ignore_case");

GetOptions( \%options, 'dir|d=s', 'input|i=s', 'nofilter', 'c', 'mode|m=i', 'f1=s', 'f2=s',
		    'suffix|s=s', 'filecount=i', 'window|w=i', 'dyn', 'thres1=s', 'thres2=s',
		    'top1=i', 'extra|e=s' ) || usage();

if( ! $options{'dir'} && ! $options{'input'}
                            ||
      $options{'dir'} && $options{'input'} ){
	usage();
}

our %unigram2freq;
our %bigram2freq;
our %interestWord2ID;
our $interestWordsOn = 0;
our $allBigramOccurrences = 0;

if( $options{mode} == 1 ){
	$processor = \&countUnigramFreqInWindow;
}
elsif( $options{mode} == 2 ){
	$processor = \&countBigramFreqInWindow;
	if( ! $options{'f1'} ){
		die "In bigram mode, the '--f1 word_list_file_name' must be specified\n";
	}

	loadlist( $options{'f1'}, \%unigram2freq, $options{'top1'}, $options{'extra'} );
}
else{
	die "Mode '$options{mode}' is unknown. Can only be '1' (for unigram) or '2' (for bigram)\n";
}

our ($min1gramfreq, $min1gramprob);
our ($min2gramfreq, $min2gramprob);

if( $options{'thres1'} !~ /^([\d.]+|\d+e-?\d+),([\d.]+|\d+e-?\d+)$/i ){
    die "'$options{'thres1'}': wrong number format of --thres1\n";
}
else{
    ($min1gramfreq, $min1gramprob) = split /,/, $options{'thres1'};
}

if( $options{'thres2'} !~ /^([\d.]+|\d+e-?\d+),([\d.]+|\d+e-?\d+)$/i ){
    die "'$options{'thres2'}': wrong number format of --thres2\n";
}
else{
    ($min2gramfreq, $min2gramprob) = split /,/, $options{'thres2'};
}

# use words in the file $options{'f1'} as the words of interest (words counted)
# In bigram mode, interest words are always loaded and $interestWordsOn is always 1
%interestWord2ID = %unigram2freq;
if( keys %interestWord2ID > 0 ){
	$interestWordsOn = 1;
	print scalar keys %interestWord2ID, " interest words\n";
}

if( ! $options{nofilter} && $options{c} ){
    print "WARN: -c is only valid when --nofilter is specified. It will be disabled now.\n";
    delete $options{c};
}

if( $options{c} ){
    #void passParams_( int mode, char* input, char* dir, char* f1, int top1, 
    #                  int minwords_sentence, int window, char* thres1, char* thres2 )
    passParams_( $options{mode}, $options{input}, $options{dir}, $options{f1}, $options{top1}, 
                 $options{minwords_sentence}, $options{window}, $options{dyn}, $options{thres1}, $options{thres2} );
                 
    if($interestWordsOn){
        passInterestWords_( [ keys %interestWord2ID ] );
    }
}

our $totalWordCount = 0;
our $totalInterestWordCount = 0;
our $totalSentenceCount = 0;
our $totalNonEngWordCount = 0;
our $lineCount = 0;
our $fileCount = 0;
# '0' is no limit
our $fileCountLimit = $options{filecount} || 0;

sub processFile
{
    my ($filename, $fileFullname) = @_;
    my $line;

 	open(FH, "< $fileFullname" ) || die "Cannot open '$fileFullname' to read: $!\n";
	print "$filename:\n";
	my $sentenceCount = 0;
    my $filesize = 0;
    
	while( $line = <FH> ){
	    $filesize += length($line);
		chomp $line;
		next if !$line;
		$lineCount++;
		$line = lc($line);
		my @sentences;
		my $sentence;
		if( ! $options{nofilter} ){
    		$line =~ s/&[a-z];//g;
    		@sentences = split /[,;?:!\"()]|\. |\.$|--/, $line;
    	}
    	else{
    	    @sentences = ( $line );
    	}
		for(my $i = 0; $i < @sentences; $i++){
		    $sentence = $sentences[$i];
		    $sentenceCount++;
		    $totalSentenceCount++;
			my @words;

    		if( $options{c} ){
			    # if no filtering, each line is a sentence. So $. == $sentenceCount.
			    # no need to display $.
                if( $sentenceCount % 500 == 0 ){
                    printf( "\r%d, %.1fM\r", $sentenceCount, $filesize / (1024*1024) );
                }
                processSentence_($sentence);
                next;
    		}
    		
    		# old Perl code
			if( ! $options{nofilter} ){
                if( $sentenceCount % 500 == 0 ){
                    printf( "\r%d, %d, %.1fM\r", %., $sentenceCount, $filesize / (1024*1024) );
                }
                #$sentence =~ s/\'//g;

    			@words = split /\s+|\./, $sentence;
    			# remove empty entries caused by leading/trailing spaces
    			# not necessary for wiki corpus, as they are all removed
    			@words = grep { ! /^(|\$|-|¡ª|¡ê|\/|\%|\@|\#)$/  } @words;
    		}
    		else{
			    # if no filtering, each line is a sentence. So $. == $sentenceCount.
			    # no need to display $.
                if( $sentenceCount % 500 == 0 ){
                    printf( "\r%d, %.1fM\r", $sentenceCount, $filesize / (1024*1024) );
                }
    		    @words = split / /, $sentence;
    		}

    		my $wc = scalar @words;
    		# remove non-English, e.g. French, Greek..., letters
  			@words = grep { /^[a-z0-9\']+$/ } @words;

			my $nonEngWc = $wc - scalar @words;
			next if @words < $options{minwords_sentence};

			$totalWordCount += @words;
			$totalNonEngWordCount += $nonEngWc;

			&$processor(\@words);
		}
	}
	printf( "%d, %.1fM\n", $sentenceCount, $filesize / (1024*1024) );

    $fileCount++;
}

if( $options{'dir'} ){
    opendir( DH, $options{'dir'} ) || die "Failed to open $options{'dir'} as a directory: $!\n";

    my $filename;
    my $fileFullname;

    while( $filename = readdir(DH) ){
    	$fileFullname = "$options{'dir'}/$filename";
    	if( ! -f $fileFullname ){
    		next;
    	}
    	if( $options{suffix} && $filename !~ /\.$options{suffix}$/ ){
    	    next;
    	}

        processFile( $filename, $fileFullname );

    	#print ".: ", $unigram2freq{'.'} || 0, "\n";
    	if( $fileCountLimit && $fileCount >= $fileCountLimit ){
    		last;
    	}
    }
}
else{
    processFile( $options{'input'}, $options{'input'} );
}

print "$fileCount files, $lineCount lines, $totalSentenceCount sentences, ";
if( ! $options{c} ){
    print "$totalWordCount words occur, $totalNonEngWordCount non Eng, $totalInterestWordCount interest words occur\n";
}
else{
    printStats_();
}

if( $options{mode} == 1 ){
	my $top1gramFilename = getAvailName( $options{'f1'} );
	
	if( $options{c} ){
	    outputTopUnigrams_($top1gramFilename);
	}
	else{
	    outputTopUnigrams($top1gramFilename);
	}
}
elsif( $options{mode} == 2 ){
	my $topbigramFilename = getAvailName( $options{'f2'} );
	
	if( $options{c} ){
	    outputTopBigrams_($topbigramFilename);
	}
	else{
	    outputTopBigrams($topbigramFilename);
	}
}

sub loadlist
{
	my ( $listFilename, $href, $topcount, $extraUnigramFile ) = @_;
	my $FH;
	open($FH, "< $listFilename") || die "Cannot open '$listFilename' to read: $!\n";
	my @lines = <$FH>;

	%$href = map { ( split /\t/ )[0..1] } grep { !/^\#/ } @lines;
	my @words = sort { $href->{$b} <=> $href->{$a} } keys %$href;
	my $wordcount = scalar @words;
	print "$wordcount words loaded from '$listFilename'\n";
	close($FH);

	if( $topcount && $topcount < $wordcount ){
	    print "Top $topcount cuts between '$words[$topcount - 1]' and '$words[$topcount]'\n";
	    @words = @words[ 0 .. $topcount - 1 ];
	}

    my @extraWords = ();
    if( $extraUnigramFile ){
    	open($FH, "< $extraUnigramFile") || die "Cannot open '$extraUnigramFile' to read: $!\n";
    	my @lines = <$FH>;
    	@extraWords = map { chomp; (split /\t/)[0] } @lines;
    	print scalar @extraWords, " extra unigrams loaded from '$extraUnigramFile'\n";
    	close($FH);
    }

    %$href = map { $_ => 0 } ( @words, @extraWords );
    $wordcount = scalar keys %$href;
    print "$wordcount valid unigrams\n";
	return $wordcount;
}

sub countUnigramFreqInWindow
{
	for( @{$_[0]} ){
		if( ! $interestWordsOn || exists $interestWord2ID{$_} ){
			$unigram2freq{$_}++;
			$totalInterestWordCount++;
		}
		# Otherwise, $_ is not in the interest words list. Ignore
	}
}

sub countBigramFreqInWindow
{
	my ($i, $j, $leftborder, $rightborder);
	my $words = $_[0];

	for( $i = 0; $i < @$words; $i++ ){
		my $w = $words->[$i];
		next if !exists $interestWord2ID{$w};
		$totalInterestWordCount++;
		
		my $windowSize;
		if( $options{dyn} ){
		    $windowSize = int( rand($options{window}) ) + 1;
		}
		else{
		    $windowSize = $options{window};
		}
		
		$leftborder = max( $i - $windowSize, 0 );
		#$rightborder = min( $i + $options{window}, scalar @$words - 1 );
		$rightborder = $i - 1;

		for( $j = $leftborder; $j <= $rightborder; $j++ ){
			next if $j == $i;
			my $w2 = $words->[$j];
			next if !exists $interestWord2ID{$w2};
			$bigram2freq{$w}{$w2}++;
    		$unigram2freq{$w}++;
    		$allBigramOccurrences++;
		}
	}
}

sub outputTopUnigrams
{
	my $top1gramFilename = shift;
	my $OUTF;

	my $min1gramfreq2 = int( $min1gramprob * $totalInterestWordCount );
	$min1gramfreq = max( $min1gramfreq, $min1gramfreq2 );
	print "Words cut-off frequency: $min1gramfreq\n";

	open($OUTF, "> $top1gramFilename") || die "Cannot open '$top1gramFilename' to write: $!\n";

	my @words = sort { $unigram2freq{$b} <=> $unigram2freq{$a} } grep { $unigram2freq{$_} >= $min1gramfreq } keys %unigram2freq;
	writeParams($OUTF, scalar @words);

	print "Saving ", scalar @words, " words into '$top1gramFilename'...\n";

	for my $w(@words){
		print $OUTF join( "\t", $w, $unigram2freq{$w},
		                   trunc( 3, log( $unigram2freq{$w} / $totalInterestWordCount ) ) ), "\n";
	}
	close($OUTF);
	print "Done.\n";
}

sub outputTopBigrams
{
	my $topbigramFilename = shift;
	my $OUTF;
	open($OUTF, "> $topbigramFilename") || die "Cannot open '$topbigramFilename' to write: $!\n";

	# sort central words according to their total frequencies
	my @words = sort { $unigram2freq{$b} <=> $unigram2freq{$a} } keys %bigram2freq;

	my $w;
	my @words2;
	my $wordCount = 0;

	for $w(@words){
		my $min2gramfreq2 = int( $min2gramprob * $unigram2freq{$w} );
		$min2gramfreq = max( $min2gramfreq, $min2gramfreq2 );

		my @neighbors = grep { $bigram2freq{$w}{$_} >= $min2gramfreq }
						keys %{ $bigram2freq{$w} };

		# some words have all neighbors filtered. ignore them
		if( @neighbors > 0 ){
		    push @words2, $w;
		    $wordCount++;
		}
	}

	writeParams( $OUTF, scalar @words2 );
    print $OUTF "Words:\n";

    my $i;
	for( $i = 0; $i < @words2; $i++ ){
	    $w = $words2[$i];
	    # the unigram probability is defined as the fraction of bigrams
	    # whose second word is $w, in all bigrams
	    print $OUTF join( ",", $w, $unigram2freq{$w},
	                        trunc( 3, log( $unigram2freq{$w} / $allBigramOccurrences ) ) );

	    if( $i < @words2 - 1 ){
	        if( $i % 10 == 9 ){
	            print $OUTF "\n";
	        }
	        else{
	            print $OUTF "\t";
            }
	    }
	}

    print $OUTF "\n\nBigrams:\n";

	my $bigramCount = 0;
	my $allKeptBigramOccurrences = 0;

	print "Saving bigrams from ", scalar @words2, " words into '$topbigramFilename'...\n";

	$wordCount = 0;

	for $w(@words2){
		my $min2gramfreq2 = int( $min2gramprob * $unigram2freq{$w} );
		$min2gramfreq = max( $min2gramfreq, $min2gramfreq2 );

		my @neighbors = sort { $bigram2freq{$w}{$b} <=> $bigram2freq{$w}{$a} }
						grep { $bigram2freq{$w}{$_} >= $min2gramfreq }
						keys %{ $bigram2freq{$w} };

		# neighbor words are words following $w, not preceding $w
		my $neighborTotalOccur = sum( @{$bigram2freq{$w}}{@neighbors} );
		$wordCount++;
		$allKeptBigramOccurrences += $neighborTotalOccur;

		my $neighbor;
		print $OUTF join( ",", $wordCount, $w, scalar @neighbors, $neighborTotalOccur, $min2gramfreq );
		print $OUTF "\n";

		for(my $i = 0; $i < @neighbors; $i++){
			$neighbor = $neighbors[$i];

			print $OUTF "\t", join( ",", $neighbor, $bigram2freq{$w}{$neighbor},
						# log-probability, truncated to 3 decimal places
						trunc( 3, log( $bigram2freq{$w}{$neighbor} / $neighborTotalOccur ) ) );

			$bigramCount++;
			# 5 bigrams each line for humans to read easily
			if( $i % 5 == 4 && $i < @neighbors - 1 ){
				print $OUTF "\n";
			}
		}
		print $OUTF "\n";

		if( $wordCount % 100 == 99 && $wordCount < @words2 - 1 ){
		    print "\r$wordCount\r";
		}

		# release memory, to avoid using up the memory during the writing (which uses working sets)
		delete $bigram2freq{$w};
	}
	print $OUTF "# Total kept bigram occurrences: $allKeptBigramOccurrences\n";
	close($OUTF);

    print "\n";
	print "$bigramCount bigrams from ", $wordCount, " words are saved into '$topbigramFilename'\n";

	if( $allKeptBigramOccurrences > $allBigramOccurrences ){
	    die "BUG: kept bigram occurs $allKeptBigramOccurrences > all bigram occurs $allBigramOccurrences\n";
	}
}

sub writeParams
{
	my ($OUTF, $wordcount) = @_;

	my @varNames = qw( f1 minwords_sentence  window  thres1  thres2 );

	if( exists $options{'input'} ){
	    unshift @varNames, "input";
	}
	else{
	    unshift @varNames, "dir";
	}
	if( $options{mode} == 2 ){
	    push @varNames, "top1";
	}

	print $OUTF "# $wordcount words, $totalInterestWordCount occurrences\n";
	print $OUTF '# ', join( ', ', map { "$_=" . $options{$_} } @varNames ), "\n";
	if( $options{mode} == 2 ){
	    print $OUTF "# $allBigramOccurrences bigram occurrences\n";
	}
}

# provide $origname as the template, and return a new name if $origname is used by a file
sub getAvailName
{
	my $origname = shift;
	my $append;
	my ($name, $suffix);

	if(! -e $origname){
		return $origname;
	}

	if($origname =~ /^(.+)(\.[^.]+)$/){
		$name = $1;
		$suffix = $2;
	}
	else{
		$name = $origname;
		$suffix = "";
	}

	$append = -1;
	while(-e "$name$append$suffix"){
		$append--;
	}
	return "$name$append$suffix";
}

# if $wantarray, returns @$array
# else $array->[0]
sub arrayOrFirst
{
	my ($wantarray, $array) = @_;
	if($wantarray){
		return @$array;
	}
	return $array->[0];
}
# truncate a floating point number to the $prec digits after the decimal point
sub trunc
{
	my $prec = shift;
	my @results;

	for(@_){
		push @results, 0 + sprintf("%.${prec}f", $_);
	}
	return arrayOrFirst(wantarray, \@results);
}

__DATA__
__CPP__

#undef open
#undef seekdir
#undef seed

#include <hash_map>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>
#include <sstream>
#include <numeric>
#include <random>
#include "/corpus/perlxs.h"

using namespace __gnu_cxx;
using namespace std;

namespace __gnu_cxx {
    template <> struct hash<std::string> {
        size_t operator() (const std::string& x) const 
        {
            return hash<const char*>()(x.c_str());
        }
    };
}

/*

This operator when used with copy(), crashes mysteriously. Inspect later
//copy( h.begin(), h.end(), pairs.begin() );

namespace std {
    template<> 
    template<>
    pair<string, int>& pair<string, int>::operator=(const pair<const string, int>& a)
    {
        printf("%s\n", a.first.c_str());
        this->first = a.first.c_str();
        this->second = a.second;
        return *this;
    }
}
*/

typedef hash_map<string, int> S2I;
// maps a word to its ID
S2I interestWord2ID;
vector<string> ID2word;

vector<int> unigram2freq;

typedef hash_map<int, int> I2I;
vector<I2I> bigram2freq;

bool interestWordsOn = false;
double totalInterestWordCount = 0;
double allBigramOccurrences = 0;
int totalWordCount = 0;
int totalNonEngWordCount = 0;

struct options{
    int mode;
    string input;
    string dir;
    string f1;
    int top1;
    int minwords_sentence;
    int window;
    int dyn;    // actually bool
    string thres1;
    string thres2;
    int min1gramfreq;
    double min1gramprob;
    int min2gramfreq;
    double min2gramprob;
} opt;

default_random_engine randGen;
uniform_int_distribution<int>* pUniform;

// remove non-English, e.g. French, Greek..., letters
// include processive forms like "people's" as unigrams
bool isWordChar[256];

// Checking & operating as "existKey(); h[key]++" takes twice time.
// Use "ppair = h.find(key); ppair->second++" instead to reduce unnecessary time
template<class HashType, typename KeyType>
bool existKey(HashType& h, KeyType key)
{
    if( h.find(key) != h.end() )
	    return true;
    return false;
}

/*
This cmp function is extremely slow when used in sort(). do not know why. Inspect later
template< typename K, typename V, bool reversed=true, typename P=pair<K, V> >
bool cmpvalue(const P& p1, const P& p2)
{ 
    return ( p1.second < p2.second ) ^ reversed; 
}
*/

template< typename K, typename V, typename P=pair<K, V> >
void sortHash2Pairs( hash_map<K, V>& h, vector<P>& pairs )
{
    pairs.clear();
    pairs.reserve(h.size());

    for( auto iter = h.begin(); iter != h.end(); ++iter ){
        pairs.push_back( P( iter->first, iter->second ) );
    }
    
    //printf("Sorting %d pairs... ", pairs.size());    
       
    sort( pairs.begin(), pairs.end(), [](const P& p1, const P& p2) { return ( p1.second > p2.second ); } );
    
    //printf("Done.\n");
}

// only sort the beginning n elements in vec
template< typename V, typename P=pair<int, V> >
void sortVec2Pairs( vector<V>& vec, int N, vector<P>& pairs )
{
    pairs.clear();
    pairs.reserve(N);

    for( int i = 0; i < N; i++ ){
        pairs.push_back( P( i, vec[i] ) );
    }
    
    //printf("Sorting %d pairs... ", pairs.size());    
       
    sort( pairs.begin(), pairs.end(), [](const P& p1, const P& p2) { return ( p1.second > p2.second ); } );
    
    //printf("Done.\n");
}

#define ASSIGN(field) opt.field = field;
void passParams_( int mode, char* input, char* dir, char* f1, int top1, 
                  int minwords_sentence, int window, int dyn, char* thres1, char* thres2 )
{
    ASSIGN(mode)
    ASSIGN(input)
    ASSIGN(dir)
    ASSIGN(f1)
    ASSIGN(top1)
    ASSIGN(minwords_sentence)
    ASSIGN(window)
    ASSIGN(dyn)
    ASSIGN(thres1)
    ASSIGN(thres2)
    
    stringstream ss;
    ss << thres1;
    ss >> opt.min1gramfreq >> opt.min1gramprob;

    ss << thres2;
    ss >> opt.min2gramfreq >> opt.min2gramprob;
    
    fill( isWordChar, isWordChar + 256, false );
    // already in lower case
    // fill( isWordChar + 'A', isWordChar + 'Z' + 1, true );
    fill( isWordChar + 'a', isWordChar + 'z' + 1, true );
    isWordChar['\''] = true;
    
    // top 100,000 words, big enough for usual need
    bigram2freq.reserve(100000);
    
    if( opt.dyn ){
        pUniform = new uniform_int_distribution<int>( 1, opt.window );
    }
}

void passInterestWords_( char** pInterestWords )
{
    interestWordsOn = true;
    char** pw = pInterestWords;
    char* w;
    int id = 0;
    while( w = *pw ){
        interestWord2ID[w] = id;
        ID2word.push_back(w);
        pw++;
        id++;
    }
    
    unigram2freq.resize( interestWord2ID.size() );
    
    if( opt.mode == 2 )
        bigram2freq.resize( interestWord2ID.size() );
        
    printf( "%d interest words are passed to C++ module\n", interestWord2ID.size() );
}

int lookupWordID(string& word)
{
	auto ppair = interestWord2ID.find(word);
	
	if( ppair == interestWord2ID.end() ){
	    if(interestWordsOn)
	        return -1;
	    else{
	        // if interestWordsOn=false, then store all words in interestWord2ID
	        int wid = interestWord2ID.size();
	        
	        // always make unigram2freq & ID2word big enough
	        if( unigram2freq.size() <= wid )
	            unigram2freq.resize( wid * 2 + 2 );
	            
	        if( ID2word.size() <= wid )
	            ID2word.resize( wid * 2 + 2 );
	        
	        if( opt.mode == 2 && bigram2freq.size() <= wid )
	            bigram2freq.resize( wid * 2 + 2 );
	        
	        ID2word[wid] = word.c_str();   
	        interestWord2ID[word] = wid;
	        return wid;
        }
    }
    else{
        return ppair->second; 
    }
}

void countUnigramFreqInWindow_(vector<string>& words)
{
	for(int i = 0; i < words.size(); i++){
	    string& word = words[i];
	    int wid = lookupWordID(word);
	    if( wid >= 0 ){
	        unigram2freq[wid]++;
    	    totalInterestWordCount++;
        }
        
		// Otherwise, word is not in the interest words list. Ignore
	}
}

void countBigramFreqInWindow_(vector<string>& words, int window)
{
	int i, j, leftborder, rightborder;

	for( i = 0; i < words.size(); i++ ){
		string& w = words[i];
		
		int wid = lookupWordID(w);
        if( wid < 0 )
            continue;
            
		totalInterestWordCount++;
		
		int windowSize;
		if( opt.dyn )
		    windowSize = (*pUniform)(randGen);
		else
		    windowSize = opt.window;
		    
		leftborder = max( i - windowSize, 0 );
		rightborder = i - 1;

		for( j = leftborder; j <= rightborder; j++ ){
			if(j == i)
			    continue;
			string& w2 = words[j];
			int w2id = lookupWordID(w2);
			if( w2id < 0 )
			    continue;
    		bigram2freq[wid][w2id]++;
			unigram2freq[wid]++;
    		allBigramOccurrences++;
		}
	}
}

void processSentence_(char* sentence)
{
    vector<string> words;
    char* w;
    int wc = 0;
    
    w = strtok(sentence, " ");
    while(w){
        unsigned char* pchar = (unsigned char*)w;
        bool isLegalWord = true;
        while(*pchar){
    		// remove words containing non-English, e.g. French, Greek..., letters
    		// include processive forms like "people's" as unigrams
            if( ! isWordChar[*pchar] ){
                isLegalWord = false;
                break;
            }
            pchar++;
        }
        if(isLegalWord){
            words.push_back(w);
        }
            
        wc++;
        w = strtok(NULL, " ");
    }

    int nonEngWc = wc - words.size();
    
    if( words.size() < opt.minwords_sentence )
        return;

    if( opt.mode == 1 )
        countUnigramFreqInWindow_(words);
    else
        countBigramFreqInWindow_( words, opt.window );
        
    totalWordCount += words.size();
    totalNonEngWordCount += nonEngWc;
}

void printStats_()
{
    printf( "%d words occur, %d non Eng, %d interest words occur\n",
                totalWordCount, totalNonEngWordCount, (int)totalInterestWordCount );
}

#define PPARAMS(x) fprintf( OUTF, ", %s=%s", #x, opt.x.c_str() );
#define PPARAMD(x) fprintf( OUTF, ", %s=%d", #x, opt.x );

void writeParams_(FILE* OUTF, int wordcount)
{
	// output: input/dir f1 minwords_sentence  window  thres1  thres2

	fprintf( OUTF, "# %d words, %I64d occurrences\n", wordcount, (long long)totalInterestWordCount );

	if( opt.input.length() )
	    fprintf( OUTF, "# input=%s", opt.input.c_str() );
	else
	    fprintf( OUTF, "# dir=%s", opt.dir.c_str() );

    PPARAMS(f1)
    PPARAMD(minwords_sentence)
    PPARAMD(window)
    PPARAMD(dyn)
    PPARAMS(thres1)
    PPARAMS(thres2)
    
	if( opt.mode == 2 )
	    PPARAMD(top1)

    fprintf( OUTF, "\n" );
    
    if( opt.mode == 2 )
        fprintf( OUTF, "# %I64d bigram occurrences\n", (long long)allBigramOccurrences );
}

void outputTopUnigrams_(const char* top1gramFilename)
{
/*	ofstream OUTF;
	OUTF.open( top1gramFilename, ios_base::out );
    OUTF.setf(ios::fixed, ios::floatfield);
    OUTF.precision(3);
*/
	FILE* OUTF;
	OUTF = fopen( top1gramFilename, "w" );

	int min1gramfreq2 = int( opt.min1gramprob * totalInterestWordCount );
	int min1gramfreq = max( opt.min1gramfreq, min1gramfreq2 );
	printf( "Words cut-off frequency: %d\n", min1gramfreq );

    typedef pair<int, int> P;
    vector<P> wIDfreqs;
    
    sortVec2Pairs<int>( unigram2freq, interestWord2ID.size(), wIDfreqs );
	
    vector<int> wordIDs;
    for( auto iter=wIDfreqs.begin(); iter != wIDfreqs.end(); ++iter ){
        // wIDfreqs are already ordered in descending orders by the frequencies
        if( iter->second < min1gramfreq )
            break;
        wordIDs.push_back(iter->first);
    }
    
	printf( "Saving %d words into '%s'...\n", wordIDs.size(), top1gramFilename );
    
	writeParams_( OUTF, wordIDs.size() );


	for(auto iter = wordIDs.begin(); iter != wordIDs.end(); ++iter){
	    int wid = *iter;
	    string& w = ID2word[wid];
		fprintf( OUTF, "%s\t%d\t%.3f\n", w.c_str(), unigram2freq[wid], log( unigram2freq[wid] / totalInterestWordCount ) );
	}
	printf("Done.\n");
}

void outputTopBigrams_(const char* topbigramFilename)
{
	FILE* OUTF;
	OUTF = fopen( topbigramFilename, "w" );

    typedef pair<int, int> P;
    vector<P> wIDfreqs;
    
	// sort central words according to their total frequencies
    sortVec2Pairs<int>( unigram2freq, interestWord2ID.size(), wIDfreqs );

	vector<int> wordIDs;
	int wordCount = 0;

    int i = 0;
	for( auto iter = wIDfreqs.begin(); iter != wIDfreqs.end(); ++iter ){
	    int wid = iter->first;
	    i++;
	    
	    // In case some words in unigram2freq are not in bigram2freq
	    // Maybe this check is unnecessary
	    I2I& pw_hash = bigram2freq[wid];
	    if( pw_hash.size() == 0 )
	        continue;

		int min2gramfreq2 = int( opt.min2gramprob * unigram2freq[wid] );
		int min2gramfreq = max( opt.min2gramfreq, min2gramfreq2 );

        // pw_hash->second is the second level hash
        int neighborCount = count_if( pw_hash.begin(), pw_hash.end(), 
                                      [ min2gramfreq ]( const P& p2 ){ return p2.second > min2gramfreq; } );
                                        
		// some words have all neighbors filtered. ignore them
		if( neighborCount > 0 ){
		    wordIDs.push_back(wid);
		    wordCount++;
		}
	}

	writeParams_( OUTF, wordCount );
    fprintf( OUTF, "Words:\n" );

    i = 0;
	for( auto iter = wordIDs.begin(); iter != wordIDs.end(); ++iter ){
	    i++;
	    int wid = *iter;
	    string& w = ID2word[wid];
	    // the unigram probability is defined as the fraction of bigrams
	    // whose second word is w, in all bigrams
	    fprintf( OUTF, "%s,%d,%.3f", w.c_str(), unigram2freq[wid], log( unigram2freq[wid] / allBigramOccurrences ) );

	    if( i < wordCount ){
	        if( i % 10 == 0 )
	            fprintf( OUTF, "\n" );
    	    else
    	        fprintf( OUTF, "\t" );
    	}
	}

    fprintf( OUTF, "\n\nBigrams:\n" );

	int bigramCount = 0;
	double allKeptBigramOccurrences = 0;

	printf( "Saving bigrams of %d focus words into '%s'...\n", wordCount, topbigramFilename );

    vector<P> wIDfreqs2;
    // reserve the maximum number of words
    wIDfreqs2.reserve(wordCount);

	int wc = 0;
    
	for( auto iter = wordIDs.begin(); iter != wordIDs.end(); ++iter ){
	    int wid = *iter;
	    string& w = ID2word[wid];
		int min2gramfreq2 = int( opt.min2gramprob * unigram2freq[wid] );
		int min2gramfreq = max( opt.min2gramfreq, min2gramfreq2 );

        wIDfreqs2.clear();
        
        for( auto iter = bigram2freq[wid].begin(); iter != bigram2freq[wid].end(); ++iter ){
            if( iter->second >= min2gramfreq )
                wIDfreqs2.push_back( P(iter->first, iter->second) );
        }
                
        sort( wIDfreqs2.begin(), wIDfreqs2.end(), [](const P& p1, const P& p2) { return ( p1.second > p2.second ); } );
        
		// neighbor words are words preceding w, not following w
		
		double neighborTotalOccur = 0;
		
		for( auto iter = wIDfreqs2.begin(); iter != wIDfreqs2.end(); ++iter )
		    neighborTotalOccur += iter->second;
		
		wc++;
		allKeptBigramOccurrences += neighborTotalOccur;

		int neighborCount = wIDfreqs2.size();

        fprintf( OUTF, "%d,%s,%d,%d,%d\n", wc, w.c_str(), neighborCount, (int)neighborTotalOccur, min2gramfreq );

        int i = 0;
		for( auto iter = wIDfreqs2.begin(); iter != wIDfreqs2.end(); ++iter ){
			const P& wIDfreq2 = *iter;
		    int wid2 = wIDfreq2.first;

            fprintf( OUTF, "\t%s,%d,%.3f", ID2word[wid2].c_str(), 
                            wIDfreq2.second, log( wIDfreq2.second / neighborTotalOccur ) );

			bigramCount++;
			i++;
			// 5 bigrams each line for humans to read easily
			if( i % 5 == 0 && i < neighborCount ){
				fprintf( OUTF, "\n" );
			}
		}
		fprintf( OUTF, "\n" );

		if( wc % 100 == 0 && wc < wordCount ){
		    printf("\r%d\r", wc);
		}
	}
	// for integrity check. allKeptBigramOccurrences should <= allBigramOccurrences
	fprintf( OUTF, "# Total kept bigram occurrences: %I64d\n", (long long)allKeptBigramOccurrences );

	printf( "\n%d bigrams of %d focus words are saved into '%s'\n", bigramCount, wordCount, topbigramFilename );

    if( allKeptBigramOccurrences > allBigramOccurrences ){
        raise(5);
	    // "BUG: kept bigram occurs allKeptBigramOccurrences > all bigram occurs allBigramOccurrences\n";
	}
}
