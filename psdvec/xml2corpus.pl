use strict; 
use warnings;
use XML::LibXML;
use File::Find;

my $rootdir = "D:/corpus/rcv1/";
my $fc = 0;
my $totalbytes = 0;

find({ wanted => \&process_file, no_chdir => 1 }, $rootdir);
my $totalMB = int( $totalbytes / 1024 / 1024 );
print STDERR "$fc files processed, totally $totalMB MB\n";

sub process_file {
    if ( /\.xml$/ ) {
        my $doc = XML::LibXML->load_xml(location => $_);
        for my $textnode ( $doc->findnodes('/newsitem/text') ){
            print $textnode->textContent();
            $totalbytes += length( $textnode->textContent() );
            $totalMB = int( $totalbytes / 1024 / 1024 );
        }
        $fc++;
        if( $fc % 500 == 0 ){
            print STDERR "\r$fc $totalMB\r";
        }
    }
}
