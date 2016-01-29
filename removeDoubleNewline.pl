$wc = 0;
$doubleNewlineCount = 0;
while(<>){
    $wc++;
    if( /^\r?\n$/ ){
        if( $lastIsNewline ){
            $lastIsNewline = 0;
            $doubleNewlineCount++;
            next;
        }
        else{
            print;
            $lastIsNewline = 1;
        }
    }
    else{
        print;
    }
    if( $wc % 1000 == 0 ){
        print STDERR "\r$wc $doubleNewlineCount\r";
    }
}
print STDERR "$wc $doubleNewlineCount\n";
