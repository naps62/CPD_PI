#!/usr/bin/perl -w
use warnings;

@cases = ("tiny","small","medium","big","huge");
$samples = 10;

for ( $i = 0 ; $i < $samples ; ++$i )
{
	foreach $case ( @cases )
	{
#		system( "echo" , $i, "bin/polu.optim.bytes" , "data/xml/$case.param.xml" );
		system( "bin/polu.optim.bytes" , "data/xml/$case.param.xml" );
	}
}
