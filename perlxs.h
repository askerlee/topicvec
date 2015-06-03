// http://ftp.ledas.ac.uk/software/lheasoft/lheasoft6.3.1/source/heacore/pil/perl/Av_CharPtrPtr.c

/* Used by the INPUT typemap for char**.
 * Will convert a Perl AV* (containing strings) to a C char**.
 */
char** XS_unpack_charPtrPtr( SV* rv )
{
	AV *av;
	SV **ssv;
	char **s;
	int avlen;
	int x;

	if( SvROK( rv ) && (SvTYPE(SvRV(rv)) == SVt_PVAV) )
		av = (AV*)SvRV(rv);
    else {
		warn("XS_unpack_charPtrPtr: rv was not an AV ref");
		return( (char**)NULL );
	}

	/* is it empty? */
	avlen = av_len(av);
	if( avlen < 0 ){
		warn("XS_unpack_charPtrPtr: array was empty");
		return( (char**)NULL );
	}

	/* av_len+2 == number of strings, plus 1 for an end-of-array sentinel.
	 */
	s = (char **)safemalloc( sizeof(char*) * (avlen + 2) );
	if( s == NULL ){
		warn("XS_unpack_charPtrPtr: unable to malloc char**");
		return( (char**)NULL );
	}
	for( x = 0; x <= avlen; ++x ){
		ssv = av_fetch( av, x, 0 );
		if( ssv != NULL ){
			if( SvPOK( *ssv ) ){
				s[x] = (char *)safemalloc( SvCUR(*ssv) + 1 );
				if( s[x] == NULL )
					warn("XS_unpack_charPtrPtr: unable to malloc char*");
				else
					strcpy( s[x], SvPV( *ssv, PL_na ) );
			}
			else
				warn("XS_unpack_charPtrPtr: array elem %d was not a string.", x );
		}
		else
			s[x] = (char*)NULL;
	}
	s[x] = (char*)NULL; /* sentinel */
	return( s );
}

/* Used by the OUTPUT typemap for char**.
 * Will convert a C char** to a Perl AV*.
 */
void XS_pack_charPtrPtr( SV * st, char ** s )
{
	AV *av = newAV();
	SV *sv;
	char **c;

	for( c = s; *c != NULL; ++c ){
		sv = newSVpv( *c, 0 );
		av_push( av, sv );
	}
	free ( s );
	sv = newSVrv( st, NULL );	/* upgrade stack SV to an RV */
	SvREFCNT_dec( sv );	/* discard */
	SvRV( st ) = (SV*)av;	/* make stack RV point at our AV */
}


/* cleanup the temporary char** from XS_unpack_charPtrPtr */
void XS_release_charPtrPtr(char** s)
{
	char **c;
	for( c = s; *c != NULL; ++c )
		Safefree( *c );
	Safefree( s );
}
