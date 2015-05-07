#!/bin/bash

built_sources=""

headers=`find .. -name "*.h" -a -not -name libmesh_config.h -type f | LC_COLLATE=POSIX sort`

for header_with_path in $headers ; do

    #echo $header_with_path
    header=`basename $header_with_path`
    #echo $header
    built_sources="$built_sources $header"
done

specializations=`find .. -name "*specializations" -type f | sort`

for specialization_with_path in $specializations ; do

    #echo $specialization_with_path
    specialization=`basename $specialization_with_path`
    #echo $specialization
    built_sources="$built_sources $specialization"
done

cat <<EOF > Makefile.am
# Note - this file is automatically generated by $0
# do not edit manually

#
# include the magic script!
EXTRA_DIST = rebuild_makefile.sh

EOF

echo -n "BUILT_SOURCES =" >> Makefile.am
for built_src in $built_sources ; do
    echo " \\" >> Makefile.am
    echo -n "        "$built_src >> Makefile.am
done

echo >> Makefile.am
echo >> Makefile.am
echo "DISTCLEANFILES = \$(BUILT_SOURCES)" >> Makefile.am


# handle contrib directly
cat <<EOF >> Makefile.am

#
# contrib rules
if LIBMESH_ENABLE_FPARSER

fparser.hh: \$(top_srcdir)/contrib/fparser/fparser.hh
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

fparser_ad.hh: \$(top_srcdir)/contrib/fparser/fparser_ad.hh
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

  BUILT_SOURCES  += fparser.hh fparser_ad.hh
  DISTCLEANFILES += fparser.hh fparser_ad.hh

endif

if LIBMESH_ENABLE_NANOFLANN

nanoflann.hpp: \$(top_srcdir)/contrib/nanoflann/include/nanoflann.hpp
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

  BUILT_SOURCES  += nanoflann.hpp
  DISTCLEANFILES += nanoflann.hpp

endif

if LIBMESH_ENABLE_EXODUS_V509

exodusII.h: \$(top_srcdir)/contrib/exodusii/v5.09/include/exodusII.h
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

  BUILT_SOURCES  += exodusII.h
  DISTCLEANFILES += exodusII.h

exodusII_ext.h: \$(top_srcdir)/contrib/exodusii/v5.09/include/exodusII_ext.h
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

  BUILT_SOURCES  += exodusII_ext.h
  DISTCLEANFILES += exodusII_ext.h

endif

if LIBMESH_ENABLE_EXODUS_V522

exodusII.h: \$(top_srcdir)/contrib/exodusii/v5.22/exodus/cbind/include/exodusII.h
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

  BUILT_SOURCES  += exodusII.h
  DISTCLEANFILES += exodusII.h

endif

if LIBMESH_ENABLE_NETCDF_V3

netcdf.h: \$(top_srcdir)/contrib/netcdf/v3/netcdf.h
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

  BUILT_SOURCES  += netcdf.h
  DISTCLEANFILES += netcdf.h

endif

if LIBMESH_ENABLE_NETCDF_V4

netcdf.h: \$(top_srcdir)/contrib/netcdf/v4/include/netcdf.h
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

  BUILT_SOURCES  += netcdf.h
  DISTCLEANFILES += netcdf.h

endif

if LIBMESH_INSTALL_HINNANT_UNIQUE_PTR

unique_ptr.hpp: \$(top_srcdir)/contrib/unique_ptr/unique_ptr.hpp
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) \$< \$@

  BUILT_SOURCES  += unique_ptr.hpp
  DISTCLEANFILES += unique_ptr.hpp

endif

EOF



# handle libmesh_config.h
cat <<EOF >> Makefile.am
#
# libmesh_config.h rule
libmesh_config.h: \$(top_builddir)/include/libmesh_config.h
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

  BUILT_SOURCES  += libmesh_config.h
  DISTCLEANFILES += libmesh_config.h

EOF



# now automatically handle our headers
cat <<EOF >> Makefile.am
#
# libMesh header rules
EOF
for header_with_path in $headers $specializations ; do
    header=`basename $header_with_path`
    source=`echo $header_with_path | sed 's/../$(top_srcdir)\/include/'`
    #echo "source = $source"
    cat <<EOF >> Makefile.am
$header: $source
	\$(AM_V_GEN)rm -f \$@ && \$(LN_S) -f \$< \$@

EOF
done
#cat Makefile.am
