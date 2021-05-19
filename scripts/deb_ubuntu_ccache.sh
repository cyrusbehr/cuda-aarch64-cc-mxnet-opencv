#!/usr/bin/env bash

set -ex

pushd .

apt update || true
apt install -y \
    libxslt1-dev \
    docbook-xsl \
    xsltproc \
    libxml2-utils

apt install -y --no-install-recommends \
    autoconf \
    asciidoc \
    xsltproc

mkdir -p /work/deps
cd /work/deps

git clone --recursive -b v3.4.2 https://github.com/ccache/ccache.git

cd ccache

./autogen.sh
# Manually specify x86 gcc versions so that this script remains compatible with dockcross (which uses an ARM based gcc
# by default).
CC=/usr/bin/gcc CXX=/usr/bin/g++ ./configure

# Don't build documentation #11214
#perl -pi -e 's!\s+\Q$(installcmd) -d $(DESTDIR)$(mandir)/man1\E!!g' Makefile
#perl -pi -e 's!\s+\Q-$(installcmd) -m 644 ccache.1 $(DESTDIR)$(mandir)/man1/\E!!g' Makefile
make -j$(nproc)
make install

rm -rf /work/deps/ccache

popd

