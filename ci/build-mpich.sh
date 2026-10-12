#!/usr/bin/env bash
set -euo pipefail

alloc=${MPICH_CI_ALLOCATION:-Aurora_testing}
root=${MPICH_CI_ROOT:-/lus/flare/projects/$alloc/aurora-mpich-ci}
ref=${MPICH_REF:-aurora_test}
src=$root/src/mpich
prefix=$root/mpich
log=$root/build.log

section_start() {
    echo -e "\e[0Ksection_start:$(date +%s):$1\r\e[0K${2:-${1%[\[]*}}"
}

section_end() {
    echo -e "\e[0Ksection_end:$(date +%s):${1%[\[]*}\r\e[0K"
}

stage() {
    local name rc=0
    name=$(echo "$*" | tr -cs '[:alnum:]' '_')
    section_start "$name[collapsed=true]"
    "$@" 2>&1 | tee -a "$log" || rc=$?
    section_end "$name[collapsed=true]"
    [ "$rc" = 0 ] || exit 1
}

module reset
module unload mpich
module load autoconf/2.72 automake/1.16.5 hwloc/2.12.2

hwloc=$(dirname "$(dirname "$(command -v hwloc-info)")")

mkdir -p "$root"
touch "$log"

curl -fsS --max-time 15 -o /dev/null https://github.com/ \
    || export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128

if [ ! -d "$src/.git" ]; then
    rm -rf "$src"
    mkdir -p "$src"
    git -C "$src" init -q
    git -C "$src" remote add origin https://github.com/pmodels/mpich.git
fi

stage git -C "$src" fetch --depth 1 origin "$ref"
stage git -C "$src" checkout --detach FETCH_HEAD
sha=$(git -C "$src" rev-parse HEAD)
stamp="$hwloc $(sha256sum "$0" | cut -d' ' -f1) $sha"

if [ "$(cat "$prefix/.stamp" 2>/dev/null)" = "$stamp" ]; then
    echo "==> reusing $prefix at $sha"
    exit 0
fi

rm -rf "$prefix" "$log"
touch "$log"

stage git -C "$src" submodule update --init --recursive --depth 1

[ -d "$root/bats-core" ] \
    || stage git clone --depth 1 -b v1.11.0 https://github.com/bats-core/bats-core.git "$root/bats-core"

cd "$src"
stage ./autogen.sh
stage ./configure --prefix="$prefix" \
    --disable-maintainer-mode --disable-silent-rules \
    --enable-shared --enable-static \
    --with-pm=no --enable-romio --without-ibverbs --enable-wrapper-rpath=yes \
    --with-yaksa=embedded --with-ze --with-ch4-shmmods=posix,gpudirect \
    --with-hwloc="$hwloc" \
    --enable-fortran --with-slurm=no --with-pmi=pmix --with-pmix=/usr \
    --without-cuda --without-hip \
    --with-device=ch4:ofi --with-libfabric=/opt/cray/libfabric/2.3.1 \
    --enable-libxml2 --with-datatype-engine=yaksa --with-xpmem=/usr \
    --with-file-system=daos+lustre+nfs+ufs --with-daos=/usr \
    --enable-timer-type=linux86_cycle \
    --enable-fast=O3,alwaysinline,avx,avx2,avx512f,sse2 \
    --enable-g=no --disable-debuginfo --enable-error-checking=runtime \
    --without-valgrind --enable-ch4-mt=runtime \
    --with-ze=/usr --enable-ze-native=pvc --disable-opencl \
    CC=icx CXX=icpx FC=ifx F77=ifx
stage make -j"${MPICH_CI_JOBS:-48}"
stage make install

grep -q 'device *: ch4:ofi' "$log"
grep -q 'gpu support *: ZE' "$log"
printf '%s\n' "$stamp" >"$prefix/.stamp"
echo "==> built $prefix at $sha"
