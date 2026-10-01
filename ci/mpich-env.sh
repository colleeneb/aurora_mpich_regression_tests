root=${MPICH_CI_ROOT:-/lus/flare/projects/datascience/aurora-mpich-ci}
prefix=$root/mpich

[ -f "$prefix/.stamp" ] || { echo "no MPICH build at $prefix" >&2; exit 1; }

module reset
module unload mpich
module load hwloc/2.12.2

export PATH=$prefix/bin:$root/bats-core/bin:$PATH
export LD_LIBRARY_PATH=$prefix/lib:/opt/cray/libfabric/2.3.1/lib:/opt/cray/libfabric/2.3.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PALS_PMI=pmix
export FI_PROVIDER='cxi,tcp;ofi_rxm'
export HWLOC_XMLFILE=/run/hwloc/topology.xml
export HWLOC_THISSYSTEM=1
export MPIR_CVAR_CH4_SHM_POSIX_TOPO_ENABLE=1
export MPIR_CVAR_INIT_SKIP_PMI_BARRIER=0
export MPIR_CVAR_CH4_OFI_MAX_NICS=1
export FI_CXI_RDZV_THRESHOLD=16384
export FI_CXI_RDZV_EAGER_SIZE=2048
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=1024
export FI_CXI_OFLOW_BUF_SIZE=12582912
export FI_CXI_OFLOW_BUF_COUNT=3
export FI_CXI_RX_MATCH_MODE=hardware
export FI_CXI_REQ_BUF_MIN_POSTED=6
export FI_CXI_REQ_BUF_SIZE=12582912
export FI_CXI_REQ_BUF_MAX_CACHED=0
export FI_MR_CACHE_MAX_SIZE=-1
export FI_MR_CACHE_MAX_COUNT=524288
export MPIR_CVAR_CH4_OFI_EAGER_THRESHOLD=1000000
export MPIR_CVAR_ERROR_CHECKING=1
export MPIR_CVAR_PROGRESS_TIMEOUT=600

probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
printf '#include <mpi.h>\nint main(void){return MPI_Init(0, 0);}\n' >"$probe/probe.c"
mpicc "$probe/probe.c" -o "$probe/probe"
ldd "$probe/probe" | grep -q "$prefix/lib/libmpi.so" \
    || { echo "libmpi does not resolve to the CI build" >&2; exit 1; }
