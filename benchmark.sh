#!/usr/bin/env bash
# =============================================================================
# benchmark.sh — Distri-Cluster HPC Scaling Study
#
# Runs all K-Means variants across thread/rank configurations, extracts
# wall-clock times, and writes benchmark_results.csv for plotting.
#
# Usage:
#   chmod +x benchmark.sh
#   ./benchmark.sh                          # uses vectors.bin, K=10
#   ./benchmark.sh text_vectors.bin 20      # custom file and K
#
# Output:
#   benchmark_results.csv
#   Console table with speedup and efficiency
# =============================================================================

VECTORS="${1:-vectors.bin}"
K="${2:-10}"
MAX_ITER=50
CSV="benchmark_results.csv"

# ── Colour output ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

die() { echo -e "${RED}ERROR: $1${NC}"; exit 1; }

# ── Sanity checks ──────────────────────────────────────────────────────────────
[ -f "$VECTORS" ]     || die "Vector file '$VECTORS' not found. Run extract_vectors.py first."
[ -f "./kmeans" ]     || die "Binary 'kmeans' not found. Run: make"
[ -f "./kmeans_omp" ] || die "Binary 'kmeans_omp' not found. Run: make"
[ -f "./kmeans_mpi" ] || die "Binary 'kmeans_mpi' not found. Run: make"
command -v mpirun &>/dev/null || die "mpirun not found. Install openmpi-bin."

echo -e "${BOLD}${CYAN}=== Distri-Cluster Benchmark ===${NC}"
echo "Vectors : $VECTORS"
echo "K       : $K"
echo "MAX_ITER: $MAX_ITER"
echo ""

# ── Helper: run binary, extract time from stdout ───────────────────────────────
# Returns time in seconds (float) via $LAST_TIME
run_and_time() {
    local cmd="$@"
    local output
    output=$($cmd 2>&1)
    local status=$?
    if [ $status -ne 0 ]; then
        echo -e "${RED}  FAILED: $cmd${NC}"
        echo "$output" | tail -5
        LAST_TIME="ERR"
        return 1
    fi
    # Extract "Time: X.XXXX seconds" from output
    LAST_TIME=$(echo "$output" | grep -oP '(?<=Time: )\d+\.\d+')
    if [ -z "$LAST_TIME" ]; then
        LAST_TIME="ERR"
        echo -e "${RED}  Could not parse time from output${NC}"
        echo "$output" | tail -3
        return 1
    fi
    echo -e "${GREEN}  Time: ${LAST_TIME}s${NC}"
}

# Write CSV header
echo "variant,ranks,threads_per_rank,total_procs,time_seconds" > "$CSV"

# ── 1. Serial baseline ─────────────────────────────────────────────────────────
echo -e "\n${BOLD}[1/4] Serial baseline${NC}"
run_and_time ./kmeans "$VECTORS" "$K" "$MAX_ITER"
SERIAL_TIME="$LAST_TIME"
echo "serial,1,1,1,$SERIAL_TIME" >> "$CSV"
echo "  → Serial time (reference): ${SERIAL_TIME}s"

# ── 2. OpenMP — thread sweep ───────────────────────────────────────────────────
echo -e "\n${BOLD}[2/4] OpenMP thread sweep (1 node)${NC}"
for T in 1 2 4 8; do
    echo -n "  OMP threads=$T ... "
    run_and_time ./kmeans_omp "$VECTORS" "$K" "$MAX_ITER" "$T"
    echo "omp,1,$T,$T,$LAST_TIME" >> "$CSV"
done

# ── 3. MPI — rank sweep, 1 thread/rank ────────────────────────────────────────
echo -e "\n${BOLD}[3/4] MPI rank sweep (1 thread/rank)${NC}"
for R in 1 2 4; do
    echo -n "  MPI ranks=$R threads=1 ... "
    run_and_time mpirun --oversubscribe -np "$R" ./kmeans_mpi "$VECTORS" "$K" "$MAX_ITER" 1
    echo "mpi,$R,1,$R,$LAST_TIME" >> "$CSV"
done

# ── 4. MPI+OMP hybrid — best of both ──────────────────────────────────────────
echo -e "\n${BOLD}[4/4] MPI+OMP hybrid (ranks × threads)${NC}"
for R in 2 4; do
    for T in 2 4; do
        TOTAL=$(( R * T ))
        echo -n "  MPI ranks=$R × OMP threads=$T (total=$TOTAL procs) ... "
        run_and_time mpirun --oversubscribe -np "$R" ./kmeans_mpi "$VECTORS" "$K" "$MAX_ITER" "$T"
        echo "mpi_omp,$R,$T,$TOTAL,$LAST_TIME" >> "$CSV"
    done
done

# ── Summary table ──────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}=== Scaling Summary ===${NC}"
printf "%-20s %-8s %-8s %-10s %-10s %-10s\n" "Variant" "Ranks" "Threads" "TotalProcs" "Time(s)" "Speedup"
printf "%-20s %-8s %-8s %-10s %-10s %-10s\n" "--------------------" "--------" "--------" "----------" "--------" "--------"

while IFS=',' read -r variant ranks threads total time; do
    [ "$variant" = "variant" ] && continue  # skip header
    if [ "$time" = "ERR" ]; then
        speedup="ERR"
    else
        speedup=$(echo "scale=3; $SERIAL_TIME / $time" | bc 2>/dev/null || echo "?")
    fi
    printf "%-20s %-8s %-8s %-10s %-10s %-10s\n" "$variant" "$ranks" "$threads" "$total" "$time" "$speedup"
done < "$CSV"

echo -e "\n${GREEN}Results saved to: $CSV${NC}"
echo "Run next:  python3 plot_scaling.py"
