CXX      = g++
MPICXX   = mpicxx
CXXFLAGS = -O2 -std=c++17
OMPFLAGS = -fopenmp

# ── Targets ────────────────────────────────────────────────────────────────────
.PHONY: all clean

all: kmeans kmeans_omp kmeans_mpi

# Serial baseline
kmeans: kmeans.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Built: kmeans"

# OpenMP shared-memory parallel
kmeans_omp: kmeans_omp.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "Built: kmeans_omp"

# MPI + OpenMP hybrid distributed
kmeans_mpi: kmeans_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "Built: kmeans_mpi"

clean:
	rm -f kmeans kmeans_omp kmeans_mpi \
	      cluster_labels.bin cluster_labels_omp.bin cluster_labels_mpi.bin \
	      benchmark_results.csv
	@echo "Cleaned"
