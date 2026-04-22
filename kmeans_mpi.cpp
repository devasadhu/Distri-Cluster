#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <omp.h>
#include <mpi.h>

// ── Dataset ─────────────────────────────────────────────────────────────────

struct Dataset {
    int N, D;
    std::vector<float> data;
};

// FIX: Only rank 0 reads the file. Data is then broadcast to all ranks.
// Previously all ranks read simultaneously — correct output but wastes I/O
// bandwidth proportional to number of ranks, and does not scale.
// Rank-0-read + MPI_Bcast is the standard distributed pattern.
Dataset load_and_distribute(const std::string& path, int rank) {
    Dataset ds;

    if (rank == 0) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Cannot open " << path << "\n"; MPI_Abort(MPI_COMM_WORLD, 1); }
        f.read(reinterpret_cast<char*>(&ds.N), sizeof(int));
        f.read(reinterpret_cast<char*>(&ds.D), sizeof(int));
        ds.data.resize((size_t)ds.N * ds.D);
        f.read(reinterpret_cast<char*>(ds.data.data()), (size_t)ds.N * ds.D * sizeof(float));
        std::cout << "Loaded " << ds.N << " vectors, " << ds.D << " dims\n";
    }

    // Broadcast header so all ranks can allocate
    MPI_Bcast(&ds.N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ds.D, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) ds.data.resize((size_t)ds.N * ds.D);

    // Broadcast full data — all ranks need it for local slicing
    MPI_Bcast(ds.data.data(), ds.N * ds.D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    return ds;
}

// ── Distance ─────────────────────────────────────────────────────────────────

float dist_sq(const float* vec, const float* centroid, int D) {
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = vec[d] - centroid[d];
        sum += diff * diff;
    }
    return sum;
}

// ── K-Means++ init (rank 0 only, then broadcast) ────────────────────────────

std::vector<float> init_centroids_pp(const Dataset& ds, int K, int rank) {
    int N = ds.N, D = ds.D;
    std::vector<float> centroids(K * D, 0.0f);

    if (rank == 0) {
        std::srand(42);
        std::vector<float> min_dist(N, std::numeric_limits<float>::max());

        int first = std::rand() % N;
        std::copy(ds.data.begin() + first * D,
                  ds.data.begin() + first * D + D,
                  centroids.begin());

        for (int k = 1; k < K; k++) {
            float total = 0.0f;
            for (int i = 0; i < N; i++) {
                float d = dist_sq(&ds.data[i * D], &centroids[(k-1) * D], D);
                if (d < min_dist[i]) min_dist[i] = d;
                total += min_dist[i];
            }
            float threshold = ((float)std::rand() / RAND_MAX) * total;
            float cumsum = 0.0f;
            int chosen = 0;
            for (int i = 0; i < N; i++) {
                cumsum += min_dist[i];
                if (cumsum >= threshold) { chosen = i; break; }
            }
            std::copy(ds.data.begin() + chosen * D,
                      ds.data.begin() + chosen * D + D,
                      centroids.begin() + k * D);
        }
        std::cout << "K-Means++ initialisation done\n";
    }

    MPI_Bcast(centroids.data(), K * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return centroids;
}

// ── Local assign (OpenMP inside each MPI rank) ───────────────────────────────

int assign_local(const std::vector<float>& local_data, int local_N, int D,
                 const std::vector<float>& centroids, std::vector<int>& labels, int K) {
    int changes = 0;
    #pragma omp parallel for reduction(+:changes) schedule(dynamic, 64)
    for (int i = 0; i < local_N; i++) {
        float best_dist = std::numeric_limits<float>::max();
        int   best_k    = 0;
        for (int k = 0; k < K; k++) {
            float d = dist_sq(&local_data[i * D], &centroids[k * D], D);
            if (d < best_dist) { best_dist = d; best_k = k; }
        }
        if (labels[i] != best_k) { labels[i] = best_k; changes++; }
    }
    return changes;
}

// ── Global centroid update via MPI_Allreduce ─────────────────────────────────

void update_centroids_mpi(const std::vector<float>& local_data, int local_N, int D,
                          const std::vector<int>& labels,
                          std::vector<float>& centroids, int K) {

    std::vector<float> local_sums(K * D, 0.0f);
    std::vector<int>   local_counts(K, 0);

    #pragma omp parallel
    {
        std::vector<float> thread_sums(K * D, 0.0f);
        std::vector<int>   thread_counts(K, 0);

        // FIX: removed nowait — barrier here ensures all threads finish
        // accumulating before any thread enters the critical section below
        #pragma omp for
        for (int i = 0; i < local_N; i++) {
            int k = labels[i];
            thread_counts[k]++;
            for (int d = 0; d < D; d++)
                thread_sums[k * D + d] += local_data[i * D + d];
        }

        #pragma omp critical
        {
            for (int k = 0; k < K; k++) {
                local_counts[k] += thread_counts[k];
                for (int d = 0; d < D; d++)
                    local_sums[k * D + d] += thread_sums[k * D + d];
            }
        }
    }

    std::vector<float> global_sums(K * D, 0.0f);
    std::vector<int>   global_counts(K, 0);

    MPI_Allreduce(local_sums.data(),   global_sums.data(),   K * D, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_counts.data(), global_counts.data(), K,     MPI_INT,   MPI_SUM, MPI_COMM_WORLD);

    for (int k = 0; k < K; k++) {
        if (global_counts[k] == 0) continue;
        for (int d = 0; d < D; d++)
            centroids[k * D + d] = global_sums[k * D + d] / global_counts[k];
    }
}

// ── Gather all labels to rank 0 and save ─────────────────────────────────────

void gather_and_save_labels(const std::vector<int>& local_labels, int local_N,
                            int N, int size, int rank) {
    // Build recvcounts and displacements for variable-length gather
    std::vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; r++) {
        recvcounts[r] = (r < size - 1) ? N / size : N - (size - 1) * (N / size);
        displs[r]     = r * (N / size);
    }

    std::vector<int> all_labels;
    if (rank == 0) all_labels.resize(N);

    MPI_Gatherv(local_labels.data(), local_N, MPI_INT,
                all_labels.data(), recvcounts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream f("cluster_labels_mpi.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(&N), sizeof(int));
        f.write(reinterpret_cast<const char*>(all_labels.data()), N * sizeof(int));
        std::cout << "Labels saved to cluster_labels_mpi.bin\n";

        std::vector<int> counts(10, 0);  // K known at call site — print distribution
        // Note: K not passed here; caller prints distribution instead
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string input = (argc > 1) ? argv[1] : "vectors.bin";
    int K             = (argc > 2) ? std::atoi(argv[2]) : 10;
    int MAX_ITER      = (argc > 3) ? std::atoi(argv[3]) : 50;
    int threads       = (argc > 4) ? std::atoi(argv[4]) : omp_get_max_threads();

    omp_set_num_threads(threads);

    if (rank == 0)
        std::cout << "MPI ranks=" << size << "  Threads/rank=" << threads
                  << "  K=" << K << "  MAX_ITER=" << MAX_ITER << "\n";

    // Rank 0 reads, all ranks receive via Bcast
    Dataset ds = load_and_distribute(input, rank);
    int N = ds.N, D = ds.D;

    // Each rank takes a contiguous slice
    int local_N     = N / size;
    int local_start = rank * local_N;
    if (rank == size - 1) local_N = N - local_start;

    std::vector<float> local_data(
        ds.data.begin() + (size_t)local_start * D,
        ds.data.begin() + (size_t)(local_start + local_N) * D
    );

    if (rank == 0)
        std::cout << "Data distributed. Each rank handles ~" << N / size << " vectors\n";

    std::vector<float> centroids = init_centroids_pp(ds, K, rank);
    std::vector<int>   labels(local_N, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        int local_changes = assign_local(local_data, local_N, D, centroids, labels, K);

        int global_changes = 0;
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        update_centroids_mpi(local_data, local_N, D, labels, centroids, K);

        if (rank == 0)
            std::cout << "Iter " << iter+1 << ": " << global_changes << " reassignments\n";

        if (global_changes == 0) {
            if (rank == 0) std::cout << "Converged at iter " << iter+1 << "\n";
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Compute local inertia, reduce to global
    double local_inertia = 0.0;
    for (int i = 0; i < local_N; i++)
        local_inertia += dist_sq(&local_data[i * D], &centroids[labels[i] * D], D);

    double global_inertia = 0.0;
    MPI_Reduce(&local_inertia, &global_inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Time: " << elapsed << " seconds"
                  << "  (ranks=" << size << " threads=" << threads << ")\n";
        std::cout << "Inertia: " << global_inertia << "\n";
    }

    // Gather all labels to rank 0 and save
    gather_and_save_labels(labels, local_N, N, size, rank);

    if (rank == 0) {
        std::vector<int> counts(K, 0);
        // Re-read saved labels for distribution print
        std::ifstream lf("cluster_labels_mpi.bin", std::ios::binary);
        int Ncheck; lf.read(reinterpret_cast<char*>(&Ncheck), sizeof(int));
        std::vector<int> all_labels(Ncheck);
        lf.read(reinterpret_cast<char*>(all_labels.data()), Ncheck * sizeof(int));
        for (int l : all_labels) counts[l]++;
        std::cout << "Cluster sizes: ";
        for (int k = 0; k < K; k++) std::cout << counts[k] << " ";
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}