#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <chrono>

// ── Binary file format ──────────────────────────────────────────────────────
// [int32: N][int32: D][float32 × N × D]
// ────────────────────────────────────────────────────────────────────────────

struct Dataset {
    int N, D;
    std::vector<float> data;   // flat row-major: data[i*D + j] = vector i, dim j
};

Dataset load_vectors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; exit(1); }

    Dataset ds;
    f.read(reinterpret_cast<char*>(&ds.N), sizeof(int));
    f.read(reinterpret_cast<char*>(&ds.D), sizeof(int));
    ds.data.resize((size_t)ds.N * ds.D);
    f.read(reinterpret_cast<char*>(ds.data.data()), (size_t)ds.N * ds.D * sizeof(float));
    std::cout << "Loaded " << ds.N << " vectors, " << ds.D << " dims\n";
    return ds;
}

// Euclidean distance squared between vector i and centroid k
float dist_sq(const float* vec, const float* centroid, int D) {
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = vec[d] - centroid[d];
        sum += diff * diff;
    }
    return sum;
}

// K-Means++ initialisation
std::vector<float> init_centroids_pp(const Dataset& ds, int K) {
    std::srand(42);
    int N = ds.N, D = ds.D;
    std::vector<float> centroids(K * D);
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
    return centroids;
}

// Assign each vector to nearest centroid
// Returns number of reassignments (0 = converged)
int assign_clusters(const Dataset& ds, const std::vector<float>& centroids,
                    std::vector<int>& labels, int K) {
    int N = ds.N, D = ds.D;
    int changes = 0;
    for (int i = 0; i < N; i++) {
        float best_dist = std::numeric_limits<float>::max();
        int   best_k    = 0;
        for (int k = 0; k < K; k++) {
            float d = dist_sq(&ds.data[i * D], &centroids[k * D], D);
            if (d < best_dist) { best_dist = d; best_k = k; }
        }
        if (labels[i] != best_k) { labels[i] = best_k; changes++; }
    }
    return changes;
}

// Recompute centroids from current assignments
void update_centroids(const Dataset& ds, const std::vector<int>& labels,
                      std::vector<float>& centroids, int K) {
    int N = ds.N, D = ds.D;
    std::vector<float> sums(K * D, 0.0f);
    std::vector<int>   counts(K, 0);

    for (int i = 0; i < N; i++) {
        int k = labels[i];
        counts[k]++;
        for (int d = 0; d < D; d++)
            sums[k * D + d] += ds.data[i * D + d];
    }

    for (int k = 0; k < K; k++) {
        if (counts[k] == 0) continue;
        for (int d = 0; d < D; d++)
            centroids[k * D + d] = sums[k * D + d] / counts[k];
    }
}

// Inertia = sum of squared distances from each point to its assigned centroid
// Used for quality tracking and NMI evaluation pipeline
double compute_inertia(const Dataset& ds, const std::vector<float>& centroids,
                       const std::vector<int>& labels) {
    double inertia = 0.0;
    for (int i = 0; i < ds.N; i++)
        inertia += dist_sq(&ds.data[i * ds.D], &centroids[labels[i] * ds.D], ds.D);
    return inertia;
}

void save_labels(const std::vector<int>& labels, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    int N = labels.size();
    f.write(reinterpret_cast<const char*>(&N), sizeof(int));
    f.write(reinterpret_cast<const char*>(labels.data()), N * sizeof(int));
    std::cout << "Cluster labels saved to " << path << "\n";
}

int main(int argc, char* argv[]) {
    std::string input  = (argc > 1) ? argv[1] : "vectors.bin";
    int K              = (argc > 2) ? std::atoi(argv[2]) : 10;
    int MAX_ITER       = (argc > 3) ? std::atoi(argv[3]) : 50;

    std::cout << "K=" << K << "  MAX_ITER=" << MAX_ITER << "\n";

    Dataset ds = load_vectors(input);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> centroids = init_centroids_pp(ds, K);
    std::vector<int>   labels(ds.N, 0);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        int changes = assign_clusters(ds, centroids, labels, K);
        update_centroids(ds, labels, centroids, K);
        std::cout << "Iter " << iter + 1 << ": " << changes << " reassignments\n";
        if (changes == 0) { std::cout << "Converged at iter " << iter+1 << "\n"; break; }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Time: " << elapsed << " seconds\n";

    double inertia = compute_inertia(ds, centroids, labels);
    std::cout << "Inertia: " << inertia << "\n";

    save_labels(labels, "cluster_labels.bin");

    std::vector<int> counts(K, 0);
    for (int l : labels) counts[l]++;
    std::cout << "Cluster sizes: ";
    for (int k = 0; k < K; k++) std::cout << counts[k] << " ";
    std::cout << "\n";

    return 0;
}