#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

void generate_data(std::vector<int>& data, int size) {
    for (int i = 0; i < size; ++i) {
        data.push_back(rand() % 1000);
    }
}

void print_vector(const std::vector<int>& vec) {
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int DATA_SIZE = 1000;
    std::vector<int> data;
    std::vector<int> local_data;

    if (world_rank == 0) {
        srand(time(0));
        generate_data(data, DATA_SIZE);
        std::cout << "Data before sorting: ";
        print_vector(data);
    }

    int local_size = DATA_SIZE / world_size;
    int remainder = DATA_SIZE % world_size;

    std::vector<int> send_counts(world_size, local_size);
    std::vector<int> send_displs(world_size, 0);
    std::vector<int> recv_counts(world_size);
    std::vector<int> recv_displs(world_size, 0);

    for (int i = 0; i < remainder; ++i) {
        send_counts[i]++;
    }

    for (int i = 1; i < world_size; ++i) {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
    }

    local_data.resize(send_counts[world_rank]);
    MPI_Scatterv(data.data(), send_counts.data(), send_displs.data(), MPI_INT, local_data.data(), local_data.size(), MPI_INT, 0, MPI_COMM_WORLD);

    std::sort(local_data.begin(), local_data.end());

    std::vector<int> samples(world_size);
    for (int i = 0; i < world_size; ++i) {
        samples[i] = local_data[i * local_data.size() / world_size];
    }

    std::vector<int> gathered_samples;
    if (world_rank == 0) {
        gathered_samples.resize(world_size * world_size);
    }
    MPI_Gather(samples.data(), world_size, MPI_INT, gathered_samples.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> pivots(world_size - 1);
    if (world_rank == 0) {
        std::sort(gathered_samples.begin(), gathered_samples.end());
        for (int i = 0; i < world_size - 1; ++i) {
            pivots[i] = gathered_samples[(i + 1) * world_size];
        }
    }

    MPI_Bcast(pivots.data(), world_size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::vector<int>> buckets(world_size);
    for (int val : local_data) {
        int bucket_idx = std::upper_bound(pivots.begin(), pivots.end(), val) - pivots.begin();
        buckets[bucket_idx].push_back(val);
    }

    for (int i = 0; i < world_size; ++i) {
        recv_counts[i] = buckets[i].size();
    }

    MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = std::accumulate(send_counts.begin(), send_counts.end(), 0);
    std::vector<int> recv_buffer(total_recv);

    for (int i = 1; i < world_size; ++i) {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    std::vector<int> send_buffer;
    for (const auto& bucket : buckets) {
        send_buffer.insert(send_buffer.end(), bucket.begin(), bucket.end());
    }

    MPI_Alltoallv(send_buffer.data(), recv_counts.data(), send_displs.data(), MPI_INT, recv_buffer.data(), send_counts.data(), recv_displs.data(), MPI_INT, MPI_COMM_WORLD);

    std::sort(recv_buffer.begin(), recv_buffer.end());

    if (world_rank == 0) {
        data.clear();
        data.resize(DATA_SIZE);
    }

    MPI_Gatherv(recv_buffer.data(), total_recv, MPI_INT, data.data(), send_counts.data(), recv_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Data after sorting: ";
        print_vector(data);
    }

    MPI_Finalize();
    return 0;
}

