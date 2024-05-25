#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>

void generate_data(std::vector<int>& data, int size) {
    srand(time(0));
    for (int i = 0; i < size; ++i) {
        int value;
        std::cin >> value;
        if (std::cin.fail() || value < 0 || value >= 1000) {
            std::cerr << "Invalid data input: " << value << std::endl;
            // Handle invalid data (e.g., ignore, discard, or prompt for correction)
        } else {
            data.push_back(value);
        }
        std::cin.clear();
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
        if (data.empty()) {
            std::cerr << "Error: Input data is empty." << std::endl;
            MPI_Finalize();
            return 1;
        }

        std::cout << "Data before sorting: ";
        print_vector(data);
    }

    int local_size = DATA_SIZE / world_size;
    local_data.resize(local_size);

    int error_code = MPI_Scatter(data.data(), local_size, MPI_INT, local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (error_code != MPI_SUCCESS) {
        std::cerr << "Error during MPI_Scatter: " << MPI_Error_string(error_code) << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::sort(local_data.begin(), local_data.end());

    std::vector<int> samples(world_size);
    for (int i = 0; i < world_size; ++i) {
        samples[i] = local_data[i * local_size / world_size];
    }

    std::vector<int> gathered_samples;
    if (world_rank == 0) {
        gathered_samples.resize(world_size * world_size);
    }

    error_code = MPI_Gather(samples.data(), world_size, MPI_INT, gathered_samples.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (error_code != MPI_SUCCESS) {
        std::cerr << "Error during MPI_Gather: " << MPI_Error_string(error_code) << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<int> pivots(world_size - 1);
    if (world_rank == 0) {
        std::sort(gathered_samples.begin(), gathered_samples.end());
        for (int i = 0; i < world_size - 1; ++i) {
            pivots[i] = gathered_samples[(i + 1) * world_size];
        }
    }

    error_code = MPI_Bcast(pivots.data(), world_size - 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (error_code != MPI_SUCCESS) {
        std::cerr << "Error during MPI_Bcast: " << MPI_Error_string(error_code) << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<int> send_counts(world_size, 0);

