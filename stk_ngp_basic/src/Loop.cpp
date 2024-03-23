#include "Loop.h"

#include <iostream>
#include <random>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Types.hpp>

/*
Notes:
- The benchmarking is done for a simple nodal loop that computes the nodal velocities at the next time step.
- The nodal loop is implemented in different ways to compare the performance.
- Different implementations:
    - Direct function: The nodal loop is implemented directly in a function.
        - This is the baseline implementation.
    - Node processor with standard function: The nodal loop is implemented using a node processor with a standard function.
        - Easy and flexible to use, but slow.
    - Node processor with lambda function: The nodal loop is implemented using a node processor with a lambda function.
        - Using a template to pass the lambda function to the node processor.

Runtimes:
- 1.032539e+00s. Elapsed time (Direct Function)
- 4.625675e+00s. Elapsed time (Node Processor with Standard Function)
- 9.985610e-01s. Elapsed time (Node Processor with Lambda Function)
*/

// Benchmarking of different node loops
void BenchmarkNodeProcessing() {
    MPI_Comm communicator = MPI_COMM_WORLD;
    std::shared_ptr<stk::mesh::BulkData> p_bulk = stk::mesh::MeshBuilder(communicator).create();
    p_bulk->mesh_meta_data().use_simple_fields();
    stk::mesh::MetaData &meta_data = p_bulk->mesh_meta_data();

    stk::io::StkMeshIoBroker mesh_reader;
    mesh_reader.set_bulk_data(*p_bulk);
    mesh_reader.add_mesh_database("generated:10x10x100", stk::io::READ_MESH);
    mesh_reader.create_input_mesh();
    mesh_reader.add_all_mesh_fields_as_input_fields();

    // Create the fields
    stk::mesh::FieldBase &velocity_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "velocity", 2);
    stk::mesh::put_field_on_entire_mesh(velocity_field, 3);
    stk::io::set_field_output_type(velocity_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_role(velocity_field, Ioss::Field::TRANSIENT);

    stk::mesh::FieldBase &acceleration_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "acceleration", 2);
    stk::mesh::put_field_on_entire_mesh(acceleration_field, 3);
    stk::io::set_field_output_type(acceleration_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_role(acceleration_field, Ioss::Field::TRANSIENT);

    mesh_reader.populate_bulk_data();

    // Create a benchmarking object
    NodeProcessingBenchmarking benchmarking(p_bulk.get());

    // Run the benchmarking
    benchmarking.Run();
}

void NodeProcessingBenchmarking::FillFields() {
    size_t num_values_per_node = 3;  // Number of values per node

    // Create a random number generator
    std::default_random_engine generator;

    // Create a uniform distribution
    std::uniform_real_distribution<double> distribution(0.0, 100.0);

    // Loop over all the buckets
    for (stk::mesh::Bucket *bucket : bulk_data->buckets(stk::topology::NODE_RANK)) {
        // Get the field data for the bucket
        double *velocity_data_n_for_bucket = stk::mesh::field_data(*velocity_field_n, *bucket);
        double *acceleration_data_n_for_bucket = stk::mesh::field_data(*acceleration_field_n, *bucket);
        double *velocity_data_np1_for_bucket = stk::mesh::field_data(*velocity_field_np1, *bucket);

        for (size_t i_node = 0, e = bucket->size(); i_node < e; i_node++) {
            for (size_t i = 0; i < num_values_per_node; i++) {
                size_t iI = i_node * num_values_per_node + i;
                // Fill with random values
                velocity_data_n_for_bucket[iI] = distribution(generator);
                acceleration_data_n_for_bucket[iI] = distribution(generator);
                velocity_data_np1_for_bucket[iI] = distribution(generator);
            }
        }
    }
}

void NodeProcessingBenchmarking::DirectFunction(double time_increment) {
    size_t num_values_per_node = 3;  // Number of values per node

    // Loop over all the buckets
    for (stk::mesh::Bucket *bucket : bulk_data->buckets(stk::topology::NODE_RANK)) {
        // Get the field data for the bucket
        double *velocity_data_n_for_bucket = stk::mesh::field_data(*velocity_field_n, *bucket);
        double *acceleration_data_n_for_bucket = stk::mesh::field_data(*acceleration_field_n, *bucket);
        double *velocity_data_np1_for_bucket = stk::mesh::field_data(*velocity_field_np1, *bucket);

        for (size_t i_node = 0, e = bucket->size(); i_node < e; i_node++) {
            // Compute the first partial update nodal velocities: v^{n+½} = v^n + (t^{n+½} − t^n)a^n
            for (size_t i = 0; i < num_values_per_node; i++) {
                size_t iI = i_node * num_values_per_node + i;
                velocity_data_np1_for_bucket[iI] = velocity_data_n_for_bucket[iI] + time_increment * acceleration_data_n_for_bucket[iI];
            }
        }
    }
}

void NodeProcessingBenchmarking::Run() {
    double time_increment = 0.123;
    size_t num_runs = 100000;
    std::cout << std::scientific << std::setprecision(6);  // Set output to scientific notation and 6 digits of precision

    // Fill the fields with random values
    FillFields();

    // ************************************************************************
    // Direct function
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        DirectFunction(time_increment);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Direct Function)\n";

    // ************************************************************************
    // Node processor with standard function
    const std::array<DoubleField *, 3> fields = {velocity_field_n, acceleration_field_n, velocity_field_np1};
    NodeProcessorWithStandardFunction<3> node_processor_with_standard_function = NodeProcessorWithStandardFunction<3>(fields, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    std::function<void(size_t, std::array<double *, 3> &)> func_node_processor_with_standard_function;
    for (size_t i = 0; i < num_runs; i++) {
        func_node_processor_with_standard_function = [&](size_t iI, std::array<double *, 3> &field_data) {
            field_data[2][iI] = field_data[0][iI] + time_increment * field_data[1][iI];
        };
        node_processor_with_standard_function.for_each_dof(func_node_processor_with_standard_function);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Standard Function)\n";

    // ************************************************************************
    // Node processor with lambda function
    NodeProcessorWithLambdaFunction<3> node_processor_with_lambda_function = NodeProcessorWithLambdaFunction<3>(fields, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_lambda_function.for_each_dof([&time_increment](size_t iI, std::array<double *, 3> &field_data) {
            field_data[2][iI] = field_data[0][iI] + time_increment * field_data[1][iI];
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function)\n";

    // ************************************************************************
    // Node processor with stk for_each_entity_run
    const std::array<NgpDoubleField *, 3> ngp_fields = {ngp_velocity_field_n, ngp_acceleration_field_n, ngp_velocity_field_np1};
    NodeProcessorWithStkMeshForEachEntityRun<3> node_processor_with_stk_mesh_for_each_entity_run = NodeProcessorWithStkMeshForEachEntityRun<3>(ngp_fields, ngp_mesh, universal_part);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_stk_mesh_for_each_entity_run.for_each_dof([&time_increment](std::array<double, 3> &field_data) {
            field_data[2] = field_data[0] + time_increment * field_data[1];
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with StkMeshForEachEntityRun)\n";
}