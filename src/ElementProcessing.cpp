#include "ElementProcessing.h"

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>

/*
Notes:
- The benchmarking is done for a simple element loop
- The element loop is implemented in different ways to compare the performance.
- Different implementations:
    - Element processor ...
        - Notes


Runtimes:
- Xs. Elapsed time (Element Processor with Lambda Function)

Conclusion:
- TBD
*/

// Benchmarking of different node loops
void BenchmarkElementProcessing() {
    MPI_Comm communicator = MPI_COMM_WORLD;
    std::shared_ptr<stk::mesh::BulkData> p_bulk = stk::mesh::MeshBuilder(communicator).create();
    p_bulk->mesh_meta_data().use_simple_fields();
    stk::mesh::MetaData &meta_data = p_bulk->mesh_meta_data();

    stk::io::StkMeshIoBroker mesh_reader;
    mesh_reader.set_bulk_data(*p_bulk);
    mesh_reader.add_mesh_database("generated:10x10x100:tet", stk::io::READ_MESH);
    mesh_reader.create_input_mesh();
    mesh_reader.add_all_mesh_fields_as_input_fields();

    // Create the fields
    // Displacement field
    stk::mesh::FieldBase &displacement_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "displacement", 3);
    stk::mesh::put_field_on_entire_mesh(displacement_field, 3);
    stk::io::set_field_output_type(displacement_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_role(displacement_field, Ioss::Field::TRANSIENT);

    // Velocity field
    stk::mesh::FieldBase &velocity_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "velocity", 2);
    stk::mesh::put_field_on_entire_mesh(velocity_field, 2);
    stk::io::set_field_output_type(velocity_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_role(velocity_field, Ioss::Field::TRANSIENT);

    // Force field
    stk::mesh::FieldBase &force_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "force", 3);
    stk::mesh::put_field_on_entire_mesh(force_field, 3);
    stk::io::set_field_output_type(force_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_role(force_field, Ioss::Field::TRANSIENT);

    // Force field, for checking the results
    stk::mesh::FieldBase &force_direct_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "force_direct", 3);
    stk::mesh::put_field_on_entire_mesh(force_direct_field, 3);
    stk::io::set_field_output_type(force_direct_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_role(force_direct_field, Ioss::Field::TRANSIENT);

    mesh_reader.populate_bulk_data();

    // Create a benchmarking object
    ElementProcessingBenchmarking benchmarking(p_bulk.get(), 4);

    // Run the benchmarking
    benchmarking.Run();
}

void ElementProcessingBenchmarking::FillFields() {
    size_t num_values_per_node = 3;  // Number of values per node

    // Create a random number generator
    std::default_random_engine generator;

    // Create a uniform distribution
    std::uniform_real_distribution<double> distribution(0.0, 100.0);

    // Loop over all the buckets
    for (stk::mesh::Bucket *bucket : bulk_data->buckets(stk::topology::NODE_RANK)) {
        // Get the field data for the bucket
        double *displacement_data_for_bucket = stk::mesh::field_data(*displacement_field, *bucket);
        double *velocity_data_for_bucket = stk::mesh::field_data(*velocity_field, *bucket);

        for (size_t i_node = 0, e = bucket->size(); i_node < e; i_node++) {
            for (size_t i = 0; i < num_values_per_node; i++) {
                size_t iI = i_node * num_values_per_node + i;
                // Fill with random values
                displacement_data_for_bucket[iI] = distribution(generator);
                velocity_data_for_bucket[iI] = distribution(generator);
            }
        }
    }
}

void SomeElementFunction(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> &node_coordinates,
                         const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> &node_displacements,
                         const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> &node_velocities,
                         Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> &force, bool verbose = false) {
    for (int i = 0; i < node_coordinates.rows(); i++) {
        for (size_t j = 0; j < 3; j++) {
            force(i, j) = node_coordinates(i, j) + node_displacements(i, j) + node_velocities(i, j);
            if (verbose) {
                std::cout << "-------------------\n";
                std::cout << "node_coordinates(" << i << ", " << j << ") = " << node_coordinates(i, j) << std::endl;
                std::cout << "node_displacements(" << i << ", " << j << ") = " << node_displacements(i, j) << std::endl;
                std::cout << "node_velocities(" << i << ", " << j << ") = " << node_velocities(i, j) << std::endl;
                std::cout << "force(" << i << ", " << j << ") = " << force(i, j) << std::endl;
            }
        }
    }
}

void ElementProcessingBenchmarking::DirectFunction() {
    // Get the number of nodes per element and set up the matrices
    // 8 is max number of nodes per element.
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> node_coordinates;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> node_displacements;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> node_velocities;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> force;
    node_coordinates.resize(num_nodes_per_element, 3);
    node_displacements.resize(num_nodes_per_element, 3);
    node_velocities.resize(num_nodes_per_element, 3);
    force.resize(num_nodes_per_element, 3);

    bool verbose = false;

    // Loop over all the buckets
    for (stk::mesh::Bucket *bucket : bulk_data->buckets(stk::topology::ELEMENT_RANK)) {
        // Loop over the elements
        for (auto &&mesh_element : *bucket) {
            // Get the element's nodes
            stk::mesh::Entity const *element_nodes = bulk_data->begin_nodes(mesh_element);

            // Gather the coordinates, displacements, and velocities of the nodes
            for (size_t i = 0; i < num_nodes_per_element; ++i) {
                double *element_node_coordinates = stk::mesh::field_data(*coordinates_field, element_nodes[i]);
                double *element_node_displacements = stk::mesh::field_data(*displacement_field, element_nodes[i]);
                double *element_node_velocities = stk::mesh::field_data(*velocity_field, element_nodes[i]);
                for (size_t j = 0; j < 3; ++j) {
                    node_coordinates(i, j) = element_node_coordinates[j];
                    node_displacements(i, j) = element_node_displacements[j];
                    node_velocities(i, j) = element_node_velocities[j];
                }
            }

            SomeElementFunction(node_coordinates, node_displacements, node_velocities, force, verbose);

            // Scatter the force to the nodes
            for (size_t i = 0; i < num_nodes_per_element; ++i) {
                double *element_node_force = stk::mesh::field_data(*force_direct_field, element_nodes[i]);
                for (size_t j = 0; j < 3; ++j) {
                    element_node_force[j] += force(i, j);
                }
            }
        }
    }
}

void ElementProcessingBenchmarking::CheckForces() {
    double sum_force = 0.0;
    // Loop over all the buckets
    for (stk::mesh::Bucket *bucket : bulk_data->buckets(stk::topology::NODE_RANK)) {
        // Get the field data for the bucket
        double *force_data = stk::mesh::field_data(*force_field, *bucket);
        double *force_direct_data = stk::mesh::field_data(*force_direct_field, *bucket);

        for (size_t i_node = 0, e = bucket->size(); i_node < e; i_node++) {
            for (size_t i = 0; i < 3; i++) {
                size_t iI = i_node * 3 + i;
                if (std::abs((force_data[iI] - force_direct_data[iI])) / force_data[iI] > 1.0e-6) {
                    std::cout << "Error: Forces do not match!" << std::endl;
                    std::cout << "  iI = " << iI << std::endl;
                    std::cout << "  force_data[iI] = " << force_data[iI] << std::endl;
                    std::cout << "  force_direct_data[iI] = " << force_direct_data[iI] << std::endl;
                    return;
                }
                sum_force += force_data[iI];
            }
        }
    }
    if (std::abs(sum_force) < 1.0e-6) {
        std::cout << "Error: Sum of forces is zero!" << std::endl;
        std::cout << "Sum of forces: " << sum_force << std::endl;
    }
}

void ElementProcessingBenchmarking::Run() {
    size_t num_runs = 5000;
    std::cout << std::scientific << std::setprecision(6);  // Set output to scientific notation and 6 digits of precision

    // Fill the fields with random values
    FillFields();

    // ************************************************************************
    // Direct function
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        DirectFunction();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Direct Function)\n";

    // ************************************************************************
    // Element processor with lambda function
    const std::array<DoubleField *, 3> fields_to_gather = {coordinates_field, displacement_field, velocity_field};
    ElementProcessorWithLambdaFunction<3> element_processor_with_lambda_function = ElementProcessorWithLambdaFunction<3>(fields_to_gather, force_field, num_nodes_per_element, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        element_processor_with_lambda_function.for_each_element([](const std::array<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3>, 3> &field_data_to_gather, Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> &result) {
            SomeElementFunction(field_data_to_gather[0], field_data_to_gather[1], field_data_to_gather[2], result);
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function)\n";

    CheckForces();

    // ************************************************************************
    // Element processor with lambda function
    stk::mesh::field_fill(0.0, *force_field);
    const std::vector<DoubleField *> fields_to_gather_vec = {coordinates_field, displacement_field, velocity_field};
    ElementProcessorWithLambdaFunctionVector element_processor_with_lambda_function_vector = ElementProcessorWithLambdaFunctionVector(fields_to_gather_vec, force_field, num_nodes_per_element, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        element_processor_with_lambda_function_vector.for_each_element([](const std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3>> &field_data_to_gather, Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> &result) {
            SomeElementFunction(field_data_to_gather[0], field_data_to_gather[1], field_data_to_gather[2], result);
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function. Vector instead of array)\n";

    CheckForces();
}
