#include "Loop.h"

#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Types.hpp>

typedef stk::mesh::Field<double> DoubleField;
typedef stk::mesh::NgpField<double> NgpDoubleField;

class NodeLoopTestFixture : public ::testing::Test {
   protected:
    void SetUp() override {
    }

    void AddMeshDatabase(size_t num_elements_x, size_t num_elements_y, size_t num_elements_z) {
        MPI_Comm communicator = MPI_COMM_WORLD;
        bulk_data = stk::mesh::MeshBuilder(communicator).create();
        bulk_data->mesh_meta_data().use_simple_fields();
        stk::mesh::MetaData *meta_data = &bulk_data->mesh_meta_data();

        stk::io::StkMeshIoBroker mesh_reader;
        mesh_reader.set_bulk_data(*bulk_data);
        const std::string mesh_spec = "generated:" + std::to_string(num_elements_x) + "x" + std::to_string(num_elements_y) + "x" + std::to_string(num_elements_z);
        expected_num_nodes = (num_elements_x + 1) * (num_elements_y + 1) * (num_elements_z + 1);
        mesh_reader.add_mesh_database(mesh_spec, stk::io::READ_MESH);
        mesh_reader.create_input_mesh();
        mesh_reader.add_all_mesh_fields_as_input_fields();

        // Create the fields
        DoubleField *velocity_field = &meta_data->declare_field<double>(stk::topology::NODE_RANK, "velocity", 2);
        stk::mesh::put_field_on_entire_mesh(*velocity_field, 3);
        stk::io::set_field_output_type(*velocity_field, stk::io::FieldOutputType::VECTOR_3D);
        stk::io::set_field_role(*velocity_field, Ioss::Field::TRANSIENT);

        DoubleField *acceleration_field = &meta_data->declare_field<double>(stk::topology::NODE_RANK, "acceleration", 2);
        stk::mesh::put_field_on_entire_mesh(*acceleration_field, 3);
        stk::io::set_field_output_type(*acceleration_field, stk::io::FieldOutputType::VECTOR_3D);
        stk::io::set_field_role(*acceleration_field, Ioss::Field::TRANSIENT);

        mesh_reader.populate_bulk_data();

        // Get the field states
        velocity_field_n = &velocity_field->field_of_state(stk::mesh::StateN);
        velocity_field_np1 = &velocity_field->field_of_state(stk::mesh::StateNP1);
        acceleration_field_n = &acceleration_field->field_of_state(stk::mesh::StateN);

        // Fill the fields with initial values
        FillFields();

        ngp_velocity_field_n = stk::mesh::get_updated_ngp_field<double>(*velocity_field_n);
        ngp_velocity_field_np1 = stk::mesh::get_updated_ngp_field<double>(*velocity_field_np1);
        ngp_acceleration_field_n = stk::mesh::get_updated_ngp_field<double>(*acceleration_field_n);

        // Get the ngp mesh
        ngp_mesh = &stk::mesh::get_updated_ngp_mesh(*bulk_data);

        // Get the universal part
        universal_part = bulk_data->mesh_meta_data().universal_part();

        // For the NodeProcessorWithStandardFunction
        fields = {velocity_field_n, acceleration_field_n, velocity_field_np1};
        node_processor_with_standard_function = std::make_shared<NodeProcessorWithStandardFunction<3>>(fields, bulk_data.get());
        node_processor_with_lambda_function = std::make_shared<NodeProcessorWithLambdaFunction<3>>(fields, bulk_data.get());
    }

    void FillFields() {
        // Fill the fields with initial values
        stk::mesh::field_fill(initial_velocity, *velocity_field_n);
        stk::mesh::field_fill(0.0, *velocity_field_np1);
        stk::mesh::field_fill(initial_acceleration, *acceleration_field_n);
    }

    void DirectFunction(double time_increment) {
        // Loop over all the buckets
        for (stk::mesh::Bucket *bucket : bulk_data->buckets(stk::topology::NODE_RANK)) {
            // Get the field data for the bucket
            double *velocity_data_n_for_bucket = stk::mesh::field_data(*velocity_field_n, *bucket);
            double *acceleration_data_n_for_bucket = stk::mesh::field_data(*acceleration_field_n, *bucket);
            double *velocity_data_np1_for_bucket = stk::mesh::field_data(*velocity_field_np1, *bucket);

            for (size_t i_node = 0, e = bucket->size(); i_node < e; i_node++) {
                // Compute the first partial update nodal velocities: v^{n+½} = v^n + (t^{n+½} − t^n)a^n
                for (size_t i = 0; i < 3; i++) {
                    size_t iI = i_node * 3 + i;
                    velocity_data_np1_for_bucket[iI] = velocity_data_n_for_bucket[iI] + time_increment * acceleration_data_n_for_bucket[iI];
                }
            }
        }
    }

    void StandardFunction(double time_increment) {
        func_node_processor_with_standard_function = [&](size_t iI, std::array<double *, 3> &field_data) {
            field_data[2][iI] = field_data[0][iI] + time_increment * field_data[1][iI];
        };
        node_processor_with_standard_function->for_each_dof(func_node_processor_with_standard_function);
    }

    void LambdaFunction(double time_increment) {
        node_processor_with_lambda_function->for_each_dof([&time_increment](size_t iI, std::array<double *, 3> &field_data) {
            field_data[2][iI] = field_data[0][iI] + time_increment * field_data[1][iI];
        });
    }

    void StkForEachEntityRun(double time_increment) {
        // Test as if we are running the time integration loop for num_runs iterations
        // Clear the sync state of the fields
        ngp_velocity_field_n.clear_sync_state();
        ngp_acceleration_field_n.clear_sync_state();
        ngp_velocity_field_np1.clear_sync_state();

        stk::mesh::for_each_entity_run(
            *ngp_mesh, stk::topology::NODE_RANK, universal_part,
            KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity) {
                for (size_t j = 0; j < 3; j++) {
                    ngp_velocity_field_np1(entity, j) = ngp_velocity_field_n(entity, j) + time_increment * ngp_acceleration_field_n(entity, j);
                }
            });

        // Set modified on device
        ngp_velocity_field_n.modify_on_device();
        ngp_acceleration_field_n.modify_on_device();
        ngp_velocity_field_np1.modify_on_device();
    }

    template <typename Func>
    double BenchmarkFunction(size_t num_runs, const Func &func) {
        FillFields();
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_runs; i++) {
            func(time_increment);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        CheckFields();
        return elapsed_seconds.count();
    }

    void CheckFields() {
        double expected_velocity_data_np1 = initial_velocity + time_increment * initial_acceleration;
        size_t num_nodes = 0;
        for (stk::mesh::Bucket *bucket : bulk_data->buckets(stk::topology::NODE_RANK)) {
            // Get the field data for the bucket
            double *velocity_data_np1_for_bucket = stk::mesh::field_data(*velocity_field_np1, *bucket);

            for (size_t i_node = 0, e = bucket->size(); i_node < e; i_node++) {
                for (size_t i = 0; i < 3; i++) {
                    size_t iI = i_node * 3 + i;
                    ASSERT_DOUBLE_EQ(velocity_data_np1_for_bucket[iI], expected_velocity_data_np1);
                }
                ++num_nodes;
            }
        }
        EXPECT_EQ(num_nodes, expected_num_nodes);
    }

   protected:
    double time_increment = 0.123;
    double initial_velocity = -2.7;
    double initial_acceleration = 3.1;
    size_t num_elements_x = 10;
    size_t num_elements_y = 10;
    size_t num_elements_z = 10;
    size_t expected_num_nodes;
    std::shared_ptr<stk::mesh::BulkData> bulk_data;
    stk::mesh::NgpMesh *ngp_mesh;
    stk::mesh::Selector universal_part;
    DoubleField *velocity_field_n;
    DoubleField *velocity_field_np1;
    DoubleField *acceleration_field_n;
    NgpDoubleField ngp_velocity_field_n;
    NgpDoubleField ngp_velocity_field_np1;
    NgpDoubleField ngp_acceleration_field_n;
    std::array<DoubleField *, 3> fields;
    std::shared_ptr<NodeProcessorWithStandardFunction<3>> node_processor_with_standard_function;
    std::shared_ptr<NodeProcessorWithLambdaFunction<3>> node_processor_with_lambda_function;
    std::function<void(size_t, std::array<double *, 3> &)> func_node_processor_with_standard_function;
};

// Test directly looping over the nodes
TEST_F(NodeLoopTestFixture, Direct) {
    // Run the benchmarking
    AddMeshDatabase(num_elements_x, num_elements_y, num_elements_z);
    DirectFunction(time_increment);
    CheckFields();
}

// Test using a standard function
TEST_F(NodeLoopTestFixture, StandardFunction) {
    AddMeshDatabase(num_elements_x, num_elements_y, num_elements_z);
    StandardFunction(time_increment);
    CheckFields();
}

// Test using a lambda function
TEST_F(NodeLoopTestFixture, LambdaFunction) {
    AddMeshDatabase(num_elements_x, num_elements_y, num_elements_z);
    LambdaFunction(time_increment);
    CheckFields();
}

// Test using the stk::mesh::for_each_entity_run method
TEST_F(NodeLoopTestFixture, StkMeshForEachEntityRun) {
    AddMeshDatabase(num_elements_x, num_elements_y, num_elements_z);
    StkForEachEntityRun(time_increment);
    CheckFields();
}

// Benchmark the functions
TEST_F(NodeLoopTestFixture, Benchmark) {
    // Run the benchmarking
    AddMeshDatabase(10, 10, 10000);
    size_t num_runs = 1000;
    std::cout << std::scientific << std::setprecision(6);  // Set output to scientific notation and 6 digits of precision

    double time = BenchmarkFunction(num_runs, [&](double time_increment) { DirectFunction(time_increment); });
    std::cout << time << "s. Elapsed time (Direct Function)\n";

    time = BenchmarkFunction(num_runs, [&](double time_increment) { StandardFunction(time_increment); });
    std::cout << time << "s. Elapsed time (Standard Function)\n";

    time = BenchmarkFunction(num_runs, [&](double time_increment) { LambdaFunction(time_increment); });
    std::cout << time << "s. Elapsed time (Lambda Function)\n";

    time = BenchmarkFunction(num_runs, [&](double time_increment) { StkForEachEntityRun(time_increment); });
    std::cout << time << "s. Elapsed time (StkForEachEntityRun)\n";
}
