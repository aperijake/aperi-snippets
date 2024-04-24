#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <filesystem>
#include <iomanip>
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

void run_stk_for_each_entity_state_update(double time_increment, stk::mesh::NgpMesh *ngp_mesh, stk::mesh::Selector universal_part, NgpDoubleField *ngp_velocity_field_n, NgpDoubleField *ngp_velocity_field_np1) {
    // Clear the sync state of the fields
    ngp_velocity_field_n->clear_sync_state();
    ngp_velocity_field_np1->clear_sync_state();

    NgpDoubleField velocity_field_n = *ngp_velocity_field_n;
    NgpDoubleField velocity_field_np1 = *ngp_velocity_field_np1;

    stk::mesh::for_each_entity_run(
        *ngp_mesh, stk::topology::NODE_RANK, universal_part,
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity) {
            for (size_t j = 0; j < 3; j++) {
                // Print the component and values
                // double value_n = velocity_field_n(entity, j);
                // double value_np1 = velocity_field_np1(entity, j);
                velocity_field_np1(entity, j) = velocity_field_n(entity, j) + time_increment;
                // if (j == 0){
                //     printf("Value n: %f -> %f;  Value np1: %f -> %f\n", value_n, velocity_field_n(entity, j), value_np1, velocity_field_np1(entity, j));
                // }
            }
        });

    // Set modified on device
    ngp_velocity_field_n->modify_on_device();
    ngp_velocity_field_np1->modify_on_device();
}

class NodeStateUpdateTestFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        on_gpu = Kokkos::DefaultExecutionSpace::concurrency() > 1 ? true : false;
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

        mesh_reader.populate_bulk_data();

        // Get the field states
        velocity_field_n = &velocity_field->field_of_state(stk::mesh::StateN);
        velocity_field_np1 = &velocity_field->field_of_state(stk::mesh::StateNP1);

        // Get the ngp mesh
        ngp_mesh = &stk::mesh::get_updated_ngp_mesh(*bulk_data);

        // Get the ngp fields
        ngp_velocity_field_n = &stk::mesh::get_updated_ngp_field<double>(*velocity_field_n);
        ngp_velocity_field_np1 = &stk::mesh::get_updated_ngp_field<double>(*velocity_field_np1);

        // Fill the fields with initial values
        FillFields();

        // Get the universal part
        universal_part = bulk_data->mesh_meta_data().universal_part();
    }

    void FillFields() {
        // Clear the sync state of the fields
        ngp_velocity_field_n->clear_sync_state();
        ngp_velocity_field_np1->clear_sync_state();

        // Fill the fields with initial values, on host
        stk::mesh::field_fill(initial_velocity, *velocity_field_n);
        stk::mesh::field_fill(0.0, *velocity_field_np1);

        // Set modified on host
        ngp_velocity_field_n->modify_on_host();
        ngp_velocity_field_np1->modify_on_host();

        // Sync to device
        ngp_velocity_field_n->sync_to_device();
        ngp_velocity_field_np1->sync_to_device();
    }

    void UpdateStates(bool rotate_device_states) {
        bulk_data->update_field_data_states(rotate_device_states);
    }

    void StkForEachEntityRun(double time_increment, bool update_device_states = false) {
        EXPECT_TRUE(final_time / time_increment > 2);
        double current_time = 0.0;
        while (current_time <= final_time) {
            // std::cout << "Current time: " << current_time << "\n";
            run_stk_for_each_entity_state_update(time_increment, ngp_mesh, universal_part, ngp_velocity_field_n, ngp_velocity_field_np1);
            UpdateStates(update_device_states);
            current_time += time_increment;
        }
    }

    void CheckFields(double expected_velocity_data_np1 = 0.0) {
        ngp_velocity_field_np1->sync_to_host();
        ngp_velocity_field_n->sync_to_host();
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

    bool on_gpu;
    double time_increment = 1.0;
    double final_time = 10.0;
    double initial_velocity = 0.0;
    size_t num_elements_x = 1;
    size_t num_elements_y = 1;
    size_t num_elements_z = 1;
    size_t expected_num_nodes;
    std::shared_ptr<stk::mesh::BulkData> bulk_data;
    stk::mesh::NgpMesh *ngp_mesh;
    stk::mesh::Selector universal_part;
    DoubleField *velocity_field_n;
    DoubleField *velocity_field_np1;
    NgpDoubleField *ngp_velocity_field_n;
    NgpDoubleField *ngp_velocity_field_np1;
    std::array<DoubleField *, 2> fields;
};

// Device states are not updated
TEST_F(NodeStateUpdateTestFixture, NoDeviceStateUpdate) {
    AddMeshDatabase(num_elements_x, num_elements_y, num_elements_z);
    bool update_device_states = false;
    StkForEachEntityRun(time_increment, update_device_states);
    double expected_velocity_data_np1 = initial_velocity + time_increment;  // only gets updated once, since device states are not updated
    if (!on_gpu) {
        expected_velocity_data_np1 = initial_velocity + final_time;  // gets updated for each iteration, since not on device
    }
    CheckFields(expected_velocity_data_np1);
}

// Device states are updated
TEST_F(NodeStateUpdateTestFixture, UpdateDeviceStates) {
    AddMeshDatabase(num_elements_x, num_elements_y, num_elements_z);
    bool update_device_states = true;
    StkForEachEntityRun(time_increment, update_device_states);
    double expected_velocity_data_np1 = initial_velocity + final_time;  // gets updated for each iteration, since device states are updated, regardless of device or host
    CheckFields(expected_velocity_data_np1);
}