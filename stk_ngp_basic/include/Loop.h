#pragma once

#include <mpi.h>

#include <iostream>
#include <memory>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>
#include <vector>

// Notes on things to test
// Direct call in a function
// Lambda in a function

// A Node processor that applies a std::function to each degree of freedom of each node
// Fields are passed in as an array of pointers at construction time
// Function is passed in to the for_each_dof method
template <size_t N>
class NodeProcessorWithStandardFunction {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithStandardFunction(const std::array<DoubleField *, N> fields, stk::mesh::BulkData *bulk_data)
        : m_fields(fields), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    void for_each_dof(const std::function<void(size_t, std::array<double *, N> &)> &func) const {
        size_t num_values_per_node = 3;      // Number of values per node
        std::array<double *, N> field_data;  // Array to hold field data

        // Loop over all the buckets
        for (stk::mesh::Bucket *bucket : m_bulk_data->buckets(stk::topology::NODE_RANK)) {
            // Get the field data for the bucket
            for (size_t i = 0, e = m_fields.size(); i < e; ++i) {
                field_data[i] = stk::mesh::field_data(*m_fields[i], *bucket);
            }
            // Loop over each node in the bucket
            for (size_t i_node = 0; i_node < bucket->size(); i_node++) {
                // Loop over each component of the node
                for (size_t i = 0; i < num_values_per_node; i++) {
                    size_t iI = i_node * num_values_per_node + i;  // Index into the field data
                    func(iI, field_data);                          // Call the function
                }
            }
        }
    }

   private:
    const std::array<DoubleField *, N> m_fields;  // The fields to process
    stk::mesh::BulkData *m_bulk_data;             // The bulk data object.
};

// A Node processor that applies a lambda function to each degree of freedom of each node
// Fields are passed in as an array of pointers at construction time
// Function is passed in as a lambda to the for_each_dof method
template <size_t N>
class NodeProcessorWithLambdaFunction {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunction(const std::array<DoubleField *, N> fields, stk::mesh::BulkData *bulk_data)
        : m_fields(fields), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    template <typename Func>
    void for_each_dof(const Func &func) const {
        size_t num_values_per_node = 3;      // Number of values per node
        std::array<double *, N> field_data;  // Array to hold field data

        // Loop over all the buckets
        for (stk::mesh::Bucket *bucket : m_bulk_data->buckets(stk::topology::NODE_RANK)) {
            // Get the field data for the bucket
            for (size_t i = 0, e = m_fields.size(); i < e; ++i) {
                field_data[i] = stk::mesh::field_data(*m_fields[i], *bucket);
            }
            // Loop over each node in the bucket
            for (size_t i_node = 0; i_node < bucket->size(); i_node++) {
                // Loop over each component of the node
                for (size_t i = 0; i < num_values_per_node; i++) {
                    size_t iI = i_node * num_values_per_node + i;  // Index into the field data
                    func(iI, field_data);                          // Call the function
                }
            }
        }
    }

   private:
    const std::array<DoubleField *, N> m_fields;  // The fields to process
    stk::mesh::BulkData *m_bulk_data;             ///< The bulk data object.
};

// A Node processor that use the stk::mesh::for_each_entity_run method to apply a lambda function to each degree of freedom of each node
// template <size_t N>
// class NodeProcessorWithStkMeshForEachEntityRun {
//     typedef stk::mesh::NgpField<double> NgpDoubleField;
//
//    public:
//     NodeProcessorWithStkMeshForEachEntityRun(const std::array<NgpDoubleField *, N> fields, stk::mesh::NgpMesh *ngp_mesh, stk::mesh::Selector selector)
//         : m_fields(fields), m_ngp_mesh(ngp_mesh), m_selector(selector) {}
//
//     // Loop over each node and apply the function
//     template <typename Func>
//     void for_each_dof(const Func &func) const {
//         size_t num_values_per_node = 3;    // Number of values per node
//         std::array<double, N> field_data;  // Array to hold field data
//
//         // Clear the sync state of the fields
//         for (size_t i = 0, e = m_fields.size(); i < e; ++i) {
//             m_fields[i]->clear_sync_state();
//         }
//         stk::mesh::for_each_entity_run(
//             m_ngp_mesh, stk::topology::NODE_RANK, m_selector,
//             KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity) {
//                 for (size_t i = 0; i < num_values_per_node; i++) {
//                     for (size_t j = 0, e = m_fields.size(); j < e; ++j) {
//                         field_data[j] = m_fields[j](entity, i);
//                     }
//                     func(field_data);
//                 }
//             });
//
//         // Set modified on device
//         for (size_t i = 0, e = m_fields.size(); i < e; ++i) {
//             m_fields[i]->modify_on_device();
//         }
//     }
//
//    private:
//     const std::array<NgpDoubleField *, N> m_fields;  // The fields to process
//     stk::mesh::NgpMesh *m_ngp_mesh;                  ///< The ngp mesh object.
//     stk::mesh::Selector m_selector;                  ///< The selector for the mesh
// };

class NodeProcessingBenchmarking {
    typedef stk::mesh::Field<double> DoubleField;
    typedef stk::mesh::NgpField<double> NgpDoubleField;

   public:
    NodeProcessingBenchmarking(stk::mesh::BulkData *bulk_data) : bulk_data(bulk_data) {
        meta_data = &bulk_data->mesh_meta_data();
        // Get the velocity and acceleration fields
        velocity_field = meta_data->get_field<double>(stk::topology::NODE_RANK, "velocity");
        acceleration_field = meta_data->get_field<double>(stk::topology::NODE_RANK, "acceleration");

        ngp_velocity_field = &stk::mesh::get_updated_ngp_field<double>(*velocity_field);
        ngp_acceleration_field = &stk::mesh::get_updated_ngp_field<double>(*acceleration_field);

        // Get the field states
        velocity_field_n = &velocity_field->field_of_state(stk::mesh::StateN);
        velocity_field_np1 = &velocity_field->field_of_state(stk::mesh::StateNP1);
        acceleration_field_n = &acceleration_field->field_of_state(stk::mesh::StateN);

        ngp_velocity_field_n = &stk::mesh::get_updated_ngp_field<double>(*velocity_field_n);
        ngp_velocity_field_np1 = &stk::mesh::get_updated_ngp_field<double>(*velocity_field_np1);
        ngp_acceleration_field_n = &stk::mesh::get_updated_ngp_field<double>(*acceleration_field_n);

        // Get the ngp mesh
        ngp_mesh = &stk::mesh::get_updated_ngp_mesh(*bulk_data);

        // Get the universal part
        universal_part = bulk_data->mesh_meta_data().universal_part();
    }
    void FillFields();

    ~NodeProcessingBenchmarking() {}

    void Run();

   protected:
    void DirectFunction(double time_step);

    stk::mesh::MetaData *meta_data;
    stk::mesh::BulkData *bulk_data;
    stk::mesh::NgpMesh *ngp_mesh;
    stk::mesh::Selector universal_part;
    DoubleField *velocity_field;
    DoubleField *acceleration_field;
    DoubleField *velocity_field_n;
    DoubleField *velocity_field_np1;
    DoubleField *acceleration_field_n;
    NgpDoubleField *ngp_velocity_field;
    NgpDoubleField *ngp_acceleration_field;
    NgpDoubleField *ngp_velocity_field_n;
    NgpDoubleField *ngp_velocity_field_np1;
    NgpDoubleField *ngp_acceleration_field_n;
};

void BenchmarkNodeProcessing();