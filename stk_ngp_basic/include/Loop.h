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

struct UpdateVelocity2 {
    double time_increment;

    UpdateVelocity2(double time_increment) : time_increment(time_increment) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(double *velocity_data_np1, double *velocity_data_n, double *acceleration_data_n) const {
        *velocity_data_np1 = *velocity_data_n + time_increment * *acceleration_data_n;
    }
};

// A Node processor that uses the stk::mesh::NgpForEachEntity to apply a lambda function to each degree of freedom of each node
template <size_t N>
class NodeProcessorStkNgp {
    typedef stk::mesh::NgpField<double> NgpDoubleField;

   public:
    NodeProcessorStkNgp(const std::shared_ptr<Kokkos::Array<NgpDoubleField, N>> fields,
                        const std::shared_ptr<stk::mesh::NgpMesh> ngp_mesh,
                        const std::shared_ptr<stk::mesh::Selector> selector)
        : m_fields(fields), m_ngp_mesh(ngp_mesh), m_selector(selector) {}

    // Loop over each node and apply the function
    template <typename Func, std::size_t... Is>
    void for_each_dof_impl(const Func &func, std::index_sequence<Is...>) const {
        // Clear the sync state of the fields
        for (size_t i = 0; i < N; i++) {
            (*m_fields.get())[i].clear_sync_state();
        }

        // Declare a device-accessible variable to hold the sum
        // Kokkos::View<int, Kokkos::CudaSpace> d_sum("d_sum");

        //// Initialize the sum to zero
        // Kokkos::deep_copy(d_sum, 0);

        auto &fields = *m_fields;
        // printf("fields.size() = %i\n", (int)fields.size());

        // printf("Calling for_each_entity_run\n");
        stk::mesh::for_each_entity_run(
            *m_ngp_mesh.get(), stk::topology::NODE_RANK, *m_selector.get(),
            KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity) {
                // printf("fields[0]->get(entity, 0) = %f\n", fields[0](entity, 0));
                // printf("fields[1]->get(entity, 0) = %f\n", fields[1](entity, 0));
                // printf("fields[2]->get(entity, 0) = %f\n", fields[2](entity, 0));
                for (size_t j = 0; j < 3; j++) {
                    // func(&fields[0](entity, j), &fields[1](entity, j), &fields[2](entity, j));
                    func(&fields[Is](entity, j)...);
                }
                // printf("after fields[0]->get(entity, 0) = %f\n", fields[0](entity, 0));
                // printf("after fields[1]->get(entity, 0) = %f\n", fields[1](entity, 0));
                // printf("after fields[2]->get(entity, 0) = %f\n", fields[2](entity, 0));

                // // Increment the sum
                // Kokkos::atomic_fetch_add(&d_sum(), 1);
            });

        // Copy the sum back to the host
        // int h_sum = 0;
        // Kokkos::deep_copy(h_sum, d_sum);
        // printf("h_sum = %i\n", h_sum);

        // Modify the fields on the device
        for (size_t i = 0; i < N; i++) {
            (*m_fields.get())[i].modify_on_device();
        }
    }

    template <typename Func>
    void for_each_dof(const Func &func) const {
        for_each_dof_impl(func, std::make_index_sequence<N>{});
    }

   private:
    const std::shared_ptr<Kokkos::Array<NgpDoubleField, N>> m_fields;  // The fields to process
    const std::shared_ptr<stk::mesh::NgpMesh> m_ngp_mesh;              // The ngp mesh object.
    const std::shared_ptr<stk::mesh::Selector> m_selector;             // The selector for the nodes
};