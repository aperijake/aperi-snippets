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