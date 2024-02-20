#pragma once

#include <mpi.h>

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_topology/topology.hpp>
#include <vector>

// A Element processor that applies a lambda function
template <size_t N>
class ElementProcessorWithLambdaFunction {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    ElementProcessorWithLambdaFunction(const std::array<DoubleField *, N> fields_to_gather, const DoubleField *field_to_scatter, size_t nodes_per_element, stk::mesh::BulkData *bulk_data)
        : m_fields_to_gather(fields_to_gather), m_field_to_scatter(field_to_scatter), m_num_nodes_per_element(nodes_per_element), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    template <typename Func>
    void for_each_element(const Func &func) {
        // Set up the field data to gather
        std::array<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3>, N> field_data_to_gather;
        for (size_t f = 0; f < N; ++f) {
            field_data_to_gather[f].resize(m_num_nodes_per_element, 3);
        }
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> result;
        result.resize(m_num_nodes_per_element, 3);

        // Loop over all the buckets
        for (stk::mesh::Bucket *bucket : m_bulk_data->buckets(stk::topology::ELEMENT_RANK)) {
            // Loop over the elements
            for (auto &&mesh_element : *bucket) {
                // Get the element's nodes
                stk::mesh::Entity const *element_nodes = m_bulk_data->begin_nodes(mesh_element);

                // Gather the coordinates, displacements, and velocities of the nodes
                for (size_t f = 0; f < N; ++f) {
                    for (size_t i = 0; i < m_num_nodes_per_element; ++i) {
                        double *element_node_data = stk::mesh::field_data(*m_fields_to_gather[f], element_nodes[i]);
                        for (size_t j = 0; j < 3; ++j) {
                            field_data_to_gather[f](i, j) = element_node_data[j];
                        }
                    }
                }
                // Apply the function to the gathered data
                func(field_data_to_gather, result);

                // Scatter the force to the nodes
                for (size_t i = 0; i < m_num_nodes_per_element; ++i) {
                    double *element_node_data = stk::mesh::field_data(*m_field_to_scatter, element_nodes[i]);
                    for (size_t j = 0; j < 3; ++j) {
                        element_node_data[j] += result(i, j);
                    }
                }
            }
        }
    }

   private:
    const std::array<DoubleField *, N> m_fields_to_gather;  ///< The fields to gather
    const DoubleField *m_field_to_scatter;                  ///< The field to scatter
    size_t m_num_nodes_per_element;                         ///< The number of nodes per element
    stk::mesh::BulkData *m_bulk_data;                       ///< The bulk data object.
};

// A Element processor that applies a lambda function
class ElementProcessorWithLambdaFunctionVector {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    ElementProcessorWithLambdaFunctionVector(const std::vector<DoubleField *> fields_to_gather, const DoubleField *field_to_scatter, size_t nodes_per_element, stk::mesh::BulkData *bulk_data)
        : m_fields_to_gather(fields_to_gather), m_field_to_scatter(field_to_scatter), m_num_nodes_per_element(nodes_per_element), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    template <typename Func>
    void for_each_element(const Func &func) {
        // Set up the field data to gather
        size_t N = m_fields_to_gather.size();
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3>> field_data_to_gather(N);
        for (size_t f = 0; f < N; ++f) {
            field_data_to_gather[f].resize(m_num_nodes_per_element, 3);
        }
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> result;
        result.resize(m_num_nodes_per_element, 3);

        // Loop over all the buckets
        for (stk::mesh::Bucket *bucket : m_bulk_data->buckets(stk::topology::ELEMENT_RANK)) {
            // Loop over the elements
            for (auto &&mesh_element : *bucket) {
                // Get the element's nodes
                stk::mesh::Entity const *element_nodes = m_bulk_data->begin_nodes(mesh_element);

                // Gather the coordinates, displacements, and velocities of the nodes
                for (size_t f = 0; f < N; ++f) {
                    for (size_t i = 0; i < m_num_nodes_per_element; ++i) {
                        double *element_node_data = stk::mesh::field_data(*m_fields_to_gather[f], element_nodes[i]);
                        for (size_t j = 0; j < 3; ++j) {
                            field_data_to_gather[f](i, j) = element_node_data[j];
                        }
                    }
                }
                // Apply the function to the gathered data
                func(field_data_to_gather, result);

                // Scatter the force to the nodes
                for (size_t i = 0; i < m_num_nodes_per_element; ++i) {
                    double *element_node_data = stk::mesh::field_data(*m_field_to_scatter, element_nodes[i]);
                    for (size_t j = 0; j < 3; ++j) {
                        element_node_data[j] += result(i, j);
                    }
                }
            }
        }
    }

   private:
    const std::vector<DoubleField *> m_fields_to_gather;  ///< The fields to gather
    const DoubleField *m_field_to_scatter;                ///< The field to scatter
    size_t m_num_nodes_per_element;                       ///< The number of nodes per element
    stk::mesh::BulkData *m_bulk_data;                     ///< The bulk data object.
};

class ElementProcessorWithLambdaFunction3Fields {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    ElementProcessorWithLambdaFunction3Fields(const DoubleField *field0_to_gather, const DoubleField *field1_to_gather, const DoubleField *field2_to_gather, const DoubleField *field_to_scatter, size_t nodes_per_element, stk::mesh::BulkData *bulk_data)
        : m_field0_to_gather(field0_to_gather), m_field1_to_gather(field1_to_gather), m_field2_to_gather(field2_to_gather), m_field_to_scatter(field_to_scatter), m_num_nodes_per_element(nodes_per_element), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    template <typename Func>
    void for_each_element(const Func &func) {
        // Set up the field data to gather
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> field0_data_to_gather;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> field1_data_to_gather;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> field2_data_to_gather;
        field0_data_to_gather.resize(m_num_nodes_per_element, 3);
        field1_data_to_gather.resize(m_num_nodes_per_element, 3);
        field2_data_to_gather.resize(m_num_nodes_per_element, 3);
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor, 8, 3> result;
        result.resize(m_num_nodes_per_element, 3);

        // Loop over all the buckets
        for (stk::mesh::Bucket *bucket : m_bulk_data->buckets(stk::topology::ELEMENT_RANK)) {
            // Loop over the elements
            for (auto &&mesh_element : *bucket) {
                // Get the element's nodes
                stk::mesh::Entity const *element_nodes = m_bulk_data->begin_nodes(mesh_element);

                // Gather the coordinates, displacements, and velocities of the nodes
                for (size_t i = 0; i < m_num_nodes_per_element; ++i) {
                    double *element_node_data0 = stk::mesh::field_data(*m_field0_to_gather, element_nodes[i]);
                    double *element_node_data1 = stk::mesh::field_data(*m_field1_to_gather, element_nodes[i]);
                    double *element_node_data2 = stk::mesh::field_data(*m_field2_to_gather, element_nodes[i]);
                    for (size_t j = 0; j < 3; ++j) {
                        field0_data_to_gather(i, j) = element_node_data0[j];
                        field1_data_to_gather(i, j) = element_node_data1[j];
                        field2_data_to_gather(i, j) = element_node_data2[j];
                    }
                }
                // Apply the function to the gathered data
                func(field0_data_to_gather, field1_data_to_gather, field2_data_to_gather, result);

                // Scatter the force to the nodes
                for (size_t i = 0; i < m_num_nodes_per_element; ++i) {
                    double *element_node_data = stk::mesh::field_data(*m_field_to_scatter, element_nodes[i]);
                    for (size_t j = 0; j < 3; ++j) {
                        element_node_data[j] += result(i, j);
                    }
                }
            }
        }
    }

   private:
    const DoubleField *m_field0_to_gather;  ///< The first field to gather
    const DoubleField *m_field1_to_gather;  ///< The second field to gather
    const DoubleField *m_field2_to_gather;  ///< The third field to gather
    const DoubleField *m_field_to_scatter;  ///< The field to scatter
    size_t m_num_nodes_per_element;         ///< The number of nodes per element
    stk::mesh::BulkData *m_bulk_data;       ///< The bulk data object.
};

class ElementProcessingBenchmarking {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    ElementProcessingBenchmarking(stk::mesh::BulkData *bulk_data, size_t node_per_element) : bulk_data(bulk_data), num_nodes_per_element(node_per_element) {
        meta_data = &bulk_data->mesh_meta_data();
        // Get the coordinate, displacement, velocity, and force fields
        coordinates_field = meta_data->get_field<double>(stk::topology::NODE_RANK, meta_data->coordinate_field_name());
        displacement_field = meta_data->get_field<double>(stk::topology::NODE_RANK, "displacement");
        velocity_field = meta_data->get_field<double>(stk::topology::NODE_RANK, "velocity");
        force_field = meta_data->get_field<double>(stk::topology::NODE_RANK, "force");
        force_direct_field = meta_data->get_field<double>(stk::topology::NODE_RANK, "force_direct");
    }
    void FillFields();
    void CheckForces();

    ~ElementProcessingBenchmarking() {}

    void Run();

   protected:
    void DirectFunction();

    stk::mesh::MetaData *meta_data;
    stk::mesh::BulkData *bulk_data;
    size_t num_nodes_per_element;
    DoubleField *coordinates_field;
    DoubleField *displacement_field;
    DoubleField *velocity_field;
    DoubleField *force_field;
    DoubleField *force_direct_field;
};

void BenchmarkElementProcessing();