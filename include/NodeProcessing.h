#pragma once

#include <mpi.h>

#include <iostream>
#include <memory>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
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

// A Node processor that applies a std::function to each degree of freedom of each node
// Fields are passed in as an array of pointers at construction time
// Function is passed in as at construction time
template <size_t N>
class NodeProcessorWithStandardFunctionAtConstruction {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithStandardFunctionAtConstruction(const std::function<void(size_t, double, std::array<double *, N> &)> func, const std::array<DoubleField *, N> fields, stk::mesh::BulkData *bulk_data)
        : m_func(func), m_fields(fields), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    void for_each_dof(double time_increment) const {
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
                    m_func(iI, time_increment, field_data);        // Call the function
                }
            }
        }
    }

   private:
    const std::function<void(size_t, double, std::array<double *, N> &)> m_func;  // The function to apply
    const std::array<DoubleField *, N> m_fields;                                  // The fields to process
    stk::mesh::BulkData *m_bulk_data;                                             // The bulk data object.
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

// A Node processor that applies a lambda function to each degree of freedom of each node
// Fields are passed in as three pointers at construction time
// Function is passed in as a lambda to the for_each_dof method
class NodeProcessorWithLambdaFunctionSeparateFields {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunctionSeparateFields(const DoubleField *field_0, const DoubleField *field_1, const DoubleField *field_2, stk::mesh::BulkData *bulk_data)
        : m_field_0(field_0), m_field_1(field_1), m_field_2(field_2), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    template <typename Func>
    void for_each_dof(const Func &func) const {
        size_t num_values_per_node = 3;  // Number of values per node

        // Loop over all the buckets
        for (stk::mesh::Bucket *bucket : m_bulk_data->buckets(stk::topology::NODE_RANK)) {
            // Get the field data for the bucket
            double *field_data_0 = stk::mesh::field_data(*m_field_0, *bucket);
            double *field_data_1 = stk::mesh::field_data(*m_field_1, *bucket);
            double *field_data_2 = stk::mesh::field_data(*m_field_2, *bucket);
            // Loop over each node in the bucket
            for (size_t i_node = 0; i_node < bucket->size(); i_node++) {
                // Loop over each component of the node
                for (size_t i = 0; i < num_values_per_node; i++) {
                    size_t iI = i_node * num_values_per_node + i;        // Index into the field data
                    func(iI, field_data_0, field_data_1, field_data_2);  // Call the function
                }
            }
        }
    }

   private:
    const DoubleField *m_field_0;  // The fields to process
    const DoubleField *m_field_1;
    const DoubleField *m_field_2;
    stk::mesh::BulkData *m_bulk_data;  ///< The bulk data object.
};

// A Node processor that applies a function to each degree of freedom of each node
// Fields are passed in as three pointers at construction time
// Function is passed in as a functor with an operator() method at construction time

// Functor class
template <size_t N>
struct NodeProcessorFunctor {
    void operator()(size_t iI, double time_increment, std::array<double *, N> &field_data) const {
        field_data[2][iI] = field_data[0][iI] + time_increment * field_data[1][iI];
    }
};

template <size_t N>
class NodeProcessorWithFunctorClass {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithFunctorClass(const std::array<DoubleField *, N> fields, stk::mesh::BulkData *bulk_data, NodeProcessorFunctor<N> func)
        : m_fields(fields), m_bulk_data(bulk_data), m_func(func) {}

    // Loop over each node and apply the function
    void for_each_dof(double time_increment) const {
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
                    m_func(iI, time_increment, field_data);        // Call the function
                }
            }
        }
    }

   private:
    const std::array<DoubleField *, N> m_fields;  // The fields to process
    stk::mesh::BulkData *m_bulk_data;             // The bulk data object.
    NodeProcessorFunctor<N> m_func;               // The functor to apply
};

// A Node processor that applies a function to each degree of freedom of each node
// Fields are passed in as three pointers at construction time
// Function is passed in as a derived functor with an operator() method at construction time

// Functor base class
template <size_t N>
struct NodeProcessorFunctorBase {
    virtual void operator()(size_t iI, double time_increment, std::array<double *, N> &field_data) = 0;
};

// Functor derived class
struct NodeProcessorFunctorDerived : public NodeProcessorFunctorBase<3> {
    void operator()(size_t iI, double time_increment, std::array<double *, 3> &field_data) override {
        field_data[2][iI] = field_data[0][iI] + time_increment * field_data[1][iI];
    }
};

template <size_t N>
class NodeProcessorWithDerivedFunctorClass {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithDerivedFunctorClass(const std::array<DoubleField *, N> fields, stk::mesh::BulkData *bulk_data, std::shared_ptr<NodeProcessorFunctorBase<N>> func)
        : m_fields(fields), m_bulk_data(bulk_data), m_func(func) {}

    // Loop over each node and apply the function
    void for_each_dof(double time_increment) const {
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
                    size_t iI = i_node * num_values_per_node + i;        // Index into the field data
                    m_func->operator()(iI, time_increment, field_data);  // Call the function
                }
            }
        }
    }

   private:
    const std::array<DoubleField *, N> m_fields;          // The fields to process
    stk::mesh::BulkData *m_bulk_data;                     // The bulk data object.
    std::shared_ptr<NodeProcessorFunctorBase<N>> m_func;  // The functor to apply
};

// A Node processor that applies a lambda function to each degree of freedom of each node
// Fields are passed in as an array of pointers at construction time
// Function is passed in as a lambda without a template to the for_each_dof method
template <size_t N>
class NodeProcessorWithLambdaFunctionNoTemplate {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunctionNoTemplate(const std::array<DoubleField *, N> fields, stk::mesh::BulkData *bulk_data)
        : m_fields(fields), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    void for_each_dof(const std::vector<double> &data, void (*func)(size_t iI, const std::vector<double> &data, std::array<double *, N> &field_data)) const {
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
                    func(iI, data, field_data);                    // Call the function
                }
            }
        }
    }

   private:
    const std::array<DoubleField *, N> m_fields;  // The fields to process
    stk::mesh::BulkData *m_bulk_data;             ///< The bulk data object.
};

// A Node processor that applies a lambda function to each degree of freedom of each node
// Fields are passed in as an vector of pointers at construction time
// Function is passed in as a lambda without a template to the for_each_dof method
class NodeProcessorWithLambdaFunctionNoTemplateVector {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunctionNoTemplateVector(const std::vector<DoubleField *> fields, stk::mesh::BulkData *bulk_data)
        : m_fields(fields), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    void for_each_dof(const std::vector<double> &data, void (*func)(size_t iI, const std::vector<double> &data, std::vector<double *> &field_data)) const {
        size_t num_values_per_node = 3;                     // Number of values per node
        std::vector<double *> field_data(m_fields.size());  // Array to hold field data

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
                    func(iI, data, field_data);                    // Call the function
                }
            }
        }
    }

   private:
    const std::vector<DoubleField *> m_fields;  // The fields to process
    stk::mesh::BulkData *m_bulk_data;           ///< The bulk data object.
};

// A Node processor that applies a lambda function to each degree of freedom of each node
// Fields are passed in as an vector of pointers at construction time
// Function is passed in as a lambda without a template passed in on construction to the for_each_dof method
class NodeProcessorWithLambdaFunctionNoTemplateVectorConstruction {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunctionNoTemplateVectorConstruction(const std::vector<DoubleField *> fields, void (*func)(size_t iI, const std::vector<double> &data, std::vector<double *> &field_data), stk::mesh::BulkData *bulk_data)
        : m_fields(fields), m_func(func), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    void for_each_dof(const std::vector<double> &data) const {
        size_t num_values_per_node = 3;                     // Number of values per node
        std::vector<double *> field_data(m_fields.size());  // Array to hold field data

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
                    m_func(iI, data, field_data);                  // Call the function
                }
            }
        }
    }

   private:
    const std::vector<DoubleField *> m_fields;                                                      // The fields to process
    void (*m_func)(size_t iI, const std::vector<double> &data, std::vector<double *> &field_data);  // The function to apply
    stk::mesh::BulkData *m_bulk_data;                                                               ///< The bulk data object.
};

// A Node processor that applies a lambda function to each degree of freedom of each node
// Fields are passed in as an vector of pointers at construction time
// Function is passed in as a lambda without a template to the for_each_dof method
// It uses a base and derived class to pass the function
class NodeProcessorWithLambdaFunctionNoTemplateVectorBase {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunctionNoTemplateVectorBase() {}

    // Loop over each node and apply the function
    virtual void for_each_dof(const std::vector<double> &data, void (*func)(size_t iI, const std::vector<double> &data, std::vector<double *> &field_data)) const = 0;
};

class NodeProcessorWithLambdaFunctionNoTemplateVectorDerived : public NodeProcessorWithLambdaFunctionNoTemplateVectorBase {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunctionNoTemplateVectorDerived(const std::vector<DoubleField *> fields, stk::mesh::BulkData *bulk_data)
        : m_fields(fields), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    void for_each_dof(const std::vector<double> &data, void (*func)(size_t iI, const std::vector<double> &data, std::vector<double *> &field_data)) const override {
        size_t num_values_per_node = 3;                     // Number of values per node
        std::vector<double *> field_data(m_fields.size());  // Array to hold field data

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
                    func(iI, data, field_data);                    // Call the function
                }
            }
        }
    }

   private:
    const std::vector<DoubleField *> m_fields;  // The fields to process
    stk::mesh::BulkData *m_bulk_data;           ///< The bulk data object.
};

// A Node processor that applies a lambda function to each degree of freedom of each node
// Fields are passed in as an array of pointers at construction time
// Function is passed in as a lambda without a template to the for_each_dof method
// It uses a base and derived class to pass the function
template <size_t N, size_t M>
class NodeProcessorWithLambdaFunctionNoTemplateBase {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunctionNoTemplateBase() {}

    // Loop over each node and apply the function
    virtual void for_each_dof(const std::array<double, M> &data, void (*func)(size_t iI, const std::array<double, M> &data, std::array<double *, N> &field_data)) const = 0;
};

template <size_t N, size_t M>
class NodeProcessorWithLambdaFunctionNoTemplateDerived : public NodeProcessorWithLambdaFunctionNoTemplateBase<N, M> {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    NodeProcessorWithLambdaFunctionNoTemplateDerived(const std::array<DoubleField *, N> fields, stk::mesh::BulkData *bulk_data)
        : m_fields(fields), m_bulk_data(bulk_data) {}

    // Loop over each node and apply the function
    void for_each_dof(const std::array<double, M> &data, void (*func)(size_t iI, const std::array<double, M> &data, std::array<double *, N> &field_data)) const override {
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
                    func(iI, data, field_data);                    // Call the function
                }
            }
        }
    }

   private:
    const std::array<DoubleField *, N> m_fields;  // The fields to process
    stk::mesh::BulkData *m_bulk_data;             ///< The bulk data object.
};

class Benchmarking {
    typedef stk::mesh::Field<double> DoubleField;

   public:
    Benchmarking(stk::mesh::BulkData *bulk_data) : bulk_data(bulk_data) {
        meta_data = &bulk_data->mesh_meta_data();
        // Get the velocity and acceleration fields
        velocity_field = meta_data->get_field<double>(stk::topology::NODE_RANK, "velocity");
        acceleration_field = meta_data->get_field<double>(stk::topology::NODE_RANK, "acceleration");

        // Get the field states
        velocity_field_n = &velocity_field->field_of_state(stk::mesh::StateN);
        velocity_field_np1 = &velocity_field->field_of_state(stk::mesh::StateNP1);
        acceleration_field_n = &acceleration_field->field_of_state(stk::mesh::StateN);
    }
    void FillFields();

    ~Benchmarking() {}

    void Run();

   protected:
    void DirectFunction(double time_step);

    stk::mesh::MetaData *meta_data;
    stk::mesh::BulkData *bulk_data;
    DoubleField *velocity_field;
    DoubleField *acceleration_field;
    DoubleField *velocity_field_n;
    DoubleField *velocity_field_np1;
    DoubleField *acceleration_field_n;
};

void BenchmarkNodeProcessing();