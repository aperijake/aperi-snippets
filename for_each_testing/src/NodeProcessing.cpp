#include "NodeProcessing.h"

#include <iostream>
#include <random>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>

/*
Notes:
- The benchmarking is done for a simple nodal loop that computes the nodal velocities at the next time step.
- The nodal loop is implemented in different ways to compare the performance.
- Different implementations:
    - Direct function: The nodal loop is implemented directly in a function.
        - This is the baseline implementation.
    - Node processor with standard function: The nodal loop is implemented using a node processor with a standard function.
        - Easy and flexible to use, but slow.
    - Node processor with standard function set at construction: The nodal loop is implemented using a node processor with a standard function that is set at construction.
        - Trying to speed up the node processor with a standard function by setting the function at construction.
    - Node processor with lambda function: The nodal loop is implemented using a node processor with a lambda function.
        - Using a template to pass the lambda function to the node processor.
    - Node processor with lambda function with separate fields: The nodal loop is implemented using a node processor with a lambda function and separate fields.
        - Trying to speed up the above implementation by passing the fields separately to the lambda function instead of using an array.
    - Node processor with functor class: The nodal loop is implemented using a node processor with a functor class.
        - Templated virtual functions are not allowed in C++. Trying to use a functor class to get around this.
    - Node processor with derived functor class: The nodal loop is implemented using a node processor with a derived functor class.
        - The above requires hardcoding the exact functor class to use. Trying to use a derived functor class to get around this.
    - Node processor with lambda function without a template: The nodal loop is implemented using a node processor with a lambda function without a template.
        - A bit rigid as it requires a specific function signature. But fast and doesn't need to be templated.
    - Node processor with lambda function without a template and vector for field data: The nodal loop is implemented using a node processor with a lambda function without a template and a vector for field data.
        - Using a vector for field data instead of an array to give more flexibility.
    - Node processor with lambda function without a template and vector for field data and function set at construction: The nodal loop is implemented using a node processor with a lambda function without a template and a vector for field data and the function set at construction.
        - Trying to speed up the above implementation by setting the function at construction. But it actually slows it down.
    - Node processor with lambda function without a template and vector for field data and base and derived classes: The nodal loop is implemented using a node processor with a lambda function without a template and a vector for field data and base and derived classes.
        - Trying the speed out when using base and derived classes.
    - Node processor with lambda function without a template and base and derived classes: The nodal loop is implemented using a node processor with a lambda function without a template and base and derived classes.
        - Seeing if going back to arrays instead of vectors for field data makes a difference.


Runtimes:
- 1.032539e+00s. Elapsed time (Direct Function)
- 4.625675e+00s. Elapsed time (Node Processor with Standard Function)
- 4.265332e+00s. Elapsed time (Node Processor with Standard Function at Construction)
- 9.985610e-01s. Elapsed time (Node Processor with Lambda Function)
- 1.178645e+00s. Elapsed time (Node Processor with Lambda Function with Separate Fields)
- 1.030284e+00s. Elapsed time (Node Processor with Functor Class)
- 2.115628e+00s. Elapsed time (Node Processor with Derived Functor Class)
- 1.006570e+00s. Elapsed time (Node Processor with Lambda Function without a Template)
- 9.962860e-01s. Elapsed time (Node Processor with Lambda Function without a Template and Vector for Field Data)
- 4.251263e+00s. Elapsed time (Node Processor with Lambda Function without a Template and Vector for Field Data and Function Set at Construction)
- 2.074354e+00s. Elapsed time (Node Processor with Lambda Function without a Template and Vector for Field Data and Base and Derived Classes)
- 1.684123e+00s. Elapsed time (Node Processor with Lambda Function without a Template and Base and Derived Classes)

Conclusion:
- Should use Lambda Function without a Template and Vector for Field Data for the best performance.
- Don't use polymorphism for the node processor. It slows it down. Try using macros instead.
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
    stk::mesh::put_field_on_entire_mesh(velocity_field, 2);
    stk::io::set_field_output_type(velocity_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_role(velocity_field, Ioss::Field::TRANSIENT);

    stk::mesh::FieldBase &acceleration_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "acceleration", 2);
    stk::mesh::put_field_on_entire_mesh(acceleration_field, 2);
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
    // Node processor with standard function set at construction
    const std::function<void(size_t, double, std::array<double *, 3> &)> func_node_processor_with_standard_function_at_construction = [&](size_t iI, double time_increment, std::array<double *, 3> &field_data) {
        field_data[2][iI] = field_data[0][iI] + time_increment * field_data[1][iI];
    };
    NodeProcessorWithStandardFunctionAtConstruction<3> node_processor_with_standard_function_at_construction = NodeProcessorWithStandardFunctionAtConstruction<3>(func_node_processor_with_standard_function_at_construction, fields, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_standard_function_at_construction.for_each_dof(time_increment);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Standard Function at Construction)\n";

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
    // Node processor with lambda function with separate fields
    NodeProcessorWithLambdaFunctionSeparateFields node_processor_with_lambda_function_separate_fields = NodeProcessorWithLambdaFunctionSeparateFields(velocity_field_n, acceleration_field_n, velocity_field_np1, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_lambda_function_separate_fields.for_each_dof([&time_increment](size_t iI, double *field_data0, double *field_data1, double *field_data2) {
            field_data2[iI] = field_data0[iI] + time_increment * field_data1[iI];
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function with Separate Fields)\n";

    // ************************************************************************
    // Node processor with functor class
    NodeProcessorFunctor<3> node_processor_functor = NodeProcessorFunctor<3>();
    NodeProcessorWithFunctorClass<3> node_processor_with_functor_class = NodeProcessorWithFunctorClass<3>(fields, bulk_data, node_processor_functor);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_functor_class.for_each_dof(time_increment);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Functor Class)\n";

    // ************************************************************************
    // Node processor with derived functor class

    std::shared_ptr<NodeProcessorFunctorBase<3>> p_node_processor_functor_derived = std::make_shared<NodeProcessorFunctorDerived>();

    NodeProcessorWithDerivedFunctorClass<3> node_processor_with_derived_functor_class = NodeProcessorWithDerivedFunctorClass<3>(fields, bulk_data, p_node_processor_functor_derived);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_derived_functor_class.for_each_dof(time_increment);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Derived Functor Class)\n";

    // ************************************************************************
    // Node processor with lambda function without a template
    NodeProcessorWithLambdaFunctionNoTemplate node_processor_with_lambda_function_no_template = NodeProcessorWithLambdaFunctionNoTemplate(fields, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_lambda_function_no_template.for_each_dof({time_increment}, [](size_t iI, const std::vector<double> &data, std::array<double *, 3> &field_data) {
            field_data[2][iI] = field_data[0][iI] + data[0] * field_data[1][iI];
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function without a Template)\n";

    // ************************************************************************
    // Node processor with lambda function without a template, vector for field data
    std::vector<DoubleField *> fields_no_template = {velocity_field_n, acceleration_field_n, velocity_field_np1};
    NodeProcessorWithLambdaFunctionNoTemplateVector node_processor_with_lambda_function_no_template_vector = NodeProcessorWithLambdaFunctionNoTemplateVector(fields_no_template, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_lambda_function_no_template_vector.for_each_dof({time_increment}, [](size_t iI, const std::vector<double> &data, std::vector<double *> &field_data) {
            field_data[2][iI] = field_data[0][iI] + data[0] * field_data[1][iI];
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function without a Template and Vector for Field Data)\n";

    // ************************************************************************
    // Node processor with lambda function without a template, vector for field data, function set at construction
    void (*func)(size_t iI, const std::vector<double> &data, std::vector<double *> &field_data) = [](size_t iI, const std::vector<double> &data, std::vector<double *> &field_data) {
        field_data[2][iI] = field_data[0][iI] + data[0] * field_data[1][iI];
    };
    NodeProcessorWithLambdaFunctionNoTemplateVectorConstruction node_processor_with_lambda_function_no_template_vector_construction = NodeProcessorWithLambdaFunctionNoTemplateVectorConstruction(fields_no_template, func, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_lambda_function_no_template_vector_construction.for_each_dof({time_increment});
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function without a Template and Vector for Field Data and Function Set at Construction)\n";

    // ************************************************************************
    // Node processor with lambda function without a template, vector for field data, base and derived classes
    std::shared_ptr<NodeProcessorWithLambdaFunctionNoTemplateVectorBase> node_processor_with_lambda_function_no_template_vector_derived = std::make_shared<NodeProcessorWithLambdaFunctionNoTemplateVectorDerived>(fields_no_template, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_lambda_function_no_template_vector_derived->for_each_dof({time_increment}, [](size_t iI, const std::vector<double> &data, std::vector<double *> &field_data) {
            field_data[2][iI] = field_data[0][iI] + data[0] * field_data[1][iI];
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function without a Template and Vector for Field Data and Base and Derived Classes)\n";

    // ************************************************************************
    // Node processor with lambda function without a template, array for field data, base and derived classes
    std::shared_ptr<NodeProcessorWithLambdaFunctionNoTemplateBase<3, 1>> node_processor_with_lambda_function_no_template_derived = std::make_shared<NodeProcessorWithLambdaFunctionNoTemplateDerived<3, 1>>(fields, bulk_data);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_runs; i++) {
        node_processor_with_lambda_function_no_template_derived->for_each_dof({time_increment}, [](size_t iI, const std::array<double, 1> &data, std::array<double *, 3> &field_data) {
            field_data[2][iI] = field_data[0][iI] + data[0] * field_data[1][iI];
        });
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << "s. Elapsed time (Node Processor with Lambda Function without a Template and Base and Derived Classes)\n";
}
