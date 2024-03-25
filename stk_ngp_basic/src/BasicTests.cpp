#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <iostream>
#include <memory>
#include <stk_io/FillMesh.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/DeviceField.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldState.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/Types.hpp>

TEST(NgpHelloWorld, HelloWorld) {
    // Scope to enforce destruction of Kokkos execution space
    bool is_gpu = Kokkos::DefaultExecutionSpace::concurrency() > 1;

    // Create a Kokkos parallel for loop that runs on the GPU
    if (is_gpu) {
        printf("Cuda Kokkos execution space %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
        Kokkos::parallel_for(
            "gpu_work", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 10), KOKKOS_LAMBDA(int i) {
                printf("Hello from the gpu %d\n", i);
            });
    }

    // Create a Kokkos parallel for loop that runs on the CPU
    printf("Serial Kokkos execution space %s\n", typeid(Kokkos::Serial).name());
    Kokkos::parallel_for(
        "cpu_work", Kokkos::RangePolicy<Kokkos::Serial>(0, 10), KOKKOS_LAMBDA(int i) {
            printf("Hello from the cpu %d\n", i);
        });
}

TEST(NgpField, Fill) {
    if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) {
        return;
    }

    std::shared_ptr<stk::mesh::BulkData> bulk = stk::mesh::MeshBuilder(MPI_COMM_WORLD).create();
    bulk->mesh_meta_data().use_simple_fields();
    stk::mesh::MetaData& meta = bulk->mesh_meta_data();

    stk::mesh::EntityRank rank = stk::topology::ELEMENT_RANK;
    unsigned numStates = 1;

    const int init = 1;
    stk::mesh::Field<int>& field = meta.declare_field<int>(rank, "field_1", numStates);
    stk::mesh::put_field_on_mesh(field, meta.universal_part(), &init);

    stk::io::fill_mesh("generated:1x1x1", *bulk);
    field.sync_to_device();
    stk::mesh::NgpMesh& ngpMesh = stk::mesh::get_updated_ngp_mesh(*bulk);
    stk::mesh::NgpField<int>& ngpField = stk::mesh::get_updated_ngp_field<int>(field);
    int fieldVal = 5;

    stk::mesh::for_each_entity_run(
        ngpMesh, rank, meta.universal_part(), KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity) {
            ngpField(entity, 0) = fieldVal;
        });

    ngpField.modify_on_device();
    ngpField.sync_to_host();

    stk::mesh::EntityId id = 1;
    stk::mesh::Entity entity = bulk->get_entity(rank, id);
    int* data = stk::mesh::field_data(field, entity);

    EXPECT_EQ(data[0], fieldVal);
}