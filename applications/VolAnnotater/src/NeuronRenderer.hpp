#pragma once

#include "Common.hpp"
#include <Model/Mesh.hpp>

class NeuronRenderer : public no_copy_t{
public:
    struct NeuronRendererCreateInfo{

    };
    explicit NeuronRenderer(const NeuronRendererCreateInfo& info);

    ~NeuronRenderer();

    void Reset();

    void AddNeuronMesh(const MeshData0& mesh, PatchID patch_id);

    /**
     * @brief 如果patch_id以及存在 那么会删除原来的数据
     */
    void AddNeuronMesh(const MeshData1& mesh, PatchID patch_id);

    void DeleteNeuronMesh(PatchID patch_id);

    void Begin(const mat4& view, const mat4& proj);

    void Draw(PatchID patch_id);

    void End();

private:
    program_t shader;

    struct alignas(16) Transform{
        mat4 model;
        mat4 proj_view;
    }tf_params;
    std140_uniform_block_buffer_t<Transform> tf_params_buffer;

    struct DrawPatch{
        mat4 model = mat4::identity();
        vertex_array_t vao;
        vertex_buffer_t<Vertex> vbo;
        index_buffer_t<uint32_t> ebo;
    };
    std::unordered_map<PatchID, DrawPatch> patches;

};