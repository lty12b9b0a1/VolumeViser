#include "NeuronRenderer.hpp"

NeuronRenderer::NeuronRenderer(const NeuronRenderer::NeuronRendererCreateInfo &info) {
    shader = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("glsl/neuron_shading.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("glsl/neuron_shading.frag")
            );

    tf_params_buffer.initialize_handle();
    tf_params_buffer.reinitialize_buffer_data(nullptr, GL_DYNAMIC_DRAW);
}

NeuronRenderer::~NeuronRenderer() {

}

void NeuronRenderer::Reset() {
    patches.clear();
}

void NeuronRenderer::AddNeuronMesh(const MeshData0 &mesh, PatchID patch_id) {

    if(patches.count(patch_id)){
        DeleteNeuronMesh(patch_id);
    }
    assert(patches.count(patch_id) == 0);
    auto& patch = patches[patch_id];
    patch.vao.initialize_handle();
    patch.vbo.initialize_handle();
    patch.vbo.reinitialize_buffer_data(mesh.vertices.data(), mesh.vertices.size(), GL_STATIC_DRAW);
    patch.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0), patch.vbo, &Vertex::pos, 0);
    patch.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(1), patch.vbo, &Vertex::normal, 1);
    patch.vao.enable_attrib(attrib_var_t<vec3f>(0));
    patch.vao.enable_attrib(attrib_var_t<vec3f>(1));
    patch.ebo.initialize_handle();
    patch.ebo.reinitialize_buffer_data(mesh.indices.data(), mesh.indices.size(), GL_STATIC_DRAW);
    patch.vao.bind_index_buffer(patch.ebo);

}

void NeuronRenderer::AddNeuronMesh(const MeshData1 &mesh, PatchID patch_id) {
    AddNeuronMesh(ConvertFrom(mesh), patch_id);
}

void NeuronRenderer::DeleteNeuronMesh(PatchID patch_id) {
    patches.erase(patch_id);
}

void NeuronRenderer::Draw(const mat4 &view, const mat4 &proj) {
    shader.bind();

    tf_params.proj_view = proj * view;
    tf_params_buffer.set_buffer_data(&tf_params);
    tf_params_buffer.bind(0);

    for(auto& [uid, draw_patch] : patches){
        draw_patch.vao.bind();

        GL_EXPR(glDrawElements(GL_TRIANGLES, draw_patch.ebo.index_count(), GL_UNSIGNED_INT, nullptr));

        draw_patch.vao.unbind();
    }

    shader.unbind();
}
