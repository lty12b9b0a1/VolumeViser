#include "Common.hpp"
#include "imfilebrowser.h"

class VolAnnotaterGUI : public gl_app_t{
public:
    using gl_app_t::gl_app_t;

    void Initialize();

private:

    void initialize() override;

    void frame() override;

    void destroy() override;

    void handle_events() override;
private:
    //渲染开始前的各种准备 包括各种状态的重置
    void pre_render();

    void update_per_frame();

    void show_editor_ui();

    void show_editor_menu(bool* p_open);

    void show_editor_vol_info_window(bool* p_open);

    void show_editor_vol_render_info_window(bool* p_open);

    void show_editor_vol_render_window(bool* p_open);

    void show_editor_mesh_render_info_window(bool* p_open);

    void show_editor_mesh_render_window(bool* p_open);

    void show_editor_swc_window(bool* p_open);

    void show_editor_swc_op_window(bool* p_open);

    void show_editor_swc_tree_window(bool* p_open);

    void show_editor_neuron_mesh_window(bool *p_open);

    void show_smooth_mesh_window(bool* p_open);

    void show_swc_load_window(bool* p_open);

    void show_debug_window(bool *p_open);
private:
    void check_window_resize_event();

    void on_vol_render_window_resize();

    void on_mesh_render_window_resize();

    void on_swc_view_window_resize();
private:
    void load_volume(const std::string& filename);

    void update_vol_camera_setting(bool init);

private:
    // framebuffer 大小改变时需要调用
    void register_cuda_gl_interop_resource();

    void create_vol_render_gl_framebuffer_resource();

    void create_mesh_render_gl_framebuffer_resource();

private:
    void init_ui_func();

    void frame_vol_render();

    // 大规模体绘制一帧
    void render_volume();

    void update_vol_render_per_frame_params(PerFrameParams& params);

    // 负责查询鼠标点击处是否有对应的swc point，如果有则记录并返回true
    bool pick_swc_point();

    //更新一些状态量，以及被选中后的渲染颜色
    void process_picked_swc_point();

    // 点查询 与体绘制(render_volume)串行
    bool query_volume();

    void render_swc();

    void frame_mesh_render();

    bool can_start_annotating();

    bool is_annotating();

    void check_and_add_swc_pt();

    void update_swc_influenced_blocks(bool all = false);

    bool check_and_start_annotating();

    bool stop_and_save_annotating();

    void generate_modified_mesh();

//    void frame_swc_view_render();

private:
    std::unique_ptr<ViserRescPack> viser_resc;

    std::unique_ptr<VolRenderRescPack> vol_render_resc;

    std::unique_ptr<SWCRescPack> swc_resc;

    std::unique_ptr<SWC2MeshRescPack> swc2mesh_resc;

    // 使得整个程序的fps保持稳定的60帧 其它的绘制任务以异步形式执行 即如果没有在规定时间内完成 保持上一帧结果
    vutil::thread_group_t render_group;

    // 将swc转mesh任务以异步形式启动 不影响到ui绘制线程
    vutil::thread_group_t swc2mesh_group;

    ImGui::FileBrowser vol_file_dialog;
    ImGui::FileBrowser swc_file_dialog;
    ImGui::FileBrowser mesh_file_dialog;

    struct{
        vec2i vol_render_window_size = {0, 0};
        vec2i vol_render_window_pos = {0, 0};
        bool vol_render_resize = false;
        vec2i mesh_render_window_size = {0, 0};
        vec2i mesh_render_window_pos = {0, 0};
        bool mesh_render_resize = false;
        vec2i swc_view_window_size = {0, 0};
        vec2i swc_view_window_pos = {0, 0};
        bool swc_view_resize = false;
        bool vol_mesh_render_hovered = false;
    }window_priv_data;

    enum Status : size_t{
        VOL_DATA_LOADED = 1ull,
        VOL_CAMERA_CHANGED = 1ull << 1,
        VOL_ANNOTATING = 1ull << 2,
        VOL_DRAW_VOLUME = 1ull << 3,
        VOL_DRAW_SWC = 1ull << 4,
        VOL_SWC_VOLUME_BLEND_WITH_DEPTH = 1ull << 5,
        VOL_DRAW_SWC_TAG = 1ull << 6,
        VOL_RENDER_PARAMS_CHANGED = 1ull << 7
    };

    size_t status_flags = 0;


    enum SWCOpStatus{
        SWC_OP_Point,
        SWC_OP_Segment
    };
    enum SWCOps : int{
        SWC_OP_Point_UpdateR,
        SWC_OP_Point_Delete,
        SWC_OP_Seg_InterpR,
        SWC_OP_Seg_Delete,
        SWC_OP_Seg_Add
    };
    SWCOpStatus swc_op = SWC_OP_Point;

    std::unordered_map<SWCOps, std::function<void()>> swc_op_ui;

    //体绘制 用于opengl cuda资源交互相关
    struct{
        cudaGraphicsResource_t cuda_frame_color_resc = nullptr;
        cudaGraphicsResource_t cuda_frame_depth_resc = nullptr;
        pixel_unpack_buffer color_pbo;
        pixel_unpack_buffer depth_pbo;
    }vol_render_cudaGL_interop;

    // 体绘制/swc线绘制的OpenGL framebuffer资源
    struct{
        int frame_width = 0, frame_height = 0;
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t color;
        texture2d_t depth;
        texture2d_t hash_id;
        vutil::image2d_t<int> m_hash_id;
    }vol_swc_render_framebuffer;


    struct{
        float lambda = 0.3;
        float mu = -0.2;
        int iterations = 100;
    }mesh_smooth_priv_data;

    // 神经元网格绘制的OpenGL framebuffer资源
    struct{
        int frame_width = 0, frame_height = 0;
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t color;
    }mesh_render_framebuffer;

    struct{
        float light_h_degree = 90.f;
        float light_v_degree = 0.f;
        Float3 light_color = Float3(1.f);
        float light_intensity = 1.f;
    };

    vertex_array_t quad_vao;

    // view to proj depth
    struct{
        program_t v2p_shader;
        struct alignas(16) V2PParams{
            mat4 proj;
            float fov; // rad
            float w_over_h;
        }v2p_params;
        std140_uniform_block_buffer_t<V2PParams> v2p_params_buffer;
    }v2p_priv_data;

    struct{
        bool mesh_render_line_mode = false;
    }debug_priv_data;

    Timer vol_render_timer;

    struct{
        bool vol_mesh_render_sync = true;

        float vol_camera_move_speed = 0.f;

        float vol_camera_view_ratation_speed = 0.f;

        bool vol_swc_blend_with_depth = false;

        bool vol_render_swc = true;

        bool vol_render_swc_point_tag = false;

        bool vol_render_volume = true;

        bool vol_render_save_frame = false;
    };

    struct{
        bool editor_menu_window_open = true;
        bool vol_info_window_open = true;
        bool vol_render_info_window_open = true;
        bool vol_render_window_open = true;
        bool mesh_render_info_window_open = true;
        bool mesh_render_window_open = true;
        bool swc_info_window_open = true;
        bool swc_op_window_open = true;
        bool swc_tree_window_open = true;
        bool neuron_mesh_window_open = true;
        bool smooth_mesh_window_open = false;
        bool swc_load_window_open = false;
    };
};