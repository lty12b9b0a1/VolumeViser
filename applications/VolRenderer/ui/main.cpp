//
// Created by wyz on 2023/4/10.
//

//只用于渲染单帧 不能调整任何参数 除了相机移动 可用于debug

#include "../src/Common.hpp"
#include "Algorithm/LevelOfDetailPolicy.hpp"

#include <cuda_gl_interop.h>

#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

#define TEST_TIME(f, desc) test_time([this](){f;}, desc);
#define PI_F												3.141592654f


class UI : public gl_app_t{
  public:
    using gl_app_t::gl_app_t;

    UI& init(int argc, char** argv){
       // todo resize window for frame size in config file

       cmdline::parser cmd;

       cmd.add<std::string>("config-file", 'c', "config json filename");

       cmd.parse_check(argc, argv);

       auto filename = cmd.get<std::string>("config-file");

       SET_LOG_LEVEL_DEBUG

       LoadFromJsonFile(params, filename);

        return *this;
    }

  private:
    bool render_volume_control = true;

    void initialize() override {
//        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glDisable(GL_DEPTH_TEST));
        int w = FRAME_WIDTH;
        int h = FRAME_HEIGHT;

        //
        fbo.initialize_handle();
        rbo.initialize_handle();
        rbo.set_format(GL_DEPTH24_STENCIL8, w, h);
        fbo.attach(GL_DEPTH_STENCIL_ATTACHMENT, rbo);
        frame_color.initialize_handle();
        frame_color.initialize_texture(1, GL_RGBA32F, w, h);
        accumulate_color.initialize_handle();
        accumulate_color.initialize_texture(1, GL_RGBA32F, w, h);
        display_color.initialize_handle();
        display_color.initialize_texture(1, GL_RGBA8, w, h);

        accumulator = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/accumulate.comp")
            );

        post_processor = program_t::build_from(
            shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/post_process.comp")
            );

        screen = program_t::build_from(
            shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/quad.vert"),
            shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/quad.frag")
            );

        accu_params_buffer.initialize_handle();
        accu_params_buffer.reinitialize_buffer_data(&accu_params, GL_STATIC_DRAW);
        post_params_buffer.initialize_handle();
        post_params_buffer.reinitialize_buffer_data(&post_params, GL_STATIC_DRAW);

        quad_vao.initialize_handle();

        framebuffer = NewHandle<FrameBuffer>(ResourceType::Object);
        framebuffer->frame_width = w;
        framebuffer->frame_height = h;


        // renderer resource
        auto& resc_ins = ResourceMgr::GetInstance();
        auto host_mem_mgr_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Host,
                                                              .MaxMemBytes = params.memory.max_host_mem_bytes,
                                                              .DeviceIndex = -1});
        auto host_mem_mgr_ref = resc_ins.GetHostRef(host_mem_mgr_uid);

        auto gpu_resc_uid = resc_ins.RegisterResourceMgr({.type = ResourceMgr::Device,
                                                          .MaxMemBytes = params.memory.max_gpu_mem_bytes,
                                                          .DeviceIndex = (int)params.memory.gpu_index});
        auto gpu_mem_mgr_ref = resc_ins.GetGPURef(gpu_resc_uid);

        gpu_mem_mgr_ref->_get_cuda_context()->set_ctx();

        PBVolumeRenderer::PBVolumeRendererCreateInfo pb_info{
            .host_mem_mgr_ref = host_mem_mgr_ref, //todo: just single thread ?
            .gpu_mem_mgr_ref = gpu_mem_mgr_ref
        };

        pb_vol_renderer = NewHandle<PBVolumeRenderer>(ResourceType::Object, pb_info);

        GridVolume::GridVolumeCreateInfo vol_info;
        SetupVolumeIO(vol_info, params.data);
        auto volume = NewHandle<GridVolume>(ResourceType::Object, vol_info);

        volume_desc = volume->GetDesc();

        pb_vol_renderer->BindGridVolume(std::move(volume));



        auto render_space = (std::min)({volume_desc.voxel_space.x, volume_desc.voxel_space.y, volume_desc.voxel_space.z});

        {
//            {};

            render_params.light.updated = true;
            render_params.light.lightnum = 2;

            render_params.light.lights[0].m_T = 0;
            render_params.light.lights[0].m_T = 0;
            render_params.light.lights[0].m_Theta = PI_F * 0.25f;
            render_params.light.lights[0].m_Phi = PI_F * 0.5f;
            render_params.light.lights[0].m_Width = 0.01f;
            render_params.light.lights[0].m_Height = 0.01f;
            render_params.light.lights[0].m_Distance = 6.f;
            render_params.light.lights[0].m_Color = make_float3(0.5f, 0.48443f, 0.36765f);
            render_params.light.lights[0].m_intensity = 100.0f;


            render_params.light.lights[1].m_T = 1;
            render_params.light.lights[1].m_intensity = 100.f;
            render_params.light.lights[1].m_ColorTop = make_float3(1.f, 0.f, 0.f);
            render_params.light.lights[1].m_ColorMiddle = make_float3(0.f, 1.f, 0.f);
            render_params.light.lights[1].m_ColorBottom = make_float3(0.f, 0.f, 1.f);





            ComputeUpBoundLOD(render_params.lod.leve_of_dist, render_space,
                              w, h, vutil::deg2rad(40.f));
            render_params.lod.updated = true;
            for(int i = 0; i < vol_info.levels; i++)
                render_params.lod.leve_of_dist.LOD[i] *= 1.5f;

            render_params.tf.updated = true;
            render_params.tf.tf_pts.pts.push_back({119.f/255.f, Float4(0.f, 0.f, 0.f, 0.f)});
            render_params.tf.tf_pts.pts.push_back({142.f/255.f, Float4(0.5f, 0.48443f, 0.36765f, 0.3412f)});
            render_params.tf.tf_pts.pts.push_back({238.f/255.f, Float4(0.853f, 0.338f, 0.092f, 0.73333f)});


            render_params.raycast.updated = true;
            render_params.raycast.ray_step = render_space * 0.5f;
            render_params.raycast.max_ray_dist = 6.f;

            render_params.other.updated = true;
            render_params.other.output_depth = false;

            pb_vol_renderer->SetRenderParams(render_params);

            render_params.light.updated = false;
            render_params.lod.updated = false;
            render_params.tf.updated = false;
            render_params.raycast.updated = false;
            render_params.other.updated = false;
            render_params.other.output_depth = false;

        }

        register_cuda_gl_resource();

        Float3 default_pos = {3.60977, 2.882, 9.3109};//8.06206f
        camera.set_position(default_pos);
        camera.set_perspective(40.f, 0.001f, 10.f);
        camera.set_direction(vutil::deg2rad(-90.f), 0.f);
        camera.set_move_speed(0.005);
        camera.set_view_rotation_speed(0.0003f);
        //global camera for get proj view
        camera.set_w_over_h((float)w / h);
    }

    void frame() override {
        before_render();

        handle_events();

        if(render_volume_control){
            render_scene();
        }

        render_ui();

        after_render();
    }

    void destroy() override {

    }

    void handle_events() override {
        gl_app_t::handle_events();
    }

  private:
    void register_cuda_gl_resource(){
        CUB_CHECK(cudaGraphicsGLRegisterImage(&cuda_frame_color_resc,
                                              frame_color.handle(),
                                              GL_TEXTURE_2D, 0));
    }


    void before_render(){
        framebuffer_t::bind_to_default();
        framebuffer_t::clear_color_depth_buffer();


    }

    void clear_framebuffer(){
        fbo.bind();
        fbo.attach(GL_COLOR_ATTACHMENT0, accumulate_color);
        fbo.attach(GL_COLOR_ATTACHMENT1, display_color);
        GLenum cls[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        GL_EXPR(glDrawBuffers(2, cls));
        fbo.clear_buffer(GL_COLOR_BUFFER_BIT);
        fbo.unbind();

        cur_frame_index = 0;
        accu_params.frame_index = cur_frame_index;
        accu_params_buffer.set_buffer_data(&accu_params);
    }

    void update_per_frame(){
        auto view_not_changed = [&](const PerFrameParams& cur){
            bool pos_same = cur.cam_pos == per_frame_params.cam_pos;
            bool dir_same = cur.cam_dir == per_frame_params.cam_dir;
            bool fov_same = cur.fov     == per_frame_params.fov;
            return pos_same && dir_same && fov_same;
        };

        auto get_cur_params = [&](){
            PerFrameParams cur_params;
            cur_params.fov = vutil::deg2rad(camera.get_fov_deg());
            cur_params.cam_dir = camera.get_xyz_direction();
            cur_params.cam_pos = camera.get_position();
            cur_params.frame_width = FRAME_WIDTH;
            cur_params.frame_height = FRAME_HEIGHT;
            cur_params.frame_w_over_h = camera.get_w_over_h();
            cur_params.proj_view = camera.get_view_proj();
            static Float3 WorldUp = {0.f, 1.f, 0.f};
            cur_params.cam_right = vutil::cross(camera.get_xyz_direction(), WorldUp).normalized();
            cur_params.cam_up = vutil::cross(cur_params.cam_right, cur_params.cam_dir);
            cur_params.debug_mode = per_frame_params.debug_mode;
            return cur_params;
        };

        auto cur_params = get_cur_params();
        auto changed = !view_not_changed(cur_params);
        static bool first = true;
        if(first) changed = first = false;
        if(changed || clear_history){
            clear_framebuffer();
        }
        pb_vol_renderer->SetPerFrameParams(cur_params);

        per_frame_params = cur_params;
    }

    void render_scene(){
        // mapping

        auto w = framebuffer->frame_width;
        auto h = framebuffer->frame_height;

        CUB_CHECK(cudaGraphicsMapResources(1, &cuda_frame_color_resc));
        cudaMipmappedArray_t mip_array;
        CUB_CHECK(cudaGraphicsResourceGetMappedMipmappedArray(&mip_array, cuda_frame_color_resc));
        cudaArray_t array;
        CUB_CHECK(cudaGetMipmappedArrayLevel(&array, mip_array, 0));
        framebuffer->_color = NewHandle<CUDASurface>(ResourceType::Buffer, array);

        update_per_frame();


//        pb_vol_renderer->Render(framebuffer);

        TEST_TIME(pb_vol_renderer->Render(framebuffer), "vol-render");


        cudaGraphicsUnmapResources(1, &cuda_frame_color_resc);

        auto x = (w + 16 - 1) / 16;
        auto y = (h + 16 - 1) / 16;

        fbo.bind();
        accumulator.bind();
        accu_params_buffer.bind(0);
        frame_color.bind_image(0, 0, GL_READ_ONLY, GL_RGBA32F);
        accumulate_color.bind_image(1, 0, GL_READ_WRITE, GL_RGBA32F);
        GL_EXPR(glDispatchCompute(x, y, 1));
        accumulator.unbind();

        GL_EXPR(glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT));

        post_processor.bind();
        post_params_buffer.bind(0);
        accumulate_color.bind_image(0, 0, GL_READ_ONLY, GL_RGBA32F);
        display_color.bind_image(1, 0, GL_WRITE_ONLY, GL_RGBA8);
        GL_EXPR(glDispatchCompute(x, y, 1));
        post_processor.unbind();

        GL_EXPR(glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT));

        framebuffer_t::bind_to_default();
        screen.bind();
        quad_vao.bind();
        display_color.bind(0);
        GL_EXPR(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
        display_color.unbind(0);
        quad_vao.unbind();
        screen.unbind();
    }



    void show_editor_menu() {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground |
            ImGuiConfigFlags_NoMouseCursorChange | ImGuiWindowFlags_NoBringToFrontOnFocus;

        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        bool use_work_area = true;
        ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
        ImGui::SetNextWindowSize(use_work_area ? viewport->WorkSize : viewport->Size);


        bool xx = true;
        bool* p_open = NULL;

        ImGui::Begin("Editor menu", p_open, window_flags);

        ImGuiID main_docking_id = ImGui::GetID("Main Docking");

        // if (ImGui::DockBuilderGetNode(main_docking_id) == nullptr) {

        //     ImGui::DockBuilderRemoveNode(main_docking_id);

        //     ImGui::DockBuilderAddNode(main_docking_id, dock_flags);

        //     ImGui::DockBuilderSetNodePos(main_docking_id,
        //         ImVec2(main_viewport->WorkPos.x, main_viewport->WorkPos.y + 18.0f));
        //     ImGui::DockBuilderSetNodeSize(main_docking_id,
        //         ImVec2(1280, 720 - 18.0f));

        //     ImGui::DockBuilderFinish(main_docking_id);
        // }

        // ImGui::DockSpace(main_docking_id);
        //
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Menu")) {
                if (ImGui::MenuItem("Load Project File")) {

                }
                if (ImGui::MenuItem("New Project File")) {

                }
                if (ImGui::MenuItem("Save Project File")) {

                }
                if (ImGui::MenuItem("Exit")) {

                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Edit")) {
                if (ImGui::MenuItem(" Annotate ", nullptr)) {

                }

                ImGui::EndMenu();
            }

            // if (ImGui::BeginMenu("Window")) {
            //     ImGui::MenuItem("Vol Info", nullptr, &vol_info_window_open);
            //     ImGui::MenuItem("Render Info", nullptr, &vol_render_info_window_open);
            //     ImGui::MenuItem("Vol Render", nullptr, &vol_render_window_open);
            //     ImGui::MenuItem("Mesh Render", nullptr, &mesh_render_window_open);
            //     ImGui::MenuItem("SWC Info", nullptr, &swc_info_window_open);
            //     ImGui::MenuItem("SWC Tree", nullptr, &swc_tree_window_open);
            //     ImGui::MenuItem("Neuron Info", nullptr, &neuron_mesh_window_open);

            //     ImGui::EndMenu();
            // }

            ImGui::EndMenuBar();
        }

        ImGui::End();
    }


    void show_editor_vol_render_info_window() {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove |  ImGuiConfigFlags_NoMouseCursorChange | ImGuiWindowFlags_HorizontalScrollbar;


        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 5 / 6, viewport->WorkPos.y+18));
        ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 1 / 6, viewport->WorkSize.y * 2 / 3));
        bool xx = true;
        bool* p_open = NULL;

        if(p_open && !*p_open) return;

        if(ImGui::Begin("Vol Render Info", p_open, window_flags)){
            // if(ImGui::TreeNode("Camera Setting")){
            // auto dir = camera.get_xyz_direction();
            ImGui::Text("Camera Settings");


            Float3 campos = camera.get_position();
            Float3 camdir = camera.get_xyz_direction();
            Float2 camdirhv = camera.get_direction();
            camdirhv.x = vutil::rad2deg(camdirhv.x);
            camdirhv.y = vutil::rad2deg(camdirhv.y);

            ImGui::BulletText("Camera Dir %.3f %.3f %.3f", &camdir.x, &camdir.y, &camdir.z);
            // auto& pos = camera.get_position();
            Float3 tinput;
            tinput.x = 0;
            tinput.y = 0;
            tinput.z = 0;


            bool update = ImGui::InputFloat3("Camera Pos", &campos.x);
            update |= ImGui::SliderFloat("Camera Dir H", &camdirhv.x, -180.0, 180.0);
            update |= ImGui::SliderFloat("Camera Dir V", &camdirhv.y, -180.0, 180.0);


            float nearz = camera.get_near_z();
            float farz = camera.get_far_z();

            update |= ImGui::InputFloat("Camera Near Plane", &nearz);
            update |= ImGui::InputFloat("Camera Far  Plane", &farz);

            float rotas = camera.get_cursor_speed();
            float moves = camera.get_move_speed();

            update |= ImGui::InputFloat("Move Speed", &moves, 0.1f);
            update |= ImGui::InputFloat("Rotate Speed", &rotas, 0.1f);

            float framew = 640;
            float frameh = 400;
            ImGui::Text("Frame");
            update |= ImGui::InputFloat("Frame Width", &framew, 0.1f);
            update |= ImGui::InputFloat("Frame Height", &frameh, 0.1f);


            float Aperture = 1280;
            ImGui::Text("Aperture");
            update |= ImGui::InputFloat("Size", &Aperture, 0.1f);


            float Projection = 1280;
            ImGui::Text("Projection");
            float fov = camera.get_fov_deg();
            update |= ImGui::InputFloat("View Field", &fov, 0.1f);

            if(update){
                // update_vol_camera_setting(false);
                camera.set_position(campos);
                camera.set_perspective(fov, nearz, farz);
                camera.set_direction(vutil::deg2rad(camdirhv.x), vutil::deg2rad(camdirhv.y));
                camera.set_move_speed(moves);
                camera.set_view_rotation_speed(rotas);
                //global camera for get proj view
                camera.set_w_over_h((float)framew / frameh);
            }
            // ImGui::TreePop();
            // }
            // if(ImGui::TreeNode("Vol Render Setting")){
            if(ImGui::Checkbox("Render Volume", &render_volume_control)){
                // if(vol_render_volume) status_flags |= VOL_DRAW_VOLUME;
                // else status_flags &= ~VOL_DRAW_VOLUME;
            }

            ImGui::SameLine();
            ImGui::Checkbox("Clear History", &clear_history);

            ImGui::SameLine();
            bool mode_changed = ImGui::RadioButton("PB", &per_frame_params.debug_mode, 0);
            ImGui::SameLine();
            mode_changed |= ImGui::RadioButton("RT", &per_frame_params.debug_mode, 1);
            if(mode_changed) clear_framebuffer();


            ImGui::NewLine();

            ImGui::Text("Transfer Function");
            // if(ImGui::TreeNode("TransferFunc")){
            bool tf_update = false;

            static std::map<int, Float4> pt_mp;
            pt_mp.clear();
            for(int i =0;i<render_params.tf.tf_pts.pts.size();i++){
                pt_mp[int(render_params.tf.tf_pts.pts[i].first * 255)] = render_params.tf.tf_pts.pts[i].second;
            }




            static Float3 color;
            static bool selected_pt = false;
            static int sel_pos;

            if(selected_pt){
                color.x = pt_mp.at(sel_pos).x;
                color.y = pt_mp.at(sel_pos).y;
                color.z = pt_mp.at(sel_pos).z;
            }



            ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
            ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
            const int ysize = 255;
            canvas_sz.y = ysize;
            ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
            ImGui::InvisibleButton("tf", canvas_sz);


            ImGuiIO& io = ImGui::GetIO();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(30, 30, 30, 255));
            draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(200, 200, 200, 255));

            const bool is_hovered = ImGui::IsItemHovered(); // Hovered
            const bool is_active = ImGui::IsItemActive();   // Held
            const ImVec2 origin(canvas_p0.x, canvas_p0.y); // Lock scrolled origin
            const ImVec2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);
            const ImVec2 tf_origin(canvas_p0.x, canvas_p0.y + canvas_sz.y);

            bool check_add = false;
            if(is_active && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)){
                check_add = true;
            }

            auto canvas_y_to_alpha = [&](float y){
                return (ysize - y) / float(ysize);
            };
            auto alpha_to_canvas_y = [&](float alpha){
                return ysize - alpha * ysize;
            };

            if(is_active && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                bool pick = false;
                for(auto& [x_pos, color] : pt_mp){
                    if(std::abs(x_pos - mouse_pos_in_canvas.x) < 5
                        && std::abs(alpha_to_canvas_y(color.w) - mouse_pos_in_canvas.y) < 5){
                        selected_pt = true;
                        sel_pos = x_pos;
                        pick = true;
                        break;
                    }
                }
                if(!pick) selected_pt = false;
            }

            if(!selected_pt && check_add){
                auto it = pt_mp.upper_bound(mouse_pos_in_canvas.x);
                Float4 rgba;
                rgba.w = canvas_y_to_alpha(mouse_pos_in_canvas.y);
                if(it == pt_mp.end()){
                    auto itt = pt_mp.lower_bound(mouse_pos_in_canvas.x);
                    if(itt == pt_mp.begin()){
                        rgba.x = rgba.y = rgba.z = 0.f;
                    }
                    else{
                        itt = std::prev(itt);
                        rgba.x = itt->second.x;
                        rgba.y = itt->second.y;
                        rgba.z = itt->second.z;
                    }
                }
                else{
                    auto itt = pt_mp.lower_bound(mouse_pos_in_canvas.x);
                    if(itt == pt_mp.begin()){
                        rgba.x = it->second.x;
                        rgba.y = it->second.y;
                        rgba.z = it->second.z;
                    }
                    else{
                        itt = std::prev(itt);
                        float u = (mouse_pos_in_canvas.x - itt->first) / (float)(it->first - itt->first);
                        rgba.x = itt->second.x * (1.f - u) + it->second.x * u;
                        rgba.y = itt->second.y * (1.f - u) + it->second.y * u;
                        rgba.z = itt->second.z * (1.f - u) + it->second.z * u;
                    }
                }
                pt_mp[mouse_pos_in_canvas.x] = rgba;
                selected_pt = true;
                sel_pos = mouse_pos_in_canvas.x;
                tf_update = true;
            }

            //add
            if(is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)
                && selected_pt){
                int nx = sel_pos + io.MouseDelta.x;
                auto c = pt_mp.at(sel_pos);
                int ny = alpha_to_canvas_y(c.w) + io.MouseDelta.y;
                ny = (std::min)(ny, ysize);
                ny = (std::max)(ny, 0);
                if(nx == sel_pos || pt_mp.count(nx) == 0){
                    c.w = canvas_y_to_alpha(ny);
                    pt_mp.erase(sel_pos);
                    sel_pos = nx;
                    pt_mp[nx] = c;
                    tf_update = true;
                }
            }

            //delete
            if(is_active && ImGui::IsMouseClicked(ImGuiMouseButton_Right)
                && selected_pt){
                selected_pt = false;
                pt_mp.erase(sel_pos);
                tf_update = true;
            }

            draw_list->PushClipRect(canvas_p0, canvas_p1, true);
            bool first = true;
            ImVec2 prev;
            if(!pt_mp.empty()){
                auto it = pt_mp.begin();
                ImVec2 p = ImVec2(it->first + origin.x, alpha_to_canvas_y(it->second.w) + origin.y);
                draw_list->AddLine(ImVec2(origin.x, p.y), p, IM_COL32(0, 0, 0, 255));
                auto itt = std::prev(pt_mp.end());
                p = ImVec2(itt->first + origin.x, alpha_to_canvas_y(itt->second.w) + origin.y);
                draw_list->AddLine(p, ImVec2(origin.x + canvas_sz.x, p.y), IM_COL32(0, 0, 0, 255));
            }
            for(auto& [x, c] : pt_mp){
                ImVec2 cur = ImVec2(x + origin.x, alpha_to_canvas_y(c.w) + origin.y);
                if(first){
                    first = false;
                }
                else{
                    draw_list->AddLine(prev, cur, IM_COL32(0, 0, 0, 255));
                }
                prev = cur;
            }
            for(auto& [x, c] : pt_mp){
                ImVec2 cur = ImVec2(x + origin.x, alpha_to_canvas_y(c.w) + origin.y);
                draw_list->AddCircleFilled(cur, 5.f,
                                           IM_COL32(int(c.x * 255), int(c.y * 255), int(c.z * 255), 255));
                if(x == sel_pos && selected_pt){
                    draw_list->AddCircle(cur, 6.f, IM_COL32(255, 127, 0, 255), 0, 2.f);
                }
            }
            draw_list->PopClipRect();



            ImGui::NewLine();
            ImGui::Text("Node settings");

            ImGui::Text("Selected Node");


            if(selected_pt){
                ImGui::SameLine();
                if(ImGui::Button("Delete", ImVec2(80, 15))){
                    selected_pt = false;
                    pt_mp.erase(sel_pos);
                    tf_update = true;
                }
                ImGui::NewLine();
            }



            int intensity = 0;
            float opacity = 0.f;
            float roughness = 0.f;

            if(selected_pt){
                intensity = sel_pos;
                opacity = pt_mp.at(sel_pos).w;
            }


            ImGui::BulletText("Intensity: %d", intensity);
            tf_update |= ImGui::SliderFloat("Opacity", &opacity, 0.00f, 1.00f);
            tf_update |= ImGui::InputFloat("Glossiness", &roughness, 0.1f);

            if(ImGui::ColorEdit3("Diffuse", &color.x)){
                if(selected_pt){
                    auto& c = pt_mp.at(sel_pos);
                    c.x = color.x;
                    c.y = color.y;
                    c.z = color.z;

                    tf_update = true;
                }
            }


            if(ImGui::ColorEdit3("Specular", &color.x)){
                if(selected_pt){
                    auto& c = pt_mp.at(sel_pos);
                    c.x = color.x;
                    c.y = color.y;
                    c.z = color.z;

                    tf_update = true;
                }
            }



            if(ImGui::ColorEdit3("Emission", &color.x)){
                if(selected_pt){
                    auto& c = pt_mp.at(sel_pos);
                    c.x = color.x;
                    c.y = color.y;
                    c.z = color.z;

                    tf_update = true;
                }
            }
            // ImGui::TreePop();

            if(tf_update){
                if(selected_pt)
                pt_mp[sel_pos].w = opacity;

                render_params.tf.tf_pts.pts.clear();
                for(auto i = pt_mp.begin();i!=pt_mp.end(); i++){
                    render_params.tf.tf_pts.pts.push_back({i->first / 255.0f, i->second});
                }
                render_params.tf.updated = true;
                pb_vol_renderer->SetRenderParams(render_params);
                render_params.tf.updated = false;

                // std::vector<std::pair<float, Float4>> pts;
                // for(auto& [x, c] : pt_mp){
                //     pts.emplace_back((float)x / (float)canvas_sz.x, c);
                // }
                // vol_render_resc->UpdateTransferFunc(pts);

                // status_flags |= VOL_RENDER_PARAMS_CHANGED;
            }
            // }

            // ImGui::TreePop();
            // }
            // if(ImGui::TreeNode("SWC Render Setting")){
            // if(ImGui::Checkbox("Render SWC", &vol_render_swc)){
            //     if(vol_render_swc) status_flags |= VOL_DRAW_SWC;
            //     else status_flags &= ~VOL_DRAW_SWC;
            // }

            // if(ImGui::Checkbox("Blend With Depth", &vol_swc_blend_with_depth)){
            //     if(vol_swc_blend_with_depth) status_flags |= VOL_SWC_VOLUME_BLEND_WITH_DEPTH;
            //     else status_flags &= ~VOL_SWC_VOLUME_BLEND_WITH_DEPTH;
            // }
            // if(ImGui::Checkbox("Render SWC Point Tag", &vol_render_swc_point_tag)){
            //     if(vol_render_swc_point_tag) status_flags |= VOL_DRAW_SWC_TAG;
            //     else status_flags &= ~VOL_DRAW_SWC_TAG;
            // }


            // ImGui::TreePop();
            //}
        }

        ImGui::End();
    }

    void show_editor_vol_info_window() {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove |  ImGuiConfigFlags_NoMouseCursorChange;

        //

        bool use_work_area = true;


        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(use_work_area ? ImVec2(viewport->WorkPos.x, viewport->WorkPos.y+18) : viewport->Pos);
        ImGui::SetNextWindowSize(use_work_area ? ImVec2(viewport->WorkSize.x/6, viewport->WorkSize.y/5) : viewport->Size);
        bool xx = true;
        bool* p_open = NULL;

        static std::string buffer(256, '\0');
        if (ImGui::Begin("Vol Info", p_open, window_flags)) {
            ImGui::NewLine();
            ImGui::InputText("Volume File", &buffer[0], buffer.length());
            if (ImGui::Button("Select", ImVec2(120, 15))) {
                // vol_file_dialog.Open();
            }
            ImGui::SameLine();
            if (ImGui::Button("Load", ImVec2(120, 15))) {
                // load_volume(buffer);
            }
            ImGui::NewLine();
            ImGui::BulletText("Volume Lod Levels: %d", render_params.lod.leve_of_dist.MaxLevelCount);
            ImGui::BulletText("Volume Space Ratio: %.1f %.1f %.1f", 1.0f, 1.0f, 1.0f);

            std::string str= volume_desc.volume_name;
            char* pstr=(char*)str.data();

            if (true) {

                if (ImGui::TreeNode("Volume Desc")) {
                    ImGui::BulletText("Volume Name: %s", pstr);
                    ImGui::BulletText("Volume Dim: (%d, %d, %d)", volume_desc.shape.x, volume_desc.shape.y, volume_desc.shape.z);
                    ImGui::BulletText("Voxel Space: (%.5f, %.5f, %.5f)", volume_desc.voxel_space.x, volume_desc.voxel_space.y, volume_desc.voxel_space.z);
                    ImGui::BulletText("Samples Per Voxel: %d", volume_desc.samples_per_voxel);
                    ImGui::BulletText("Bits Per Sample: %d", volume_desc.bits_per_sample);
                    ImGui::BulletText("Voxel Is Float: %s", volume_desc.is_float ? "yes" : "no");
                    ImGui::BulletText("Block Length: %d", volume_desc.block_length);
                    ImGui::BulletText("Block Padding: %d", volume_desc.padding);
                    ImGui::BulletText("Block Dim: (%d, %d, %d)", volume_desc.blocked_dim.x, volume_desc.blocked_dim.y, volume_desc.blocked_dim.z);
                    ImGui::TreePop();
                }
            }
        }

        ImGui::End();

        // vol_file_dialog.Display();
        // if (vol_file_dialog.HasSelected()) {
        //     std::cout << vol_file_dialog.GetSelected().string() << std::endl;
        //     buffer = vol_file_dialog.GetSelected().string();
        //     buffer.resize(256, '\0');
        //     vol_file_dialog.ClearSelected();
        // }
    }



    void show_editor_vol_render_window() {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove |  ImGuiConfigFlags_NoMouseCursorChange | ImGuiWindowFlags_HorizontalScrollbar;


        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x / 6, viewport->WorkPos.y+18));
        ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 2 / 3, viewport->WorkSize.y  - 18));
        bool xx = true;
        bool* p_open = NULL;

        if (ImGui::Begin("Vol Render", p_open, window_flags)) {
            // auto [px, py] = ImGui::GetWindowPos();
            // window_priv_data.vol_render_window_pos = vec2i(px, py);
            // auto [x, y] = ImGui::GetWindowSize();
            // y -= 20;
            // if (window_priv_data.vol_render_window_size != vec2i(x, y)) {
            //     window_priv_data.vol_render_resize = true;
            //     window_priv_data.vol_render_window_size = vec2i(x, y);
            //     ImGui::End();
            //     return;
            // }
            // ImGui::InvisibleButton("vol-render", ImVec2(x, y));
            // window_priv_data.vol_mesh_render_hovered = ImGui::IsItemHovered();

            // frame_vol_render();

        }

        ImGui::End();
    }


    void show_debug_window() {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove |  ImGuiConfigFlags_NoMouseCursorChange | ImGuiWindowFlags_HorizontalScrollbar;


        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 5 / 6, viewport->WorkPos.y+ viewport->WorkSize.y * 2 / 3 + 18));
        ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 1 / 6, viewport->WorkSize.y * 1 / 3 - 18));
        bool xx = true;
        bool* p_open = NULL;

        if(p_open && !*p_open) return;

        if(ImGui::Begin("Debug Info", p_open, window_flags)){
            // if(ImGui::TreeNode("App Settings")){
            ImGui::Text("Device Info");
            ImGui::BulletText("MaxHostMem: %dd GB", params.memory.max_host_mem_bytes);
            ImGui::BulletText("MaxRenderGPUMem: %d GB", params.memory.max_gpu_mem_bytes);
            ImGui::BulletText("RenderGPUIndex: %d", 0);
            ImGui::BulletText("ComputeGPUIndex: %d", 0);
            ImGui::BulletText("MaxFixedHostMem: %.2f", 8.00);
            ImGui::BulletText("ThreadsGroupWorkerCount: %d", 512);
            ImGui::BulletText("VTexture Count: %d", pb_create_info_params.vtex_cnt);
            ImGui::BulletText("VTexture Shape: (%d, %d, %d)", pb_create_info_params.vtex_shape.x, pb_create_info_params.vtex_shape.y, pb_create_info_params.vtex_shape.z);


            ImGui::NewLine();
            ImGui::Text("Render Monitor");
            // ImGui::TreePop();
            // }
            // if(ImGui::TreeNode("Timer")){
            ImGui::BulletText("App FPS: %.2f", ImGui::GetIO().Framerate);
//            ImGui::BulletText("Vol Render Frame Time: %s", "100ms");
            ImGui::BulletText("Cur Frame Index: %d", cur_frame_index);

            for(auto& [desc, time] : time_records){
                ImGui::BulletText("%s cost time: %s ms", desc.c_str(), time.ms().fmt().c_str());
            }
            // ImGui::TreePop();
            // }

            // if(ImGui::TreeNode("Mesh Render")){
            //     ImGui::Checkbox("Line Mode", &debug_priv_data.mesh_render_line_mode);
            //     ImGui::TreePop();
            // }

        }

        ImGui::End();
    }



    void show_editor_light_setting_window() {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove | ImGuiConfigFlags_NoMouseCursorChange | ImGuiWindowFlags_HorizontalScrollbar;


        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y * 1 / 5 + 18));
        ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 1 / 6, viewport->WorkSize.y * 1 / 2 ));
        bool xx = true;
        bool* p_open = NULL;



        if (ImGui::Begin("Light Settings", p_open, window_flags)) {
            // auto& pos = camera.get_position();

            ImGui::Text("Light Source Select");

            bool update = false;
            static int selected = 0;
            ImGui::BeginChild("left pane", ImVec2(viewport->WorkSize.x * 1 / 8 - 20,  viewport->WorkSize.y * 1 / 5), true);
            for (int i = 0; i < render_params.light.lightnum; i++)
            {
                // FIXME: Good candidate to use ImGuiSelectableFlags_SelectOnNav
                std::string label = "Light ";
                label = label + std::to_string(i);

                char *tt = &label[0];
                if (ImGui::Selectable(tt, selected == i))
                    selected = i;
            }
            ImGui::EndChild();


            if(render_params.light.lightnum < 4){
                if (ImGui::Button("Add Square Light", ImVec2(120, 15))) {
                    render_params.light.lightnum += 1;
                    render_params.light.lights[render_params.light.lightnum-1].m_T = 0;
                    update = true;
                }
                ImGui::SameLine();

                if (ImGui::Button("Add Circle Light", ImVec2(120, 15))) {
                    render_params.light.lightnum += 1;
                    render_params.light.lights[render_params.light.lightnum-1].m_T = 1;
                    update = true;
                }
            }



            ImGui::NewLine();


            ImGui::Text("Light Source Settings");

            ImGui::Text("Selected Light ID: %d", selected);

            ImGui::SameLine();

            if(render_params.light.lightnum > 1){
                if (ImGui::Button("Delete", ImVec2(80, 15))) {
                    render_params.light.lights[selected] = render_params.light.lights[render_params.light.lightnum-1];
                    render_params.light.lightnum -= 1;
                    update = true;
                }
            }



            ImGui::NewLine();
            ImGui::Text("Selected Light Type: %s", render_params.light.lights[selected].m_T == 0 ? "Square" : "Circle");




            float ltheta_degree = vutil::rad2deg(render_params.light.lights[selected].m_Theta);
            float lphi_degree = vutil::rad2deg(render_params.light.lights[selected].m_Phi);


            if(render_params.light.lights[selected].m_T == 0){


                update |= ImGui::SliderFloat("Longtitude", &ltheta_degree, -180.0f, 180.0f);


                update |= ImGui::SliderFloat("Latitude", &lphi_degree, -180.0f, 180.0f);


                update |= ImGui::InputFloat("Distance", &render_params.light.lights[selected].m_Distance, 0.1f);


                static bool lock_size = true;

                update |= ImGui::InputFloat("Width", &render_params.light.lights[selected].m_Width, 0.1f);
                if(lock_size == true){
                    render_params.light.lights[selected].m_Height = render_params.light.lights[selected].m_Width;
                }
                update |= ImGui::InputFloat("Height", &render_params.light.lights[selected].m_Height, 0.1f);
                if(lock_size == true){
                    render_params.light.lights[selected].m_Width = render_params.light.lights[selected].m_Height;
                }






                if (ImGui::ColorEdit3("Light Color(RGBA)", &render_params.light.lights[selected].m_Color.x)) {

                    update = true;

                }

                update |= ImGui::SliderFloat("Intensity", &render_params.light.lights[selected].m_intensity, 1.0f, 200.0f);


                // ImGui::TreePop();
                // }
                // if(ImGui::TreeNode("Vol Render Setting")){

                if (ImGui::Checkbox("Lock Size", &lock_size)) {
                    // if(vol_render_volume) status_flags |= VOL_DRAW_VOLUME;
                    // else status_flags &= ~VOL_DRAW_VOLUME;
                }
            }
            else{

                if (ImGui::ColorEdit3("Light ColorTop(RGBA)", &render_params.light.lights[selected].m_ColorTop.x)) {

                    update = true;

                }

                if (ImGui::ColorEdit3("Light ColorMiddle(RGBA)", &render_params.light.lights[selected].m_ColorMiddle.x)) {

                    update = true;

                }

                if (ImGui::ColorEdit3("Light ColorBottem(RGBA)", &render_params.light.lights[selected].m_ColorBottom.x)) {

                    update = true;

                }

                update |= ImGui::SliderFloat("Intensity", &render_params.light.lights[selected].m_intensity, 1.0f, 200.0f);
            }

            if (update) {
                // update_vol_camera_setting(false);
                render_params.light.lights[selected].m_Theta = vutil::deg2rad(ltheta_degree);
                render_params.light.lights[selected].m_Phi = vutil::deg2rad(lphi_degree);

                render_params.light.updated = true;
                pb_vol_renderer->SetRenderParams(render_params);
                render_params.light.updated = false;
            }

            ImGui::NewLine();

        }

        ImGui::End();
    }




    void show_editor_appearance_window() {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove | ImGuiConfigFlags_NoMouseCursorChange | ImGuiWindowFlags_HorizontalScrollbar;


        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y * 7 / 10 + 18));
        ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 1 / 6, viewport->WorkSize.y * 3 / 10 - 18 ));
        bool xx = true;
        bool* p_open = NULL;



        if (ImGui::Begin("Appearance", p_open, window_flags)) {
            // auto& pos = camera.get_position();

            ImGui::Text("Appearance Settings");

            Float3 tinput;
            tinput.x = 0;
            bool update = false;

            update |= ImGui::SliderFloat("Density Scale", &pb_params.density_scale, 1.f, 100.f);

            if(update){
                pb_vol_renderer->SetPBParams(pb_params);

                clear_framebuffer();
            }


            update = false;

            update = ImGui::Checkbox("Tone Mapping", reinterpret_cast<bool*>(&post_params.tone_mapping));
            static float exposure = 1.f;
            if(ImGui::InputFloat("Exposure", &exposure)){
                post_params.inv_exposure = 1.f / exposure;
                update = true;
            }
            if(update) post_params_buffer.set_buffer_data(&post_params);



            update = false;

            static int n = 0;
            update |= ImGui::Combo("Combo", &n, "Hybrid\0Single\0Other\0\0");
            update |= ImGui::InputFloat("Gradient Factor", &tinput.x);


            update |= ImGui::SliderFloat("Ray Step", &render_params.raycast.ray_step, 0.0001f, 0.01f);
            update |= ImGui::InputFloat("Max Ray Dist", &render_params.raycast.max_ray_dist, 0.01f);

            if(update){
                render_params.raycast.updated = true;
                pb_vol_renderer->SetRenderParams(render_params);
                render_params.raycast.updated = false;
            }

        }

        ImGui::End();
    }

    void render_ui(){
        show_editor_menu();
        show_editor_vol_render_info_window();
        show_editor_vol_info_window();
        show_editor_light_setting_window();
        show_debug_window();
        show_editor_appearance_window();


//        if(ImGui::Begin("VolRenderer Settings")){
//            ImGui::Text("Cur Frame Index: %d", cur_frame_index);
//            ImGui::Checkbox("Clear History", &clear_history);
//            bool mode_changed = ImGui::RadioButton("PB", &per_frame_params.debug_mode, 0);
//            ImGui::SameLine();
//            mode_changed |= ImGui::RadioButton("RT", &per_frame_params.debug_mode, 1);
//            if(mode_changed) clear_framebuffer();
//            for(auto& [desc, time] : time_records){
//                ImGui::Text("%s cost time: %s ms", desc.c_str(), time.ms().fmt().c_str());
//            }

//            if(ImGui::TreeNode("PB Params")){
//                bool update = false;
//
//                update |= ImGui::SliderFloat("Density Scale", &pb_params.density_scale, 1.f, 100.f);
//
//
//                if(update){
//                    pb_vol_renderer->SetPBParams(pb_params);
//
//                    clear_framebuffer();
//                }
//
//
//                ImGui::TreePop();
//            }

//            if(ImGui::TreeNode("Image Post Process")){
//                bool update = ImGui::Checkbox("Tone Mapping", reinterpret_cast<bool*>(&post_params.tone_mapping));
//                static float exposure = 1.f;
//                if(ImGui::InputFloat("Exposure", &exposure)){
//                    post_params.inv_exposure = 1.f / exposure;
//                    update = true;
//                }
//                if(update) post_params_buffer.set_buffer_data(&post_params);
//
//                ImGui::TreePop();
//            }


//        }

//        ImGui::End();
    }

    void after_render(){

        if(render_volume_control)cur_frame_index++;
        else cur_frame_index = 0;

        accu_params.frame_index = cur_frame_index;
        accu_params_buffer.set_buffer_data(&accu_params);
    }


    void test_time(std::function<void()> f, const std::string& desc){
        timer.start();
        f();
        timer.stop();
        time_records[desc] = timer.duration();
    }
  private:
    VolRendererConfigParams params;

    Timer timer;
    std::map<std::string, Duration> time_records;

    struct{
        int cur_frame_index = 0;
        bool clear_history = false;

    };

    struct{
        cudaGraphicsResource_t cuda_frame_color_resc = nullptr;// float4

    };

    struct{
        framebuffer_t fbo;
        renderbuffer_t rbo;
        texture2d_t frame_color; //rgba32f
        texture2d_t accumulate_color; // rgba32f
        texture2d_t display_color; // same with accumulate_color except different format with rgba8
    };

    PerFrameParams per_frame_params;

    PBVolumeRenderer::PBParams pb_params;

    PBVolumeRenderer::PBVolumeRendererCreateInfo pb_create_info_params;

    Handle<FrameBuffer> framebuffer;

    Handle<PBVolumeRenderer> pb_vol_renderer;

    RenderParams render_params;


    GridVolume::GridVolumeDesc volume_desc;


    struct{
        struct alignas(16) AccumulateParams{
            int frame_index = 0;
        }accu_params;
        std140_uniform_block_buffer_t<AccumulateParams> accu_params_buffer;
        program_t accumulator;
        struct alignas(16) PostProcessParams{
            int tone_mapping = 1;
            float inv_exposure = 1.f;
        }post_params;
        std140_uniform_block_buffer_t<PostProcessParams> post_params_buffer;
        program_t post_processor;
        program_t screen;
        vertex_array_t quad_vao;
    };
};

int main(int argc, char** argv){
    try{
        UI(window_desc_t{.size = {WINDOW_WIDTH, WINDOW_HEIGHT}, .title = "VolRendererUI"}).init(argc, argv).run();
    }
    catch (const std::exception& err){
        std::cerr<< "VolRendererUI exited with error: " << err.what() << std::endl;
    }
}