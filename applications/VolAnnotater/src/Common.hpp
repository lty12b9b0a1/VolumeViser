#pragma once

#include <Algorithm/LevelOfDetailPolicy.hpp>
#include <Core/Renderer.hpp>
#include <Core/HashPageTable.hpp>
#include <Model/SWC.hpp>
#include <Model/Mesh.hpp>
#include <Model/SWC.hpp>
#include <IO/SWCIO.hpp>

#include <cuda_gl_interop.h>
#include <json.hpp>

#include <set>
#include <fstream>

using namespace viser;
using namespace vutil;
using namespace vutil::gl;

using PatchID = size_t;
using BlockUID = GridVolume::BlockUID;

class SWCRenderer;
class NeuronRenderer;

struct VolAnnotaterCreateInfo;

struct AppSettings{
public:
    inline static size_t MaxHostMemBytes = 0;
    inline static size_t MaxRenderGPUMemBytes = 0;
    inline static size_t MaxComputeGPUMemBytes = 0;
    inline static int RenderGPUIndex = 0;
    inline static int ComputeGPUIndex = 0;
    inline static size_t MaxFixedHostMemBytes = 0;
    inline static int ThreadsGroupWorkerCount = 0;
    inline static int VTexCount = 0;
    inline static Int3 VTexShape = Int3(0, 0, 0);

    static void Initialize(const VolAnnotaterCreateInfo& info);
};

struct ViserRescPack{
public:
    Ref<HostMemMgr> host_mem_mgr_ref;

    Ref<GPUMemMgr> render_gpu_mem_mgr_ref;

    Ref<GPUMemMgr> compute_gpu_mem_mgr_ref;

    Ref<FixedHostMemMgr> host_block_pool_ref;

    Ref<GPUVTexMgr> gpu_vtex_mgr_ref;

    Ref<GPUPageTableMgr> gpu_pt_mgr_ref;

    vutil::thread_group_t thread_group;

    struct {
        Handle<GridVolume> volume;

        int max_lod = 0;

        Float3 volume_space_ratio = {1.f, 1.f, 1.f};

    }vol_priv_data;

public:
    void Initialize();

    void LoadVolume(const std::string& filename);
private:
    /**
     * @brief 初始化一些与体数据相关的资源，在体数据加载后调用
     */
    void InitializeVolumeResc();
};

/**
 * @brief 大规模体绘制相关资源
 */
struct VolRenderRescPack{
public:

    Handle<CRTVolumeRenderer> crt_vol_renderer;

    float render_base_space = 0.f;

    Handle<FrameBuffer> framebuffer;

    viser::LevelOfDist lod;
    float lod_ratio = 1.f;

    struct{
        Float3 lod0_block_length_space;
        BoundingBox3D volume_bound;
        UInt3 lod0_block_dim;
    }render_vol;

    struct {
        std::vector<GridVolume::BlockUID> intersect_blocks;
        std::vector<GPUPageTableMgr::PageTableItem> blocks_info;
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> host_blocks;//在循环结束会释放Handle
        std::unordered_map<GridVolume::BlockUID, Handle<CUDAHostBuffer>> missed_host_blocks;
        std::map<int, std::vector<std::function<void()>>> task_mp;
        std::map<int, vutil::task_group_handle_t> task_groups;
        std::vector<int> lods;

        PerFrameParams per_frame_params;
    }vol_render_priv_data;

    struct {
        vec2i query_pos = {-1, -1};
        bool clicked = false;
        Handle<CUDAHostBuffer> query_info;
        CUDABufferView1D<float> query_info_view;
    }vol_query_priv_data;

public:
    void Initialize(ViserRescPack& _);

    void OnVolumeLoaded(ViserRescPack& _);

    void UpdateUpBoundLOD(ViserRescPack& _, float fov_rad, float ratio = 1.f);

    void UpdateDefaultLOD(ViserRescPack& _, float ratio = 1.f);
};

using SWCUID = viser::UnifiedRescUID;
using SWCNeuronID = size_t;
using SWCPointKey = SWC::SWCPointKey;

constexpr size_t SWC_MAX_NEURON_NUM = 1024ull;
constexpr size_t SWC_MAX_POINT_NUM = 1ull << 16;
/**
 * @brief SWC标注和渲染相关资源
 */
struct SWCRescPack{
public:
    Handle<SWCFile> swc_file;
    struct SWCInfo{
        Handle<SWC> swc;
        std::string name;//swc名称
        std::string filename;//文件名
    };
    std::unordered_map<SWCUID, SWCInfo> loaded_swc;//所有加载、创建的swc对象
    SWCUID selected_swc_uid = INVALID_RESC_ID;//当前被选中编辑的swc uid

    // 一个swc可能含有多条神经 每条神经是一颗树 只有一个根节点 并且每条神经之间是不连通的
    struct{

        // 没被使用的neuron id集合
        std::set<SWCNeuronID> available_neuron_ids;

        // 一个swc里的不同neuron共用一个可用的swc点集合
        std::set<SWCPointKey> available_swc_pt_ids;

        //每条神经元根节点到神经元id的mapping
        std::unordered_map<SWCPointKey, SWCNeuronID> pt_to_neuron_mp;

        SWCPointKey last_picked_swc_pt_id = SWC::INVALID_SWC_KEY;

        void Reset(){
            available_neuron_ids.clear();
            for(size_t i = 1; i <= SWC_MAX_NEURON_NUM; i++){
                available_neuron_ids.insert(i);
            }

            available_swc_pt_ids.clear();
            for(size_t i = 1; i <= SWC_MAX_POINT_NUM; i++){
                available_swc_pt_ids.insert(i);
            }
            pt_to_neuron_mp.clear();

            last_picked_swc_pt_id = SWC::INVALID_SWC_KEY;
        }
    }swc_priv_data;

    std::unique_ptr<SWCRenderer> swc_renderer;

public:
    bool Selected() const { return CheckUnifiedRescUID(selected_swc_uid); }

    SWCInfo& GetSelected() { return loaded_swc.at(selected_swc_uid); }

    void Initialize();
    /**
     * @brief 从swc文件中加载swc数据到内存模型中
     * @note 完成后不会被切换到选中状态
     */
    void LoadSWCFile(const std::string& filename);

    /**
     * @brief 创建一个新的swc文件
     * @note 成功后会调用SelectSWC
     */
    void CreateSWC(const std::string& filename = "");

    /**
     * @brief 在内存中删除当前选中的swc文件
     */
    void DeleteSelSWC();

    /**
     * @brief 切换当前选择的swc文件 会改变swc_priv_data内部成员变量
     */
    void SelectSWC(SWCUID swc_id);

    /**
     * @brief 插入一个点到当前选中的swc
     * @note 传入的点不需要id和pid 内部会生成这两个
     */
    void InsertSWCPoint(SWC::SWCPoint pt);

    /**
     * @brief 将从文件中加载的swc写会文件 如果当前选择的swc不是从文件中加载得到的 那么不会写入
     */
    void SaveSWCToFile();

    /**
     * @brief 将当前选定的swc保存到文件中 文件名包含了swc文件的格式时txt还是bin
     */
    void ExportSWCToFile(const std::string& filename);

private:

};
using MeshUID = viser::UnifiedRescUID;

/**
 * @brief SWC转换为Mesh和神经元网格渲染相关资源
 */
struct SWC2MeshRescPack{
public:
    MeshFile mesh_file;
    //一个完整的网格文件 由一个swc生成 可能包含多个不连通的神经元
    struct MeshInfo{
        Handle<Mesh> mesh;
        std::string name;
        std::string filename;
    };
    //所有加载的mesh文件
    std::unordered_map<MeshUID, MeshInfo> loaded_mesh;

    // 一个block对应的mesh 这里的PatchID同时也被渲染器使用
    enum BlockMeshStatus{
        Merged,
        Blocked
    };
    struct{
        std::set<MeshUID> available_mesh_ids;

        std::unordered_map<PatchID, Handle<Mesh>> patch_mesh_mp;

        BlockMeshStatus status;

    }s2m_priv_data;

    Handle<NeuronRenderer> neuron_renderer;

public:
    void Initialize();

    void LoadMeshFile(const std::string& filename);

    void Select(MeshUID mesh_id);

    void ResetBlockedMesh();
};
