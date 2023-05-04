#pragma once

#include <Common/Common.hpp>
#include <limits>

VISER_BEGIN

// 渲染需要的参数，可以一开始设置后一直保持不变
// 主要包括光照参数、LOD计算参数、传输函数

//使用统一的标准单位meter作为世界坐标系的度量单位

struct LevelOfDist{
    static constexpr int MaxLevelCount = 16;
    float LOD[MaxLevelCount] = {std::numeric_limits<float>::max()};
};

template<typename K, typename V>
struct TransferFuncT{
    using Key = K;
    using Value = V;
    using Item = std::pair<Key, Value>;
    mutable std::vector<Item> pts;

    std::vector<Value> Gen1DTF(int dim = 256) const ;

    std::vector<Value> Gen2DTF(const std::vector<Value>& tf1d, int dim = 256) const ;

    void Gen1DTF(Handle<CUDAHostBuffer>& buffer, int dim = 256) const ;

    void Gen2DTF(Handle<CUDAHostBuffer>& tf1d, Handle<CUDAHostBuffer>& buffer, int dim = 256) const ;
};
using TransferFunc = TransferFuncT<float, Float4>;

inline const Float3 WorldUp = Float3{0.f, 1.f, 0.f};

struct Camera{
    Float3 pos;
    Float3 target;
    Float3 up;
    float near;
    float far;
    float fov;
    int width;
    int height;

    Mat4 GetViewMatrix() const{
        return Transform::look_at(pos, target, up);
    }

    Mat4 GetProjMatrix() const{
        return Transform::perspective(vutil::deg2rad(fov * 0.5f),
                                      static_cast<float>(width) / height,
                                      near, far);
    }

    Mat4 GetProjViewMatrix() const{
        return GetProjMatrix() * GetViewMatrix();
    }

};

struct VolumeParams{
    BoundingBox3D bound;
    uint32_t block_length;
    uint32_t padding;
    UInt3 voxel_dim;
    Float3 space;
};

struct Clight{
    float			m_Theta;
    float			m_Phi;
    float			m_Width;
    float			m_InvWidth;
    float			m_HalfWidth;
    float			m_InvHalfWidth;
    float			m_Height;
    float			m_InvHeight;
    float			m_HalfHeight;
    float			m_InvHalfHeight;
    float			m_Distance;
    float			m_SkyRadius;
    float3			m_P;
    float3			m_Target;
    float3			m_N;
    float3			m_U;
    float3			m_V;
    float			m_Area;
    float			m_AreaPdf;
    float3      	m_Color;
    float           m_intensity;
    float3      	m_ColorTop;
    float3      	m_ColorMiddle;
    float3      	m_ColorBottom;
    int				m_T;
    Clight(void) :
          m_Theta(0.0f),
          m_Phi(0.0f),
          m_Width(1.0f),
          m_InvWidth(1.0f / m_Width),
          m_HalfWidth(0.5f * m_Width),
          m_InvHalfWidth(1.0f / m_HalfWidth),
          m_Height(1.0f),
          m_InvHeight(1.0f / m_Height),
          m_HalfHeight(0.5f * m_Height),
          m_InvHalfHeight(1.0f / m_HalfHeight),
          m_Distance(1.0f),
          m_SkyRadius(100.0f),
          m_Area(m_Width * m_Height),
          m_AreaPdf(1.0f / m_Area),
          m_intensity(100.0f),
          m_T(0){
        m_P = make_float3(1.0f, 1.0f, 1.0f);
        m_Target = make_float3(0.0f, 0.0f, 0.0f);
        m_N = make_float3(1.0f, 0.0f, 0.0f);
        m_U = make_float3(1.0f, 0.0f, 0.0f);
        m_V = make_float3(1.0f, 0.0f, 0.0f);
        m_Color = make_float3(10.0f, 10.0f, 10.0f);
        m_ColorTop = make_float3(10.0f, 10.0f, 10.0f);
        m_ColorMiddle = make_float3(10.0f, 10.0f, 10.0f);
        m_ColorBottom = make_float3(10.0f, 10.0f, 10.0f);
    }
};


struct RenderParams{
    void Reset(){
        light.updated   = false;
        lod.updated     = false;
        tf.updated      = false;
        {
            tf.tf_pts.pts.clear();
        }
        distrib.updated = false;
        other.updated   = false;
        raycast.updated = false;
    }

    struct {
        int updated = false;
        Clight lights[4];
        int lightnum = 0;
    }light;



    struct {
        int updated = false;
        LevelOfDist leve_of_dist;
    }lod;
    struct {
        int updated = false;
        TransferFunc tf_pts;
        int dim = 256;
    }tf;
    struct{
        int updated = false;
        int node_x_offset = 0;//in pixels
        int node_y_offset = 0;
        int world_row_count = 1;
        int world_col_count = 1;
        int node_x_index = 0;
        int node_y_index = 0;
    }distrib;
    struct{
        float ray_step = 0.f;
        float max_ray_dist = 0.f;
        float render_space = 1.f;
        int updated = false;
    }raycast;
    struct {
        int updated = false;
        int use_2d_tf = false;
        int gamma_correct = true;
        int output_depth = true;
        Float3 inv_tex_shape;
    }other;
};

// 每一帧需要更新的参数，主要是相机
struct PerFrameParams{
    Float3 cam_pos;
    float fov;
    Float3 cam_dir;
    float frame_width;
    Float3 cam_right;
    float frame_height;
    Float3 cam_up;
    float frame_w_over_h;
    Mat4 proj_view;
    int debug_mode = 1;
};
//static_assert(sizeof(PerFrameParams) == 32 * 4,"");

VISER_END