#pragma once

#include "context.hpp"


CUB_BEGIN

    template<typename T, int N>
    class cu_array;

    namespace detail{
        template<typename T>
        struct cu_array_format_traits;

#define CU_ARRAY_FORMAT_TRAITS(type, _fmt, _num) \
        template<>\
        struct cu_array_format_traits<type>{\
            static constexpr auto fmt = _fmt;\
            static constexpr int num = _num;\
        };

        CU_ARRAY_FORMAT_TRAITS(uint8_t, CU_AD_FORMAT_UNSIGNED_INT8, 1)

        CU_ARRAY_FORMAT_TRAITS(uchar1, CU_AD_FORMAT_UNSIGNED_INT8, 1)

        CU_ARRAY_FORMAT_TRAITS(ushort1, CU_AD_FORMAT_UNSIGNED_INT16, 1)

        CU_ARRAY_FORMAT_TRAITS(uint1, CU_AD_FORMAT_UNSIGNED_INT32, 1)

        CU_ARRAY_FORMAT_TRAITS(float1, CU_AD_FORMAT_FLOAT, 1)

        CU_ARRAY_FORMAT_TRAITS(float, CU_AD_FORMAT_FLOAT, 1)

        CU_ARRAY_FORMAT_TRAITS(uchar4, CU_AD_FORMAT_UNSIGNED_INT8, 4)

        CU_ARRAY_FORMAT_TRAITS(float4, CU_AD_FORMAT_FLOAT, 4)


        template<typename T, int N>
        class cu_array_base{
        public:

            cu_context get_context() const {
                return ctx;
            }
            auto get_handle() const {
                return array;
            }

            virtual ~cu_array_base(){
                CUB_CHECK(cuArrayDestroy(array));
            }
        protected:
            CUarray array;
            cu_context ctx;
        };
    }

    template<typename E>
    class cu_array<E, 1> : public detail::cu_array_base<E, 1>{
    public:
        cu_array(cu_context ctx, size_t len)
        :l(len)
        {
            this->ctx = ctx;

            CUDA_ARRAY_DESCRIPTOR desc;
            std::memset(&desc, 0, sizeof(desc));
            desc.Width = len;
            desc.Height = 1;
            desc.Format = detail::cu_array_format_traits<E>::fmt;
            desc.NumChannels = detail::cu_array_format_traits<E>::num;
            CUB_CHECK(cuArrayCreate(&this->array, &desc));
        }
        size_t size() const{
            return l;
        }
    private:
        size_t l;
    };

    template<typename E>
    class cu_array<E, 2> : public detail::cu_array_base<E, 2>{
    public:
        cu_array(cu_context ctx, size_t width, size_t height)
        :w(width), h(height)
        {
            this->ctx = ctx;

            CUDA_ARRAY_DESCRIPTOR desc;
            std::memset(&desc, 0, sizeof(desc));
            desc.Width = width;
            desc.Height = height;
            desc.Format = detail::cu_array_format_traits<E>::fmt;
            desc.NumChannels = detail::cu_array_format_traits<E>::num;
            CUB_CHECK(cuArrayCreate(&this->array, &desc));
        }
        size_t width() const{
            return w;
        }
        size_t height() const{
            return h;
        }
    private:
        size_t w, h;
    };

    enum cu_array3d_type{
        ARRAY3D,
        LAYERED,
        CUBEMAP
    };

    template<typename E>
    class cu_array<E, 3> : public detail::cu_array_base<E, 3>{
    public:
        cu_array(cu_context ctx, const cu_extent& extent, cu_array3d_type type = ARRAY3D)
        : extent(extent)
        {
            this->ctx = ctx;

            CUDA_ARRAY3D_DESCRIPTOR desc;
            std::memset(&desc, 0, sizeof(desc));
            desc.Format = detail::cu_array_format_traits<E>::fmt;
            desc.NumChannels = detail::cu_array_format_traits<E>::num;
            desc.Width = extent.width;
            desc.Height = extent.height;
            desc.Depth = extent.depth;
            if(type == LAYERED)
                desc.Flags = CUDA_ARRAY3D_LAYERED;
            else if(type == CUBEMAP)
                desc.Flags = CUDA_ARRAY3D_CUBEMAP;
            CUB_CHECK(cuArray3DCreate(&this->array, &desc));
        }
        cu_extent get_extent() const{
            return extent;
        }
    private:
        cu_extent extent;
    };

    template<typename T, int N, typename... Args>
    cu_array<T, N> cu_context::alloc_array(Args &&... args) const {
        return cu_array<T, N>(*this, std::forward<Args>(args)...);
    }

CUB_END