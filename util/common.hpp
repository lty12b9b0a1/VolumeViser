#pragma once


#define VUTIL_BEGIN namespace vutil{
#define VUTIL_END }



#if defined(_WIN32)
#define VISER_OS_WIN32
#elif defined(__linux)
#define VISER_OS_LINUX
#endif


#if defined(_MSC_VER)
#define VISER_CXX_MSVC
#elif defined(__clang__)
#define VISER_CXX_CLANG
#define VISER_CXX_IS_GNU
#elif defined(__GNUC__)
#define VISER_CXX_GCC
#define VISER_CXX_IS_GNU
#endif

#ifndef NDEBUG
#define VISER_DEBUG
#define VISER_WHEN_DEBUG(op) do { op; } while(false);
#else
#define VISER_WHEN_DEBUG(op) do { } while(false);
#endif
