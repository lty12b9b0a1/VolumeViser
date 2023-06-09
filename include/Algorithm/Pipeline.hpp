#pragma once

#include <Common/Common.hpp>

// SWC文件生成Mesh文件
// 1.加载SWC文件到内存
// 2.计算SWC涉及到区域，采用DDA算法，两个节点之间DDA，对于每个点，即填充一个圆，已知圆的方程，求出半圆
// 边上的点，然后逐行进行DDA填充圆，以两点之间形成的圆柱体积作为每个任务的cost，
// 不用预先和八叉树求交计算出可能相交的数据块/节点，GPU上将SWC根据长度分割进行计算，
// 每一段根据上述算法计算出需要填充的体素，然后体素根据其坐标计算出数据块信息，再去查询数据块真实的存储位置，
// 如果没有存在，那么新分配这一块数据相关的资源，否则直接填充更新即可。
// 3.对分块的体数据进行Marching Cube算法，先遍历一边分块体数据，生成顶点数据，然后实际的Marching Cube
// 算法只需要生成三角形的边索引，添加额外边缘信息，分块后的Mesh暂时不进行融合拼接，只在最后保存导出的时候
// 或者Mesh整体简化前进行。
// 4.对得到的Mesh进行光滑，锁定分块边缘的边不改变
// 5.对得到的Mesh进行化简，暂时不清楚可不可以锁定边缘的边进行化简以及化简的速度到底是怎么样的，如果速度可以，
// 可以添加实时简化的选项，如果速度较慢，那么只有导出前简化，
// 由于可能局部修改，那么只需要局部重构网格，如果简化可以局部的话

//预先估计SWC涉及到区域的大小，或者实现把其分割为若干段

VISER_BEGIN




VISER_END




