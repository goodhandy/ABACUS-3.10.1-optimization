#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "gint_k.h"
#include "module_basis/module_ao/ORB_read.h"
#include "grid_technique.h"
#include "module_base/ylm.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_base/blas_connector.h"
#include "module_base/timer.h"
#include "module_base/array_pool.h"
#include "module_base/vector3.h"
//#include <mkl_cblas.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __MKL
#include <mkl_service.h>
#endif


void Gint::cal_meshball_vlocal(
    const int na_grid, const int LD_pool, const int*const block_size, const int*const block_index,                
    const int grid_index, const bool*const*const cal_flag, const double*const*const psir_ylm,          
    const double*const*const psir_vlbr3, hamilt::HContainer<double>* hR)             
{
    // ========================================================================
    // 步骤一：全局转置底座 (AoS -> SoA) 与 内存池预分配
    // ========================================================================
    // 物理内存条，用于保证 dgemm_ 和 Gather 阶段的绝对连续访存 (Stride-1)
    thread_local std::vector<double> vlbr3_T;
    thread_local std::vector<double> ylm_T;
    
    // 1-vs-Many 的打包数据黑洞与结果接收池
    thread_local std::vector<double> B_pack; 
    thread_local std::vector<double> C_pack; 
    
    // 【极致微操】：元数据物流追踪器作为静态池，彻底消灭 for 循环内的 malloc/free 开销
    thread_local std::vector<int> valid_n_list;
    thread_local std::vector<double*> valid_hR_ptrs;

    const int bxyz_local = this->bxyz; 
    const int total_size = bxyz_local * LD_pool;
    
    // 安全扩容（仅在第一次调用或遭遇史无前例的大网格时触发）
    if (vlbr3_T.size() < total_size) {
        vlbr3_T.resize(total_size, 0.0);
        ylm_T.resize(total_size, 0.0);
        B_pack.resize(total_size, 0.0);
        C_pack.resize(LD_pool * LD_pool, 0.0); // 上限保证绝对安全
    }
    
    // 确保元数据池有足够的物理容量
    if (valid_n_list.capacity() < na_grid) {
        valid_n_list.reserve(na_grid);
        valid_hR_ptrs.reserve(na_grid);
    }

    // 执行连续转置，为后续所有操作铺平“高速公路”
    for (int iorb = 0; iorb < LD_pool; ++iorb) {
        double* __restrict__ dst_vlbr3 = &vlbr3_T[iorb * bxyz_local];
        double* __restrict__ dst_ylm   = &ylm_T[iorb * bxyz_local];
        
        #pragma omp simd
        for (int ib = 0; ib < bxyz_local; ++ib) {
            dst_vlbr3[ib] = psir_vlbr3[ib][iorb];
            dst_ylm[ib]   = psir_ylm[ib][iorb];
        }
    }

    // ========================================================================
    // 步骤二：1-vs-Many 异构流水线 (Gather -> Batched Compute -> Scatter)
    // ========================================================================
    const char transa = 'T', transb = 'N'; 
    const double alpha = 1.0, beta = 0.0; // beta=0: 覆盖 C_pack，省去 memset
    const int LDA = bxyz_local; 
    const int mcell_index = this->gridt->bcell_start[grid_index];

    // 以 ia1 为中心节点向外辐射
    for(int ia1 = 0; ia1 < na_grid; ++ia1)
    {
        const int bcell1 = mcell_index + ia1;
        const int iat1 = this->gridt->which_atom[bcell1];
        const int id1 = this->gridt->which_unitcell[bcell1];
        const ModuleBase::Vector3<int> r1 = this->gridt->get_ucell_coords(id1);
        
        const int m = block_size[ia1];
        const int ia1_base_offset = block_index[ia1] * bxyz_local;
        
        // 游标归零
        int n_pack = 0;
        
        // 【极致微操】：O(1) 复杂度清空运单数据，保留底层物理内存！
        valid_n_list.clear();
        valid_hR_ptrs.clear();

        // --------------------------------------------------------------------
        // 阶段 2.1: Gather (数据收集压缩，消除稀疏碎片)
        // --------------------------------------------------------------------
        for(int ia2 = 0; ia2 < na_grid; ++ia2)
        {
            const int bcell2 = mcell_index + ia2;
            const int iat2 = this->gridt->which_atom[bcell2];
            
            // 物理对称性截断，保住 50% 算力
            if(iat1 <= iat2)
            {
                // 快速碰撞检测
                bool has_overlap = false;
                for(int ib = 0; ib < bxyz_local; ++ib) {
                    if(cal_flag[ib][ia1] && cal_flag[ib][ia2]) { 
                        has_overlap = true; 
                        break; 
                    }
                }
                if(!has_overlap) continue; // 无交集，果断抛弃

                const int id2 = this->gridt->which_unitcell[bcell2];
                const ModuleBase::Vector3<int> r2 = this->gridt->get_ucell_coords(id2);
                auto tmp_matrix = hR->find_matrix(iat1, iat2, r1-r2);
                if (tmp_matrix == nullptr) continue;

                // 填写物流运单：记录该邻居的规模和目标写回地址
                const int n = tmp_matrix->get_col_size();
                valid_n_list.push_back(n);
                valid_hR_ptrs.push_back(tmp_matrix->get_pointer());

                // 【核心 Gather 指令】：将该邻居的 n 根轨道连续“吸”到 B_pack 末尾
                double* __restrict__ src = &vlbr3_T[block_index[ia2] * bxyz_local];
                double* __restrict__ dst = &B_pack[n_pack * bxyz_local];
                
                for(int iorb = 0; iorb < n; ++iorb) {
                    #pragma omp simd
                    for(int ib = 0; ib < bxyz_local; ++ib) {
                        dst[iorb * bxyz_local + ib] = src[iorb * bxyz_local + ib];
                    }
                }
                n_pack += n; // B_pack 游标后移
            }
        }

        // 如果该中心原子没有任何有效邻居（或全被 iat1 <= iat2 过滤），直接跳过
        if (n_pack == 0) continue;

        // --------------------------------------------------------------------
        // 阶段 2.2: Batched Compute (大满载降维打击)
        // --------------------------------------------------------------------
        // B_pack 此时是一个紧凑的 2D 连续内存块，包含了所有有效邻居的波函数。
        const double* ptr_A_pack = B_pack.data();
        const double* ptr_B_ia1  = ylm_T.data() + ia1_base_offset;

        // 一锤定音：一次性算完 ia1 所有的邻居，填满硬件的 SIMD 寄存器。
        // 数学原理：C_pack(n_pack x m) = B_pack^T(n_pack x bxyz_local) * A_ia1(bxyz_local x m)
        dgemm_(&transa, &transb, &n_pack, &m, &bxyz_local, &alpha,
               ptr_A_pack, &LDA, 
               ptr_B_ia1, &LDA, 
               &beta, C_pack.data(), &n_pack);

        // --------------------------------------------------------------------
        // 阶段 2.3: Scatter (结果精准切分与分发)
        // --------------------------------------------------------------------
        int current_n_offset = 0; // C_pack 中的横向切刀游标
        
        for(size_t idx = 0; idx < valid_n_list.size(); ++idx) {
            // 根据物流运单进行派送
            const int n = valid_n_list[idx];
            double* __restrict__ hR_ptr = valid_hR_ptrs[idx];
            
            for(int j = 0; j < m; ++j) {
                // src_row 定位到 C_pack 中属于该邻居的首地址
                const double* __restrict__ src_row = &C_pack[j * n_pack + current_n_offset];
                // dst_row 定位到全局离散 hR 矩阵的目标首地址
                double* __restrict__ dst_row = &hR_ptr[j * n];
                
                // 极限向量化累加写回
                #pragma omp simd
                for(int i = 0; i < n; ++i) {
                    dst_row[i] += src_row[i];
                }
            }
            current_n_offset += n; // 切刀向后移动 n 的宽度，准备切下一个邻居
        }
    }
}