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
    // 步骤一：局部即时转置 (AoS -> SoA)
    // 目标结构：[LD_pool][bxyz]
    // ========================================================================
    thread_local std::vector<double> vlbr3_T;
    thread_local std::vector<double> ylm_T;
    // 分配一个用于存放全局宏观大矩阵结果的私有池 C_global
    thread_local std::vector<double> C_global; 
    
    const int bxyz_local = this->bxyz; 
    const int total_size = bxyz_local * LD_pool;
    
    // 内存池分配
    if (vlbr3_T.size() < total_size) {
        vlbr3_T.resize(total_size, 0.0);
        ylm_T.resize(total_size, 0.0);
    }
    if (C_global.size() < LD_pool * LD_pool) {
        C_global.resize(LD_pool * LD_pool, 0.0);
    }

    // 【极致优化】：外层为轨道 iorb，内层为网格 ib。
    // 这样不仅消除了内层循环复杂的乘法寻址，还保证了目标数组绝对的连续写入！
    for (int iorb = 0; iorb < LD_pool; ++iorb) {
        // 提前算出本轨道的写入基址
        double* __restrict__ dst_vlbr3 = &vlbr3_T[iorb * bxyz_local];
        double* __restrict__ dst_ylm   = &ylm_T[iorb * bxyz_local];

        // 现代编译器会将这个纯净的内层循环直接展开为 SIMD 向量拷贝指令
        #pragma omp simd
        for (int ib = 0; ib < bxyz_local; ++ib) {
            dst_vlbr3[ib] = psir_vlbr3[ib][iorb];
            dst_ylm[ib]   = psir_ylm[ib][iorb];
        }
    }

    // ========================================================================
    // 步骤二：Macro-GEMM (宏观全局大矩阵整合乘法)
    // 前提保障：上游数据中的无效区域已被安全补零 (Zero-Padded)
    // ========================================================================
    const char transa = 'T', transb = 'N'; 
    const double alpha = 1.0, beta = 0.0; // beta=0 意为覆盖 C_global 旧数据
    const int LDA = bxyz_local; 
    
    // 暴力满载！一次性计算该网格内所有轨道的所有相互作用，流水线全开
    dgemm_(&transa, &transb, &LD_pool, &LD_pool, &bxyz_local, &alpha,
           vlbr3_T.data(), &LDA, 
           ylm_T.data(), &LDA, 
           &beta, C_global.data(), &LD_pool);

    // ========================================================================
    // 步骤三：掩码收集与写回 (Masked Scatter)
    // 根据物理规则提取有效分块，丢弃其余的零填充计算残余
    // ========================================================================
    const int mcell_index = this->gridt->bcell_start[grid_index];
    for(int ia1 = 0; ia1 < na_grid; ++ia1)
    {
        const int bcell1 = mcell_index + ia1;
        const int iat1 = this->gridt->which_atom[bcell1];
        const int id1 = this->gridt->which_unitcell[bcell1];
        const ModuleBase::Vector3<int> r1 = this->gridt->get_ucell_coords(id1);
        
        // C_global 中原子 1 的行起始偏移
        const int offset_ia1 = block_index[ia1]; 
        const int m = block_size[ia1];

        for(int ia2 = 0; ia2 < na_grid; ++ia2)
        {
            const int bcell2 = mcell_index + ia2;
            const int iat2 = this->gridt->which_atom[bcell2];
            const int id2 = this->gridt->which_unitcell[bcell2];
            const ModuleBase::Vector3<int> r2 = this->gridt->get_ucell_coords(id2);

            // 谓词掩码：物理对称性要求
            if(iat1 <= iat2)
            {
                // 谓词掩码：快速验证原子间是否存在实际的网格重叠
                bool has_overlap = false;
                for(int ib = 0; ib < bxyz_local; ++ib) {
                    if(cal_flag[ib][ia1] && cal_flag[ib][ia2]) { 
                        has_overlap = true; 
                        break; 
                    }
                }
                
                // 如果没有交集，直接过滤，省去内存提取开销
                if(!has_overlap) continue;
                
                const auto tmp_matrix = hR->find_matrix(iat1, iat2, r1-r2);
                if (tmp_matrix == nullptr) continue;
                
                const int n = block_size[ia2];
                const int offset_ia2 = block_index[ia2];
                double* __restrict__ global_ptr = tmp_matrix->get_pointer();

                // 【极致优化】：利用局部指针对提取写入过程进行加速
                for (int j = 0; j < m; ++j) {
                    // 源大矩阵中，第 j 行的起始地址
                    const double* __restrict__ src_row = &C_global[(offset_ia1 + j) * LD_pool + offset_ia2];
                    // 目标小矩阵中，第 j 行的起始地址
                    double* __restrict__ dst_row = &global_ptr[j * n];
                    
                    // 无分支的一维紧凑拷贝累加，将被完美向量化
                    #pragma omp simd
                    for (int i = 0; i < n; ++i) {
                        dst_row[i] += src_row[i];
                    }
                }
            }
        }
    }
}