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
    // 局部即时转置：构建绝对连续的 [LD_pool][bxyz] 内存结构
    // ========================================================================
    thread_local std::vector<double> vlbr3_T;
    thread_local std::vector<double> ylm_T;
    const int bxyz_local = this->bxyz; 
    const int total_size = bxyz_local * LD_pool;
    
    if (vlbr3_T.size() < total_size) {
        vlbr3_T.resize(total_size, 0.0);
        ylm_T.resize(total_size, 0.0);
    }

    // 纯二维转置，轨道在外，网格在内，保障内存连续写入速度
    for (int iorb = 0; iorb < LD_pool; ++iorb) {
        for (int ib = 0; ib < bxyz_local; ++ib) {
            vlbr3_T[iorb * bxyz_local + ib] = psir_vlbr3[ib][iorb];
            ylm_T[iorb * bxyz_local + ib]   = psir_ylm[ib][iorb];
        }
    }

    // ========================================================================
    // 暴力稠密计算区 (Dense Tensor Computation)
    // 放弃寻找稀疏边界，一切皆为连续的宏观矩阵运算
    // ========================================================================
    const char transa = 'T', transb = 'N'; 
    const double alpha = 1.0, beta = 1.0;
    
    // 你的绝杀 1：K 永远等于满编的 bxyz，没有任何截断
    const int K = bxyz_local; 
    // 你的绝杀 2：虽然 K 连满了，但换列的物理距离依然是轨道的长度 bxyz
    const int LDA = bxyz_local; 

    const int mcell_index = this->gridt->bcell_start[grid_index];
    for(int ia1 = 0; ia1 < na_grid; ++ia1)
    {
        const int bcell1 = mcell_index + ia1;
        const int iat1 = this->gridt->which_atom[bcell1];
        const int id1 = this->gridt->which_unitcell[bcell1];
        const ModuleBase::Vector3<int> r1 = this->gridt->get_ucell_coords(id1);
        
        // 提取寻址常数
        const int ia1_base_offset = block_index[ia1] * bxyz_local;

        for(int ia2 = 0; ia2 < na_grid; ++ia2)
        {
            const int bcell2 = mcell_index + ia2;
            const int iat2 = this->gridt->which_atom[bcell2];
            const int id2 = this->gridt->which_unitcell[bcell2];
            const ModuleBase::Vector3<int> r2 = this->gridt->get_ucell_coords(id2);

            // 物理对称性
            if(iat1 <= iat2)
            {
                // 我们不再遍历 cal_flag 去找边界！
                // 直接寻找全局哈密顿量矩阵指针
                const auto tmp_matrix = hR->find_matrix(iat1, iat2, r1-r2);
                
                // 周期性边界条件下，如果这两个原子完全没有相互作用，矩阵指针会为空
                if (tmp_matrix == nullptr) continue;
                
                const int m = tmp_matrix->get_row_size();
                const int n = tmp_matrix->get_col_size();

                // 获取目标矩阵的首地址
                const double* ptr_A = vlbr3_T.data() + block_index[ia2] * bxyz_local;
                const double* ptr_B = ylm_T.data()   + ia1_base_offset;

                // 纯粹的暴力计算：抛弃所有分支判断，把全长 bxyz 直接拍给 CPU！
                dgemm_(&transa, &transb, &n, &m, &K, &alpha,
                    ptr_A, &LDA, 
                    ptr_B, &LDA, 
                    &beta, tmp_matrix->get_pointer(), &n); 
            }
        }
    }
}