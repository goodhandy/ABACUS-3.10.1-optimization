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
#include <vector> // 必须引入此头文件以使用 std::vector

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __MKL
#include <mkl_service.h>
#endif

void Gint::cal_meshball_vlocal(
    const int na_grid,                          // how many atoms on this (i,j,k) grid
    const int LD_pool,
    const int*const block_size,                 // block_size[na_grid], number of columns of a band
    const int*const block_index,                // block_index[na_grid+1], count total number of atomis orbitals
    const int grid_index,                       // index of grid group, for tracing global atom index
    const bool*const*const cal_flag,            // cal_flag[this->bxyz][na_grid], whether the atom-grid distance is larger than cutoff
    const double*const*const psir_ylm,          // psir_ylm[this->bxyz][LD_pool]
    const double*const*const psir_vlbr3,        // psir_vlbr3[this->bxyz][LD_pool]
    hamilt::HContainer<double>* hR)             // this->hRGint is the container of <phi_0 | V | phi_R> matrix element.
{
    const int mcell_index = this->gridt->bcell_start[grid_index];
    
    // 使用 thread_local 容器作为高速寄存器分块 (Register Blocking)
    // 保证内存生命周期持久，且无锁竞争，避免了频繁的系统调用 (malloc/free)
    thread_local std::vector<double> C_local;

    for(int ia1=0; ia1<na_grid; ++ia1)
    {
        const int bcell1 = mcell_index + ia1;
        const int iat1 = this->gridt->which_atom[bcell1];
        const int id1 = this->gridt->which_unitcell[bcell1];
        const ModuleBase::Vector3<int> r1 = this->gridt->get_ucell_coords(id1);

        for(int ia2=0; ia2<na_grid; ++ia2)
        {
            const int bcell2 = mcell_index + ia2;
            const int iat2= this->gridt->which_atom[bcell2];
            const int id2 = this->gridt->which_unitcell[bcell2];
            const ModuleBase::Vector3<int> r2 = this->gridt->get_ucell_coords(id2);

            if(iat1<=iat2)
            {
                // 1. 轻量级边界扫描：确定积分网格的上下界和真实有效点数
                int first_ib = -1;
                int last_ib = -1;
                int valid_k = 0;
                
                for(int ib=0; ib < this->bxyz; ++ib)
                {
                    if(cal_flag[ib][ia1] && cal_flag[ib][ia2])
                    {
                        if (first_ib == -1) first_ib = ib;
                        last_ib = ib + 1;
                        valid_k++;
                    }
                }

                if (valid_k == 0) continue;
                const int ib_length = last_ib - first_ib;

                const auto tmp_matrix = hR->find_matrix(iat1, iat2, r1-r2);
                if (tmp_matrix == nullptr) continue;
                
                const int m = tmp_matrix->get_row_size();
                const int n = tmp_matrix->get_col_size();
                const int offset_ia2 = block_index[ia2];
                const int offset_ia1 = block_index[ia1];

                // 2. 准备本地栈/L1缓存累加器
                const int matrix_size = n * m;
                if (C_local.size() < matrix_size) {
                    C_local.resize(matrix_size);
                }
                // 初始化为 0，这一步在 CPU L1 缓存中执行，速度极快
                for(int i=0; i<matrix_size; ++i) C_local[i] = 0.0;
                double* C_ptr = C_local.data();

                // ------------------------------------------------------------------------
                // 3. 核心算法分流：针对微小矩阵(如13x13)的微内核计算，彻底抛弃 BLAS 启动开销
                // ------------------------------------------------------------------------
                
                if (valid_k == ib_length) 
                {
                    // 算法 A：100% 稠密高速通道 (Zero-Branch Fast-Path)
                    // 日志显示有大量完全重叠的区域，此分支去除了内部的 if 判断，彻底释放指令流水线
                    for(int ib = first_ib; ib < last_ib; ++ib)
                    {
                        const double* ptr_B = &psir_ylm[ib][offset_ia1];
                        const double* ptr_A = &psir_vlbr3[ib][offset_ia2];
                        
                        // 强制展开循环以适配 ARM SIMD (NEON/SVE) 指令
                        #pragma clang loop unroll(enable)
                        for(int j = 0; j < m; ++j) 
                        {
                            const double b_val = ptr_B[j];
                            #pragma clang loop vectorize(enable)
                            for(int i = 0; i < n; ++i) 
                            {
                                // C 矩阵的列主序累加：物理内存绝对连续
                                C_ptr[j * n + i] += ptr_A[i] * b_val;
                            }
                        }
                    }
                }
                else 
                {
                    // 算法 B：稀疏微内核 (Sparse Micro-Kernel)
                    // 带有 cal_flag 掩码，保留零拷贝(Zero-copy)，拒绝多余的内存 Write 动作
                    for(int ib = first_ib; ib < last_ib; ++ib)
                    {
                        if(cal_flag[ib][ia1] && cal_flag[ib][ia2])
                        {
                            const double* ptr_B = &psir_ylm[ib][offset_ia1];
                            const double* ptr_A = &psir_vlbr3[ib][offset_ia2];
                            
                            #pragma clang loop unroll(enable)
                            for(int j = 0; j < m; ++j) 
                            {
                                const double b_val = ptr_B[j];
                                #pragma clang loop vectorize(enable)
                                for(int i = 0; i < n; ++i) 
                                {
                                    C_ptr[j * n + i] += ptr_A[i] * b_val;
                                }
                            }
                        }
                    }
                }

                // ------------------------------------------------------------------------
                // 4. 将本地缓存中的结果一次性合并到全局堆内存中
                // 这模拟了原本 dgemm_ 内部 beta=1.0 的行为，但大幅降低了全局内存的读写频次
                // ------------------------------------------------------------------------
                double* global_C = tmp_matrix->get_pointer();
                #pragma clang loop vectorize(enable)
                for(int idx = 0; idx < matrix_size; ++idx)
                {
                    global_C[idx] += C_ptr[idx];
                }
            }
        }
    }
}