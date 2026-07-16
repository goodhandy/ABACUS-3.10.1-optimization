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
#include <vector>

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
    const char transa = 'N', transb = 'T';
    const double alpha = 1.0, beta = 1.0;
    
    // ------------------------------------------------------------------------
    // [优化点 4] 内存分配外置：找出当前网格上最大的 block_size，以此分配打包 Buffer
    // ------------------------------------------------------------------------
    int max_block_size = 0;
    for (int ia = 0; ia < na_grid; ++ia) {
        if (block_size[ia] > max_block_size) {
            max_block_size = block_size[ia];
        }
    }
    
    // 使用 std::vector 在堆上预分配连续内存，避免栈溢出。
    // 容量 = 网格点总数(bxyz) * 最大可能的轨道数(max_block_size)
    std::vector<double> pack_vlbr3(this->bxyz * max_block_size);
    std::vector<double> pack_ylm(this->bxyz * max_block_size);
    // ------------------------------------------------------------------------

    const int mcell_index = this->gridt->bcell_start[grid_index];
    for(int ia1=0; ia1<na_grid; ++ia1)
    {
        const int bcell1 = mcell_index + ia1;
        const int iat1 = this->gridt->which_atom[bcell1];
        const int id1 = this->gridt->which_unitcell[bcell1];
        const ModuleBase::Vector3<int> r1 = this->gridt->get_ucell_coords(id1);

        for(int ia2=0; ia2<na_grid; ++ia2)
        {
            const int bcell2 = mcell_index + ia2;
            const int iat2 = this->gridt->which_atom[bcell2];
            const int id2 = this->gridt->which_unitcell[bcell2];
            const ModuleBase::Vector3<int> r2 = this->gridt->get_ucell_coords(id2);

            if(iat1 <= iat2)
            {
                const auto tmp_matrix = hR->find_matrix(iat1, iat2, r1-r2);
                if (tmp_matrix == nullptr)
                {
                    continue;
                }
                
                const int m = tmp_matrix->get_row_size();
                const int n = tmp_matrix->get_col_size();
                int valid_k = 0; // 记录真实参与计算的网格点数量
                
                // 为了获取更好的数据局部性，提前获取指针偏移量
                const int offset_ia2 = block_index[ia2];
                const int offset_ia1 = block_index[ia1];

                // ------------------------------------------------------------------------
                // Gather 过程：单次遍历，将有效数据打包至连续内存中
                // 结合编译器指令强制循环向量化展开，可高效生成内存加载/存储指令
                // ------------------------------------------------------------------------
                for(int ib=0; ib < this->bxyz; ++ib)
                {
                    if(cal_flag[ib][ia1] && cal_flag[ib][ia2])
                    {
                        const double* src_vlbr3 = &psir_vlbr3[ib][offset_ia2];
                        double* dst_vlbr3 = &pack_vlbr3[valid_k * n];
                        
                        #pragma clang loop vectorize(enable) interleave(enable)
                        for(int i = 0; i < n; ++i) {
                            dst_vlbr3[i] = src_vlbr3[i];
                        }

                        const double* src_ylm = &psir_ylm[ib][offset_ia1];
                        double* dst_ylm = &pack_ylm[valid_k * m];
                        
                        #pragma clang loop vectorize(enable) interleave(enable)
                        for(int j = 0; j < m; ++j) {
                            dst_ylm[j] = src_ylm[j];
                        }
                        
                        valid_k++;
                    }
                }

                // ------------------------------------------------------------------------
                // [优化点 3] 稠密计算：一次调用，充分利用底层数学库
                // 此时 LDA = n, LDB = m，消除了原先依赖 LD_pool 导致的大步长跨距
                // ------------------------------------------------------------------------
                if(valid_k > 0)
                {
                    dgemm_(&transa, &transb, &n, &m, &valid_k, &alpha,
                        pack_vlbr3.data(), &n,          // LDA = n
                        pack_ylm.data(), &m,            // LDB = m
                        &beta, tmp_matrix->get_pointer(), &n); 
                }
            }
        }
    }
}