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
    // 局部即时转置 
    // 在进入原子循环前，将 [bxyz][LD_pool] 重排为的 [LD_pool][bxyz]
    // ========================================================================
    thread_local std::vector<double> vlbr3_T;
    thread_local std::vector<double> ylm_T;
    const int total_size = this->bxyz * LD_pool;
    if (vlbr3_T.size() < total_size) {
        vlbr3_T.resize(total_size);
        ylm_T.resize(total_size);
    }

    
    for (int ib = 0; ib < this->bxyz; ++ib) {
        for (int iorb = 0; iorb < LD_pool; ++iorb) {
            vlbr3_T[iorb * this->bxyz + ib] = psir_vlbr3[ib][iorb];
            ylm_T[iorb * this->bxyz + ib]   = psir_ylm[ib][iorb];
        }
    }
    // ========================================================================

    // 正如你推演的，因为现在送给 Fortran 的内存变成了转置结构
    // 物理公式 C = B^T * A，在 Fortran 视角的 C_F = A_F_new^T * B_F_new 中
    // 我们必须将参数对调：transa='T', transb='N'
    const char transa='T', transb='N'; 
    const double alpha=1, beta=1;
    // 你的绝杀：跨步步长现在绝对等于 bxyz！
    const int LDA = this->bxyz; 

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
            const int iat2= this->gridt->which_atom[bcell2];
            const int id2 = this->gridt->which_unitcell[bcell2];
            const ModuleBase::Vector3<int> r2 = this->gridt->get_ucell_coords(id2);

            if(iat1<=iat2)
            {
                // 找出网格有效边界
                int first_ib=0, last_ib=0;
                for(int ib=0; ib<this->bxyz; ++ib) {
                    if(cal_flag[ib][ia1] && cal_flag[ib][ia2]) { first_ib=ib; break; }
                }
                for(int ib=this->bxyz-1; ib>=0; --ib) {
                    if(cal_flag[ib][ia1] && cal_flag[ib][ia2]) { last_ib=ib+1; break; }
                }
                const int ib_length = last_ib-first_ib;
                if(ib_length<=0) continue;

                const auto tmp_matrix = hR->find_matrix(iat1, iat2, r1-r2);
                if (tmp_matrix == nullptr) continue;
                const int m = tmp_matrix->get_row_size();
                const int n = tmp_matrix->get_col_size();

                // 获取原子在 [LD_pool][bxyz] 新结构中的首地址指针
                // 注意：现在的基址偏移是 轨道索引(block_index) * 轨道长度(bxyz) + 网格起始点(first_ib)
                const double* ptr_A = &vlbr3_T[block_index[ia2] * this->bxyz + first_ib];
                const double* ptr_B = &ylm_T[block_index[ia1] * this->bxyz + first_ib];

                // 为了发挥 0 间隙读取的最大威力，我们废弃稀疏分支，统一采用密集计算
                // 直接向 dgemm_ 灌入这片完美连续的内存！
                dgemm_(&transa, &transb, &n, &m, &ib_length, &alpha,
                    ptr_A, &LDA, 
                    ptr_B, &LDA, 
                    &beta, tmp_matrix->get_pointer(), &n); 
            }
        }
    }
}