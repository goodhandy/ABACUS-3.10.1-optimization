#include "gint.h"
#include "module_base/memory.h"
#include "module_parameter/parameter.h"
#include "module_base/timer.h"
#include "module_base/array_pool.h"

#include <chrono>
#include <cstddef>

namespace
{

/**
 * @brief 融合局域势缩放与转置，直接生成收缩所需布局。
 *
 * 原流程先生成：
 *
 *     psir_vlbr3[ib][iorb]
 *         = vldr3[ib] * psir_source[ib][iorb]
 *
 * 随后 cal_meshball_vlocal 再将其转置为：
 *
 *     psir_vlbr3_T[iorb][ib]
 *
 * 本函数把两个步骤融合为一次遍历，直接生成：
 *
 *     psir_vlbr3_T[iorb][ib]
 *         = vldr3[ib] * psir_source[ib][iorb]
 *
 * 输出逻辑形状为 [LD_pool][bxyz]。
 * 对 cal_flag 为 false 的原子—网格点组合显式写0，以保持稠密路径
 * 对首尾区间内空洞进行计算时的数值语义。
 */
inline ModuleBase::Array_Pool<double>
build_psir_vlbr3_transposed(
    const int bxyz,
    const int na_grid,
    const int LD_pool,
    const int* const block_index,
    const bool* const* const cal_flag,
    const double* const vldr3,
    const double* const* const psir_source)
{
    ModuleBase::Array_Pool<double> psir_vlbr3_T(
        LD_pool,
        bxyz);

    double** const dst
        = psir_vlbr3_T.get_ptr_2D();

    /*
     * ib 为外层循环，使 psir_source[ib] 沿轨道方向连续读取。
     * 每个原子轨道块只判断一次 cal_flag，避免对每条轨道重复判断。
     */
    for (int ib = 0; ib < bxyz; ++ib)
    {
        const double* const src_row
            = psir_source[ib];

        const double potential_factor
            = vldr3[ib];

        for (int ia = 0; ia < na_grid; ++ia)
        {
            const int orbital_begin
                = block_index[ia];

            const int orbital_end
                = block_index[ia + 1];

            if (cal_flag[ib][ia])
            {
                for (int iorb = orbital_begin;
                     iorb < orbital_end;
                     ++iorb)
                {
                    dst[iorb][ib]
                        = potential_factor
                        * src_row[iorb];
                }
            }
            else
            {
                for (int iorb = orbital_begin;
                     iorb < orbital_end;
                     ++iorb)
                {
                    dst[iorb][ib] = 0.0;
                }
            }
        }
    }

    return psir_vlbr3_T;
}

/**
 * @brief 将普通 [bxyz][LD_pool] 矩阵转置为 [LD_pool][bxyz]。
 *
 * 该辅助函数只用于保持 dvlocal 调用的原有数学方向：
 * dvlocal 中第二个输入是 dpsir_ylm，而不是 psir_vlbr3，
 * 因此需要在调用 cal_meshball_vlocal 前把该第二输入转置。
 */
inline ModuleBase::Array_Pool<double>
transpose_matrix_to_pool(
    const int bxyz,
    const int LD_pool,
    const double* const* const src)
{
    ModuleBase::Array_Pool<double> dst_pool(
        LD_pool,
        bxyz);

    double** const dst
        = dst_pool.get_ptr_2D();

    for (int ib = 0; ib < bxyz; ++ib)
    {
        const double* const src_row = src[ib];

        for (int iorb = 0; iorb < LD_pool; ++iorb)
        {
            dst[iorb][ib] = src_row[iorb];
        }
    }

    return dst_pool;
}

} // 匿名命名空间

void Gint::gint_kernel_vlocal(Gint_inout* inout) {
    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal");

    ModuleBase::TITLE("Handy_Test", "test_cal_gint_vlocal");
    ModuleBase::timer::tick("Handy_Test", "test_cal_gint_vlocal");

    const UnitCell& ucell = *this->ucell;
    const int max_size = this->gridt->max_atom;
    const int lgd = this->gridt->lgd;
    const int ncyz = this->ny * this->nplane;
    const double dv = ucell.omega / this->ncxyz;
    const double delta_r = this->gridt->dr_uniform;
    hamilt::HContainer<double>* hRGint_kernel = PARAM.inp.nspin != 4 ? this->hRGint : this->hRGint_tmp[inout->ispin];
    hRGint_kernel->set_zero();

    // ========================
    // 在外部定义全局总时间累加器
    // ========================
    double total_time_prep = 0.0;
    double total_time_psir = 0.0;
    double total_time_vlbr3 = 0.0;     // 记录 psir_vlbr3 融合生成耗时
    double total_time_meshball = 0.0;  // 仅记录 cal_meshball_vlocal 耗时

#pragma omp parallel 
    {   /**
        * @brief When in OpenMP, it points to a newly allocated memory,
        */

        // ===================
        // 线程私有的时间累加器 
        // ===================
        double thread_time_prep = 0.0;     // 记录数据准备、场映射与遮罩的时间
        double thread_time_psir = 0.0;     // 记录轨道数值计算的时间
        double thread_time_vlbr3 = 0.0;    // 记录局域势缩放与转置融合时间
        double thread_time_meshball = 0.0; 
        
        // 更换为 C++11 原生高精度单调时钟时间点变量
        std::chrono::steady_clock::time_point t_start, t_end;

        hamilt::HContainer<double> hRGint_thread(*hRGint_kernel);
        std::vector<int> block_iw(max_size,0);
        std::vector<int> block_index(max_size+1,0);
        std::vector<int> block_size(max_size,0);
        std::vector<double> vldr3(this->bxyz,0.0);
        
        #pragma omp for schedule(dynamic)
        for (int grid_index = 0; grid_index < this->nbxx; grid_index++) {
            const int na_grid = this->gridt->how_many_atoms[grid_index];
            if (na_grid == 0) {
                continue;
            }
            /**
             * @brief Prepare block information
            */

            // ---------------------------------------------------------
            // 探针 A: 测量准备阶段 (内存分配 + 场映射 + 建立稀疏遮罩)
            // ---------------------------------------------------------
            t_start = std::chrono::steady_clock::now();

            ModuleBase::Array_Pool<bool> cal_flag(this->bxyz,max_size);

            Gint_Tools::get_gint_vldr3(vldr3.data(),
                                        inout->vl,
                                        this->bxyz,
                                        this->bx,
                                        this->by,
                                        this->bz,
                                        this->nplane,
                                        this->gridt->start_ind[grid_index],
                                        ncyz,
                                        dv);

            Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, 
                                        block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());

            /**
             * @brief Evaluate psi and dpsi on grids
            */
            const int LD_pool = block_index[na_grid];
            ModuleBase::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
            
            t_end = std::chrono::steady_clock::now();
            // 计算时间差并累加（单位：秒）
            thread_time_prep += std::chrono::duration<double>(t_end - t_start).count();

            // ---------------------------------------------------------
            // 探针 B: 测量轨道解析阶段 (计算波函数和球谐函数)
            // ---------------------------------------------------------
            t_start = std::chrono::steady_clock::now();

            Gint_Tools::cal_psir_ylm(*this->gridt, 
            this->bxyz, na_grid, grid_index, delta_r,
            block_index.data(), block_size.data(), 
            cal_flag.get_ptr_2D(),psir_ylm.get_ptr_2D());

            const ModuleBase::Array_Pool<double> &psir_ylm_1 = (!this->psir_func_1) ? psir_ylm : this->psir_func_1(psir_ylm, *this->gridt, grid_index, 0, block_iw, block_size, block_index, cal_flag);
            const ModuleBase::Array_Pool<double> &psir_ylm_2 = (!this->psir_func_2) ? psir_ylm : this->psir_func_2(psir_ylm, *this->gridt, grid_index, 0, block_iw, block_size, block_index, cal_flag);
        
            t_end = std::chrono::steady_clock::now();
            thread_time_psir += std::chrono::duration<double>(t_end - t_start).count();

            // ---------------------------------------------------------
            // 探针 C1: 测量局域势缩放与转置融合阶段
            // ---------------------------------------------------------
            t_start = std::chrono::steady_clock::now();

            const ModuleBase::Array_Pool<double> psir_vlbr3_T
                = build_psir_vlbr3_transposed(
                    this->bxyz,
                    na_grid,
                    LD_pool,
                    block_index.data(),
                    cal_flag.get_ptr_2D(),
                    vldr3.data(),
                    psir_ylm_1.get_ptr_2D());

            t_end = std::chrono::steady_clock::now();
            thread_time_vlbr3 += std::chrono::duration<double>(t_end - t_start).count();
            
            // ---------------------------------------------------------
            // 探针 C2: 测量张量收缩/底层矩阵乘加阶段 (cal_meshball_vlocal)
            // ---------------------------------------------------------            
            t_start = std::chrono::steady_clock::now();    
            
            ModuleBase::timer::tick("Handy_Test", "cal_meshball_vlocal");
            this->cal_meshball_vlocal(
                na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, 
                cal_flag.get_ptr_2D(), psir_ylm.get_ptr_2D(), psir_vlbr3_T.get_ptr_2D(),
                &hRGint_thread);
            ModuleBase::timer::tick("Handy_Test", "cal_meshball_vlocal");

            t_end = std::chrono::steady_clock::now();
            thread_time_meshball += std::chrono::duration<double>(t_end - t_start).count();        
        }

    #pragma omp critical
        {
            BlasConnector::axpy(hRGint_thread.get_nnr(),
                                1.0,
                                hRGint_thread.get_wrapper(),
                                1,
                                hRGint_kernel->get_wrapper(),
                                1);

            // 将各个线程私有的计时结果，安全地累加到全局总时间
            total_time_prep += thread_time_prep;
            total_time_psir += thread_time_psir;
            total_time_vlbr3 += thread_time_vlbr3;
            total_time_meshball += thread_time_meshball;                    
        }
    }

    // 格式化输出 4 个探针的结果
    printf("  [Total CPU Core-Time] prep: %8.4f s | psir: %8.4f s | vlbr3: %8.4f s | meshball: %8.4f s\n", 
           total_time_prep, total_time_psir, total_time_vlbr3, total_time_meshball);

    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal");

    ModuleBase::TITLE("Handy_Test", "test_cal_gint_vlocal");
    ModuleBase::timer::tick("Handy_Test", "test_cal_gint_vlocal");
}

void Gint::gint_kernel_dvlocal(Gint_inout* inout) {
    ModuleBase::TITLE("Gint_interface", "cal_gint_dvlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_dvlocal");
    const UnitCell& ucell = *this->ucell;
    const int max_size = this->gridt->max_atom;
    const int lgd = this->gridt->lgd;
    const int nnrg = pvdpRx_reduced[inout->ispin].get_nnr();
    const int ncyz = this->ny * this->nplane;
    const double dv = ucell.omega / this->ncxyz;
    const double delta_r = this->gridt->dr_uniform;

    if (PARAM.globalv.gamma_only_local) {
        ModuleBase::WARNING_QUIT("Gint_interface::cal_gint","dvlocal only for k point!");
    }
    pvdpRx_reduced[inout->ispin].set_zero();
    pvdpRy_reduced[inout->ispin].set_zero();
    pvdpRz_reduced[inout->ispin].set_zero();

#pragma omp parallel 
{
    hamilt::HContainer<double> pvdpRx_thread(pvdpRx_reduced[inout->ispin]);
    hamilt::HContainer<double> pvdpRy_thread(pvdpRy_reduced[inout->ispin]);
    hamilt::HContainer<double> pvdpRz_thread(pvdpRz_reduced[inout->ispin]);
    std::vector<int> block_iw(max_size,0);
    std::vector<int> block_index(max_size+1,0);
    std::vector<int> block_size(max_size,0);
    std::vector<double> vldr3(this->bxyz,0.0);
#pragma omp for schedule(dynamic)
    for (int grid_index = 0; grid_index < this->nbxx; grid_index++) {
        const int na_grid = this->gridt->how_many_atoms[grid_index];
        if (na_grid == 0) {
            continue;
        }
        Gint_Tools::get_gint_vldr3(vldr3.data(),
                                    inout->vl,
                                    this->bxyz,
                                    this->bx,
                                    this->by,
                                    this->bz,
                                    this->nplane,
                                    this->gridt->start_ind[grid_index],
                                    ncyz,
                                    dv);
    //prepare block information
        ModuleBase::Array_Pool<bool> cal_flag(this->bxyz,max_size);
        Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, 
                                    block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());
        
	//evaluate psi and dpsi on grids
        const int LD_pool = block_index[na_grid];

        ModuleBase::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_x(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_y(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_z(this->bxyz, LD_pool);
        Gint_Tools::cal_dpsir_ylm(*this->gridt, this->bxyz, na_grid, grid_index, delta_r, 
                                    block_index.data(), block_size.data(), cal_flag.get_ptr_2D(),psir_ylm.get_ptr_2D(),
                                    dpsir_ylm_x.get_ptr_2D(), dpsir_ylm_y.get_ptr_2D(), dpsir_ylm_z.get_ptr_2D());

	//calculating f_mu(r) = v(r)*psi_mu(r)*dv
        /*
         * dvlocal 的原调用方向与 vlocal 不同：
         * 第一个输入是 psir_vlbr3，第二个输入是 dpsir_ylm。
         *
         * 为保持原数学方向，这里暂时保留原布局 psir_vlbr3，
         * 并只把三个第二输入 dpsir_ylm 预先转置。
         * 当前融合实验的性能测试目标仍是 gint_kernel_vlocal。
         */
        const ModuleBase::Array_Pool<double> psir_vlbr3
            = Gint_Tools::get_psir_vlbr3(
                this->bxyz,
                na_grid,
                LD_pool,
                block_index.data(),
                cal_flag.get_ptr_2D(),
                vldr3.data(),
                psir_ylm.get_ptr_2D());

        const ModuleBase::Array_Pool<double> dpsir_ylm_x_T
            = transpose_matrix_to_pool(
                this->bxyz,
                LD_pool,
                dpsir_ylm_x.get_ptr_2D());

        const ModuleBase::Array_Pool<double> dpsir_ylm_y_T
            = transpose_matrix_to_pool(
                this->bxyz,
                LD_pool,
                dpsir_ylm_y.get_ptr_2D());

        const ModuleBase::Array_Pool<double> dpsir_ylm_z_T
            = transpose_matrix_to_pool(
                this->bxyz,
                LD_pool,
                dpsir_ylm_z.get_ptr_2D());

	//integrate (psi_mu*v(r)*dv) * psi_nu on grid
	//and accumulates to the corresponding element in Hamiltonian
        this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(),
                                    grid_index, cal_flag.get_ptr_2D(), psir_vlbr3.get_ptr_2D(),
                                    dpsir_ylm_x_T.get_ptr_2D(), &pvdpRx_thread);
        this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(),
                                    grid_index, cal_flag.get_ptr_2D(), psir_vlbr3.get_ptr_2D(),
                                    dpsir_ylm_y_T.get_ptr_2D(), &pvdpRy_thread);
        this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(),
                                    grid_index, cal_flag.get_ptr_2D(), psir_vlbr3.get_ptr_2D(),
                                    dpsir_ylm_z_T.get_ptr_2D(), &pvdpRz_thread);
    }
    #pragma omp critical(gint_k)
    {
        BlasConnector::axpy(nnrg,
                            1.0,
                            pvdpRx_thread.get_wrapper(),
                            1,
                            this->pvdpRx_reduced[inout->ispin].get_wrapper(),
                            1);
        BlasConnector::axpy(nnrg,
                            1.0,
                            pvdpRy_thread.get_wrapper(),
                            1,
                            this->pvdpRy_reduced[inout->ispin].get_wrapper(),
                            1);
        BlasConnector::axpy(nnrg,
                            1.0,
                            pvdpRz_thread.get_wrapper(),
                            1,
                            this->pvdpRz_reduced[inout->ispin].get_wrapper(),
                            1);
    }
}
    ModuleBase::TITLE("Gint_interface", "cal_gint_dvlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_dvlocal");
}

void Gint::gint_kernel_vlocal_meta(Gint_inout* inout) {
    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal_meta");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal_meta");
    const UnitCell& ucell = *this->ucell;
    const int max_size = this->gridt->max_atom;
    const int lgd = this->gridt->lgd;
    const int ncyz = this->ny * this->nplane;
    const double dv = ucell.omega / this->ncxyz;
    const double delta_r = this->gridt->dr_uniform;
    hamilt::HContainer<double>* hRGint_kernel = PARAM.inp.nspin != 4 ? this->hRGint : this->hRGint_tmp[inout->ispin];
    hRGint_kernel->set_zero();
    const int nnrg = hRGint_kernel->get_nnr();

#pragma omp parallel
{
    // define HContainer here to reference.
    //Under the condition of gamma_only, hRGint will be instantiated.
    hamilt::HContainer<double> hRGint_thread(*hRGint_kernel);
    std::vector<int> block_iw(max_size,0);
    std::vector<int> block_index(max_size+1,0);
    std::vector<int> block_size(max_size,0);
    std::vector<double> vldr3(this->bxyz,0.0);
    std::vector<double> vkdr3(this->bxyz,0.0);

#pragma omp for schedule(dynamic)
    for (int grid_index = 0; grid_index < this->nbxx; grid_index++) {
        const int na_grid = this->gridt->how_many_atoms[grid_index];
        if (na_grid == 0) {
            continue;
        }
        Gint_Tools::get_gint_vldr3(vldr3.data(),
                                inout->vl,
                                this->bxyz,
                                this->bx,
                                this->by,
                                this->bz,
                                this->nplane,
                                this->gridt->start_ind[grid_index],
                                ncyz,
                                dv);
        Gint_Tools::get_gint_vldr3(vkdr3.data(),
                                    inout->vofk,
                                    this->bxyz,
                                    this->bx,
                                    this->by,
                                    this->bz,
                                    this->nplane,
                                    this->gridt->start_ind[grid_index],
                                    ncyz,
                                    dv);
        //prepare block information
        ModuleBase::Array_Pool<bool> cal_flag(this->bxyz,max_size);
	    Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, 
                                    block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());

        //evaluate psi and dpsi on grids
        const int LD_pool = block_index[na_grid];
        ModuleBase::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_x(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_y(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_z(this->bxyz, LD_pool);

        Gint_Tools::cal_dpsir_ylm(*this->gridt,
            this->bxyz, na_grid, grid_index, delta_r,
            block_index.data(), block_size.data(), 
            cal_flag.get_ptr_2D(),
            psir_ylm.get_ptr_2D(),
            dpsir_ylm_x.get_ptr_2D(),
            dpsir_ylm_y.get_ptr_2D(),
            dpsir_ylm_z.get_ptr_2D()
        );
	
        // 直接生成转置布局 f_mu(r)=v(r)*psi_mu(r)*dv
        const ModuleBase::Array_Pool<double> psir_vlbr3_T
            = build_psir_vlbr3_transposed(
                this->bxyz,
                na_grid,
                LD_pool,
                block_index.data(),
                cal_flag.get_ptr_2D(),
                vldr3.data(),
                psir_ylm.get_ptr_2D());

        // 直接生成转置布局 df_mu(r)=vofk(r)*dpsi_mu(r)*dv
        const ModuleBase::Array_Pool<double> dpsix_vlbr3_T
            = build_psir_vlbr3_transposed(
                this->bxyz,
                na_grid,
                LD_pool,
                block_index.data(),
                cal_flag.get_ptr_2D(),
                vkdr3.data(),
                dpsir_ylm_x.get_ptr_2D());

        const ModuleBase::Array_Pool<double> dpsiy_vlbr3_T
            = build_psir_vlbr3_transposed(
                this->bxyz,
                na_grid,
                LD_pool,
                block_index.data(),
                cal_flag.get_ptr_2D(),
                vkdr3.data(),
                dpsir_ylm_y.get_ptr_2D());

        const ModuleBase::Array_Pool<double> dpsiz_vlbr3_T
            = build_psir_vlbr3_transposed(
                this->bxyz,
                na_grid,
                LD_pool,
                block_index.data(),
                cal_flag.get_ptr_2D(),
                vkdr3.data(),
                dpsir_ylm_z.get_ptr_2D());


        //integrate (psi_mu*v(r)*dv) * psi_nu on grid
        //and accumulates to the corresponding element in Hamiltonian
        this->cal_meshball_vlocal(
            na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(),
            psir_ylm.get_ptr_2D(), psir_vlbr3_T.get_ptr_2D(), &hRGint_thread);
        //integrate (d/dx_i psi_mu*vk(r)*dv) * (d/dx_i psi_nu) on grid (x_i=x,y,z)
        //and accumulates to the corresponding element in Hamiltonian
        this->cal_meshball_vlocal(
            na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(),
            dpsir_ylm_x.get_ptr_2D(), dpsix_vlbr3_T.get_ptr_2D(), &hRGint_thread);
        this->cal_meshball_vlocal(
            na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(),
            dpsir_ylm_y.get_ptr_2D(), dpsiy_vlbr3_T.get_ptr_2D(), &hRGint_thread);
        this->cal_meshball_vlocal(
            na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(),
            dpsir_ylm_z.get_ptr_2D(), dpsiz_vlbr3_T.get_ptr_2D(), &hRGint_thread);
    }

#pragma omp critical
    {
        BlasConnector::axpy(nnrg,
                            1.0,
                            hRGint_thread.get_wrapper(),
                            1,
                            hRGint_kernel->get_wrapper(),
                            1);
    }
}

    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal_meta");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal_meta");
}