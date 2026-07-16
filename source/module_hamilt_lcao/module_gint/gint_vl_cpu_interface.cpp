#include "gint.h"
#include "module_base/memory.h"
#include "module_parameter/parameter.h"
#include "module_base/timer.h"
#include "module_base/array_pool.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace
{

/**
 * @brief 在一次遍历中同时生成右转置面板和左转置面板。
 *
 * 输入布局：
 *
 *     psir_right_source[ib][iorb]
 *     psir_left_source [ib][iorb]
 *
 * 输出布局：
 *
 *     psir_right_T[iorb][ib]
 *         = psir_right_source[ib][iorb]
 *
 *     psir_left_T[iorb][ib]
 *         = scale[ib] * psir_left_source[ib][iorb]
 *
 * 两个输出的逻辑形状均为 [LD_pool][bxyz]，可以直接传入
 * cal_meshball_vlocal，不再需要在收缩函数内部执行显式转置。
 *
 * 对 cal_flag[ib][ia] 为 false 的原子轨道块显式写0，保证稠密
 * DGEMM 在首尾区间中计算空洞位置时仍保持原来的数值语义。
 */
inline void build_two_transposed_panels(
    const int bxyz,
    const int na_grid,
    const int* const block_index,
    const bool* const* const cal_flag,
    const double* const scale,
    const double* const* const psir_right_source,
    const double* const* const psir_left_source,
    double* const* const psir_right_T,
    double* const* const psir_left_T)
{
    /*
     * ib 为外层循环，使两个源矩阵均沿轨道方向连续读取。
     * 每个原子—网格点组合只判断一次 cal_flag。
     */
    for (int ib = 0; ib < bxyz; ++ib)
    {
        const double* const right_src_row
            = psir_right_source[ib];

        const double* const left_src_row
            = psir_left_source[ib];

        const double scale_value
            = scale[ib];

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
                    psir_right_T[iorb][ib]
                        = right_src_row[iorb];

                    psir_left_T[iorb][ib]
                        = scale_value
                        * left_src_row[iorb];
                }
            }
            else
            {
                for (int iorb = orbital_begin;
                     iorb < orbital_end;
                     ++iorb)
                {
                    psir_right_T[iorb][ib] = 0.0;
                    psir_left_T[iorb][ib] = 0.0;
                }
            }
        }
    }
}

/**
 * @brief 生成一个带 cal_flag 遮罩的普通转置面板。
 *
 * 输出：
 *
 *     dst_T[iorb][ib] = src[ib][iorb]
 */
inline void build_transposed_panel(
    const int bxyz,
    const int na_grid,
    const int* const block_index,
    const bool* const* const cal_flag,
    const double* const* const src,
    double* const* const dst_T)
{
    for (int ib = 0; ib < bxyz; ++ib)
    {
        const double* const src_row
            = src[ib];

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
                    dst_T[iorb][ib]
                        = src_row[iorb];
                }
            }
            else
            {
                for (int iorb = orbital_begin;
                     iorb < orbital_end;
                     ++iorb)
                {
                    dst_T[iorb][ib] = 0.0;
                }
            }
        }
    }
}

/**
 * @brief 生成一个带缩放和 cal_flag 遮罩的转置面板。
 *
 * 输出：
 *
 *     dst_T[iorb][ib]
 *         = scale[ib] * src[ib][iorb]
 */
inline void build_scaled_transposed_panel(
    const int bxyz,
    const int na_grid,
    const int* const block_index,
    const bool* const* const cal_flag,
    const double* const scale,
    const double* const* const src,
    double* const* const dst_T)
{
    for (int ib = 0; ib < bxyz; ++ib)
    {
        const double* const src_row
            = src[ib];

        const double scale_value
            = scale[ib];

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
                    dst_T[iorb][ib]
                        = scale_value
                        * src_row[iorb];
                }
            }
            else
            {
                for (int iorb = orbital_begin;
                     iorb < orbital_end;
                     ++iorb)
                {
                    dst_T[iorb][ib] = 0.0;
                }
            }
        }
    }
}

} // 匿名命名空间

void Gint::gint_kernel_vlocal(Gint_inout* inout)
{
    ModuleBase::TITLE(
        "Gint_interface",
        "cal_gint_vlocal");
    ModuleBase::timer::tick(
        "Gint_interface",
        "cal_gint_vlocal");

    ModuleBase::TITLE(
        "Handy_Test",
        "test_cal_gint_vlocal");
    ModuleBase::timer::tick(
        "Handy_Test",
        "test_cal_gint_vlocal");

    const UnitCell& ucell
        = *this->ucell;

    const int max_size
        = this->gridt->max_atom;

    const int lgd
        = this->gridt->lgd;

    const int ncyz
        = this->ny * this->nplane;

    const double dv
        = ucell.omega / this->ncxyz;

    const double delta_r
        = this->gridt->dr_uniform;

    hamilt::HContainer<double>* hRGint_kernel
        = PARAM.inp.nspin != 4
        ? this->hRGint
        : this->hRGint_tmp[inout->ispin];

    hRGint_kernel->set_zero();

    (void)lgd;

    double total_time_prep = 0.0;
    double total_time_psir = 0.0;
    double total_time_panels = 0.0;
    double total_time_meshball = 0.0;

#pragma omp parallel
    {
        double thread_time_prep = 0.0;
        double thread_time_psir = 0.0;
        double thread_time_panels = 0.0;
        double thread_time_meshball = 0.0;

        std::chrono::steady_clock::time_point
            t_start,
            t_end;

        hamilt::HContainer<double>
            hRGint_thread(*hRGint_kernel);

        std::vector<int>
            block_iw(max_size, 0);

        std::vector<int>
            block_index(max_size + 1, 0);

        std::vector<int>
            block_size(max_size, 0);

        std::vector<double>
            vldr3(this->bxyz, 0.0);

#pragma omp for schedule(dynamic)
        for (int grid_index = 0;
             grid_index < this->nbxx;
             ++grid_index)
        {
            const int na_grid
                = this->gridt
                      ->how_many_atoms[grid_index];

            if (na_grid == 0)
            {
                continue;
            }

            // -------------------------------------------------------------
            // A. 网格块准备
            // -------------------------------------------------------------
            t_start
                = std::chrono::steady_clock::now();

            ModuleBase::Array_Pool<bool>
                cal_flag(this->bxyz, max_size);

            Gint_Tools::get_gint_vldr3(
                vldr3.data(),
                inout->vl,
                this->bxyz,
                this->bx,
                this->by,
                this->bz,
                this->nplane,
                this->gridt
                    ->start_ind[grid_index],
                ncyz,
                dv);

            Gint_Tools::get_block_info(
                *this->gridt,
                this->bxyz,
                na_grid,
                grid_index,
                block_iw.data(),
                block_index.data(),
                block_size.data(),
                cal_flag.get_ptr_2D());

            const int LD_pool
                = block_index[na_grid];

            ModuleBase::Array_Pool<double>
                psir_ylm(
                    this->bxyz,
                    LD_pool);

            t_end
                = std::chrono::steady_clock::now();

            thread_time_prep
                += std::chrono::duration<double>(
                    t_end - t_start).count();

            // -------------------------------------------------------------
            // B. 基函数计算
            // -------------------------------------------------------------
            t_start
                = std::chrono::steady_clock::now();

            Gint_Tools::cal_psir_ylm(
                *this->gridt,
                this->bxyz,
                na_grid,
                grid_index,
                delta_r,
                block_index.data(),
                block_size.data(),
                cal_flag.get_ptr_2D(),
                psir_ylm.get_ptr_2D());

            const ModuleBase::Array_Pool<double>&
                psir_ylm_1
                = (!this->psir_func_1)
                ? psir_ylm
                : this->psir_func_1(
                      psir_ylm,
                      *this->gridt,
                      grid_index,
                      0,
                      block_iw,
                      block_size,
                      block_index,
                      cal_flag);

            const ModuleBase::Array_Pool<double>&
                psir_ylm_2
                = (!this->psir_func_2)
                ? psir_ylm
                : this->psir_func_2(
                      psir_ylm,
                      *this->gridt,
                      grid_index,
                      0,
                      block_iw,
                      block_size,
                      block_index,
                      cal_flag);

            t_end
                = std::chrono::steady_clock::now();

            thread_time_psir
                += std::chrono::duration<double>(
                    t_end - t_start).count();

            // -------------------------------------------------------------
            // C1. 一次遍历同时生成两个转置面板
            // -------------------------------------------------------------
            t_start
                = std::chrono::steady_clock::now();

            ModuleBase::Array_Pool<double>
                psir_right_T(
                    LD_pool,
                    this->bxyz);

            ModuleBase::Array_Pool<double>
                psir_vlbr3_T(
                    LD_pool,
                    this->bxyz);

            /*
             * 保持当前上传代码的调用语义：
             *
             * 右面板来自 psir_ylm；
             * 左面板来自 psir_ylm_1，并乘以 vldr3。
             *
             * psir_ylm_2 的计算仍然保留，但本轮实验不改变右侧来源。
             */
            (void)psir_ylm_2;

            build_two_transposed_panels(
                this->bxyz,
                na_grid,
                block_index.data(),
                cal_flag.get_ptr_2D(),
                vldr3.data(),
                psir_ylm.get_ptr_2D(),
                psir_ylm_1.get_ptr_2D(),
                psir_right_T.get_ptr_2D(),
                psir_vlbr3_T.get_ptr_2D());

            t_end
                = std::chrono::steady_clock::now();

            thread_time_panels
                += std::chrono::duration<double>(
                    t_end - t_start).count();

            // -------------------------------------------------------------
            // C2. 直接消费两个转置面板完成收缩
            // -------------------------------------------------------------
            t_start
                = std::chrono::steady_clock::now();

            ModuleBase::timer::tick(
                "Handy_Test",
                "cal_meshball_vlocal");

            this->cal_meshball_vlocal(
                na_grid,
                LD_pool,
                block_size.data(),
                block_index.data(),
                grid_index,
                cal_flag.get_ptr_2D(),
                psir_right_T.get_ptr_2D(),
                psir_vlbr3_T.get_ptr_2D(),
                &hRGint_thread);

            ModuleBase::timer::tick(
                "Handy_Test",
                "cal_meshball_vlocal");

            t_end
                = std::chrono::steady_clock::now();

            thread_time_meshball
                += std::chrono::duration<double>(
                    t_end - t_start).count();
        }

#pragma omp critical
        {
            BlasConnector::axpy(
                hRGint_thread.get_nnr(),
                1.0,
                hRGint_thread.get_wrapper(),
                1,
                hRGint_kernel->get_wrapper(),
                1);

            total_time_prep
                += thread_time_prep;

            total_time_psir
                += thread_time_psir;

            total_time_panels
                += thread_time_panels;

            total_time_meshball
                += thread_time_meshball;
        }
    }

    std::printf(
        "  [Total CPU Core-Time] prep: %8.4f s"
        " | psir: %8.4f s"
        " | panels: %8.4f s"
        " | meshball: %8.4f s\n",
        total_time_prep,
        total_time_psir,
        total_time_panels,
        total_time_meshball);

    ModuleBase::TITLE(
        "Gint_interface",
        "cal_gint_vlocal");
    ModuleBase::timer::tick(
        "Gint_interface",
        "cal_gint_vlocal");

    ModuleBase::TITLE(
        "Handy_Test",
        "test_cal_gint_vlocal");
    ModuleBase::timer::tick(
        "Handy_Test",
        "test_cal_gint_vlocal");
}

void Gint::gint_kernel_dvlocal(Gint_inout* inout)
{
    ModuleBase::TITLE(
        "Gint_interface",
        "cal_gint_dvlocal");
    ModuleBase::timer::tick(
        "Gint_interface",
        "cal_gint_dvlocal");

    const UnitCell& ucell
        = *this->ucell;

    const int max_size
        = this->gridt->max_atom;

    const int lgd
        = this->gridt->lgd;

    const int nnrg
        = pvdpRx_reduced[
              inout->ispin].get_nnr();

    const int ncyz
        = this->ny * this->nplane;

    const double dv
        = ucell.omega / this->ncxyz;

    const double delta_r
        = this->gridt->dr_uniform;

    (void)lgd;

    if (PARAM.globalv.gamma_only_local)
    {
        ModuleBase::WARNING_QUIT(
            "Gint_interface::cal_gint",
            "dvlocal only for k point!");
    }

    pvdpRx_reduced[
        inout->ispin].set_zero();

    pvdpRy_reduced[
        inout->ispin].set_zero();

    pvdpRz_reduced[
        inout->ispin].set_zero();

#pragma omp parallel
    {
        hamilt::HContainer<double>
            pvdpRx_thread(
                pvdpRx_reduced[
                    inout->ispin]);

        hamilt::HContainer<double>
            pvdpRy_thread(
                pvdpRy_reduced[
                    inout->ispin]);

        hamilt::HContainer<double>
            pvdpRz_thread(
                pvdpRz_reduced[
                    inout->ispin]);

        std::vector<int>
            block_iw(max_size, 0);

        std::vector<int>
            block_index(max_size + 1, 0);

        std::vector<int>
            block_size(max_size, 0);

        std::vector<double>
            vldr3(this->bxyz, 0.0);

#pragma omp for schedule(dynamic)
        for (int grid_index = 0;
             grid_index < this->nbxx;
             ++grid_index)
        {
            const int na_grid
                = this->gridt
                      ->how_many_atoms[grid_index];

            if (na_grid == 0)
            {
                continue;
            }

            Gint_Tools::get_gint_vldr3(
                vldr3.data(),
                inout->vl,
                this->bxyz,
                this->bx,
                this->by,
                this->bz,
                this->nplane,
                this->gridt
                    ->start_ind[grid_index],
                ncyz,
                dv);

            ModuleBase::Array_Pool<bool>
                cal_flag(
                    this->bxyz,
                    max_size);

            Gint_Tools::get_block_info(
                *this->gridt,
                this->bxyz,
                na_grid,
                grid_index,
                block_iw.data(),
                block_index.data(),
                block_size.data(),
                cal_flag.get_ptr_2D());

            const int LD_pool
                = block_index[na_grid];

            ModuleBase::Array_Pool<double>
                psir_ylm(
                    this->bxyz,
                    LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_x(
                    this->bxyz,
                    LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_y(
                    this->bxyz,
                    LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_z(
                    this->bxyz,
                    LD_pool);

            Gint_Tools::cal_dpsir_ylm(
                *this->gridt,
                this->bxyz,
                na_grid,
                grid_index,
                delta_r,
                block_index.data(),
                block_size.data(),
                cal_flag.get_ptr_2D(),
                psir_ylm.get_ptr_2D(),
                dpsir_ylm_x.get_ptr_2D(),
                dpsir_ylm_y.get_ptr_2D(),
                dpsir_ylm_z.get_ptr_2D());

            /*
             * 左面板 f_mu(r)=v(r)*psi_mu(r)*dv 会被 x/y/z
             * 三个方向共同使用，因此只生成一次。
             */
            ModuleBase::Array_Pool<double>
                psir_vlbr3_T(
                    LD_pool,
                    this->bxyz);

            build_scaled_transposed_panel(
                this->bxyz,
                na_grid,
                block_index.data(),
                cal_flag.get_ptr_2D(),
                vldr3.data(),
                psir_ylm.get_ptr_2D(),
                psir_vlbr3_T.get_ptr_2D());

            // x 方向。
            {
                ModuleBase::Array_Pool<double>
                    dpsir_ylm_x_T(
                        LD_pool,
                        this->bxyz);

                build_transposed_panel(
                    this->bxyz,
                    na_grid,
                    block_index.data(),
                    cal_flag.get_ptr_2D(),
                    dpsir_ylm_x.get_ptr_2D(),
                    dpsir_ylm_x_T
                        .get_ptr_2D());

                this->cal_meshball_vlocal(
                    na_grid,
                    LD_pool,
                    block_size.data(),
                    block_index.data(),
                    grid_index,
                    cal_flag.get_ptr_2D(),
                    dpsir_ylm_x_T
                        .get_ptr_2D(),
                    psir_vlbr3_T
                        .get_ptr_2D(),
                    &pvdpRx_thread);
            }

            // y 方向。
            {
                ModuleBase::Array_Pool<double>
                    dpsir_ylm_y_T(
                        LD_pool,
                        this->bxyz);

                build_transposed_panel(
                    this->bxyz,
                    na_grid,
                    block_index.data(),
                    cal_flag.get_ptr_2D(),
                    dpsir_ylm_y.get_ptr_2D(),
                    dpsir_ylm_y_T
                        .get_ptr_2D());

                this->cal_meshball_vlocal(
                    na_grid,
                    LD_pool,
                    block_size.data(),
                    block_index.data(),
                    grid_index,
                    cal_flag.get_ptr_2D(),
                    dpsir_ylm_y_T
                        .get_ptr_2D(),
                    psir_vlbr3_T
                        .get_ptr_2D(),
                    &pvdpRy_thread);
            }

            // z 方向。
            {
                ModuleBase::Array_Pool<double>
                    dpsir_ylm_z_T(
                        LD_pool,
                        this->bxyz);

                build_transposed_panel(
                    this->bxyz,
                    na_grid,
                    block_index.data(),
                    cal_flag.get_ptr_2D(),
                    dpsir_ylm_z.get_ptr_2D(),
                    dpsir_ylm_z_T
                        .get_ptr_2D());

                this->cal_meshball_vlocal(
                    na_grid,
                    LD_pool,
                    block_size.data(),
                    block_index.data(),
                    grid_index,
                    cal_flag.get_ptr_2D(),
                    dpsir_ylm_z_T
                        .get_ptr_2D(),
                    psir_vlbr3_T
                        .get_ptr_2D(),
                    &pvdpRz_thread);
            }
        }

#pragma omp critical(gint_k)
        {
            BlasConnector::axpy(
                nnrg,
                1.0,
                pvdpRx_thread.get_wrapper(),
                1,
                this->pvdpRx_reduced[
                    inout->ispin]
                    .get_wrapper(),
                1);

            BlasConnector::axpy(
                nnrg,
                1.0,
                pvdpRy_thread.get_wrapper(),
                1,
                this->pvdpRy_reduced[
                    inout->ispin]
                    .get_wrapper(),
                1);

            BlasConnector::axpy(
                nnrg,
                1.0,
                pvdpRz_thread.get_wrapper(),
                1,
                this->pvdpRz_reduced[
                    inout->ispin]
                    .get_wrapper(),
                1);
        }
    }

    ModuleBase::TITLE(
        "Gint_interface",
        "cal_gint_dvlocal");
    ModuleBase::timer::tick(
        "Gint_interface",
        "cal_gint_dvlocal");
}

void Gint::gint_kernel_vlocal_meta(
    Gint_inout* inout)
{
    ModuleBase::TITLE(
        "Gint_interface",
        "cal_gint_vlocal_meta");
    ModuleBase::timer::tick(
        "Gint_interface",
        "cal_gint_vlocal_meta");

    const UnitCell& ucell
        = *this->ucell;

    const int max_size
        = this->gridt->max_atom;

    const int lgd
        = this->gridt->lgd;

    const int ncyz
        = this->ny * this->nplane;

    const double dv
        = ucell.omega / this->ncxyz;

    const double delta_r
        = this->gridt->dr_uniform;

    hamilt::HContainer<double>* hRGint_kernel
        = PARAM.inp.nspin != 4
        ? this->hRGint
        : this->hRGint_tmp[inout->ispin];

    hRGint_kernel->set_zero();

    const int nnrg
        = hRGint_kernel->get_nnr();

    (void)lgd;

#pragma omp parallel
    {
        hamilt::HContainer<double>
            hRGint_thread(
                *hRGint_kernel);

        std::vector<int>
            block_iw(max_size, 0);

        std::vector<int>
            block_index(max_size + 1, 0);

        std::vector<int>
            block_size(max_size, 0);

        std::vector<double>
            vldr3(this->bxyz, 0.0);

        std::vector<double>
            vkdr3(this->bxyz, 0.0);

#pragma omp for schedule(dynamic)
        for (int grid_index = 0;
             grid_index < this->nbxx;
             ++grid_index)
        {
            const int na_grid
                = this->gridt
                      ->how_many_atoms[grid_index];

            if (na_grid == 0)
            {
                continue;
            }

            Gint_Tools::get_gint_vldr3(
                vldr3.data(),
                inout->vl,
                this->bxyz,
                this->bx,
                this->by,
                this->bz,
                this->nplane,
                this->gridt
                    ->start_ind[grid_index],
                ncyz,
                dv);

            Gint_Tools::get_gint_vldr3(
                vkdr3.data(),
                inout->vofk,
                this->bxyz,
                this->bx,
                this->by,
                this->bz,
                this->nplane,
                this->gridt
                    ->start_ind[grid_index],
                ncyz,
                dv);

            ModuleBase::Array_Pool<bool>
                cal_flag(
                    this->bxyz,
                    max_size);

            Gint_Tools::get_block_info(
                *this->gridt,
                this->bxyz,
                na_grid,
                grid_index,
                block_iw.data(),
                block_index.data(),
                block_size.data(),
                cal_flag.get_ptr_2D());

            const int LD_pool
                = block_index[na_grid];

            ModuleBase::Array_Pool<double>
                psir_ylm(
                    this->bxyz,
                    LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_x(
                    this->bxyz,
                    LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_y(
                    this->bxyz,
                    LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_z(
                    this->bxyz,
                    LD_pool);

            Gint_Tools::cal_dpsir_ylm(
                *this->gridt,
                this->bxyz,
                na_grid,
                grid_index,
                delta_r,
                block_index.data(),
                block_size.data(),
                cal_flag.get_ptr_2D(),
                psir_ylm.get_ptr_2D(),
                dpsir_ylm_x.get_ptr_2D(),
                dpsir_ylm_y.get_ptr_2D(),
                dpsir_ylm_z.get_ptr_2D());

            /*
             * 每个贡献均在局部作用域内生成两个转置面板，
             * 收缩完成后立即释放，避免同时持有多组完整面板。
             */

            // 普通局域势贡献。
            {
                ModuleBase::Array_Pool<double>
                    right_T(
                        LD_pool,
                        this->bxyz);

                ModuleBase::Array_Pool<double>
                    left_T(
                        LD_pool,
                        this->bxyz);

                build_two_transposed_panels(
                    this->bxyz,
                    na_grid,
                    block_index.data(),
                    cal_flag.get_ptr_2D(),
                    vldr3.data(),
                    psir_ylm.get_ptr_2D(),
                    psir_ylm.get_ptr_2D(),
                    right_T.get_ptr_2D(),
                    left_T.get_ptr_2D());

                this->cal_meshball_vlocal(
                    na_grid,
                    LD_pool,
                    block_size.data(),
                    block_index.data(),
                    grid_index,
                    cal_flag.get_ptr_2D(),
                    right_T.get_ptr_2D(),
                    left_T.get_ptr_2D(),
                    &hRGint_thread);
            }

            // x 方向 meta-GGA 贡献。
            {
                ModuleBase::Array_Pool<double>
                    right_T(
                        LD_pool,
                        this->bxyz);

                ModuleBase::Array_Pool<double>
                    left_T(
                        LD_pool,
                        this->bxyz);

                build_two_transposed_panels(
                    this->bxyz,
                    na_grid,
                    block_index.data(),
                    cal_flag.get_ptr_2D(),
                    vkdr3.data(),
                    dpsir_ylm_x
                        .get_ptr_2D(),
                    dpsir_ylm_x
                        .get_ptr_2D(),
                    right_T.get_ptr_2D(),
                    left_T.get_ptr_2D());

                this->cal_meshball_vlocal(
                    na_grid,
                    LD_pool,
                    block_size.data(),
                    block_index.data(),
                    grid_index,
                    cal_flag.get_ptr_2D(),
                    right_T.get_ptr_2D(),
                    left_T.get_ptr_2D(),
                    &hRGint_thread);
            }

            // y 方向 meta-GGA 贡献。
            {
                ModuleBase::Array_Pool<double>
                    right_T(
                        LD_pool,
                        this->bxyz);

                ModuleBase::Array_Pool<double>
                    left_T(
                        LD_pool,
                        this->bxyz);

                build_two_transposed_panels(
                    this->bxyz,
                    na_grid,
                    block_index.data(),
                    cal_flag.get_ptr_2D(),
                    vkdr3.data(),
                    dpsir_ylm_y
                        .get_ptr_2D(),
                    dpsir_ylm_y
                        .get_ptr_2D(),
                    right_T.get_ptr_2D(),
                    left_T.get_ptr_2D());

                this->cal_meshball_vlocal(
                    na_grid,
                    LD_pool,
                    block_size.data(),
                    block_index.data(),
                    grid_index,
                    cal_flag.get_ptr_2D(),
                    right_T.get_ptr_2D(),
                    left_T.get_ptr_2D(),
                    &hRGint_thread);
            }

            // z 方向 meta-GGA 贡献。
            {
                ModuleBase::Array_Pool<double>
                    right_T(
                        LD_pool,
                        this->bxyz);

                ModuleBase::Array_Pool<double>
                    left_T(
                        LD_pool,
                        this->bxyz);

                build_two_transposed_panels(
                    this->bxyz,
                    na_grid,
                    block_index.data(),
                    cal_flag.get_ptr_2D(),
                    vkdr3.data(),
                    dpsir_ylm_z
                        .get_ptr_2D(),
                    dpsir_ylm_z
                        .get_ptr_2D(),
                    right_T.get_ptr_2D(),
                    left_T.get_ptr_2D());

                this->cal_meshball_vlocal(
                    na_grid,
                    LD_pool,
                    block_size.data(),
                    block_index.data(),
                    grid_index,
                    cal_flag.get_ptr_2D(),
                    right_T.get_ptr_2D(),
                    left_T.get_ptr_2D(),
                    &hRGint_thread);
            }
        }

#pragma omp critical
        {
            BlasConnector::axpy(
                nnrg,
                1.0,
                hRGint_thread.get_wrapper(),
                1,
                hRGint_kernel->get_wrapper(),
                1);
        }
    }

    ModuleBase::TITLE(
        "Gint_interface",
        "cal_gint_vlocal_meta");
    ModuleBase::timer::tick(
        "Gint_interface",
        "cal_gint_vlocal_meta");
}
