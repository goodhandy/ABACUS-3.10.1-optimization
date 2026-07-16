#include "gint.h"
#include "module_base/memory.h"
#include "module_parameter/parameter.h"
#include "module_base/timer.h"
#include "module_base/array_pool.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>


namespace GintVlocalFusion
{

void build_same_source_transposed_panels_and_masks(int bxyz, int na_grid, int LD_pool, const int* block_index,
                                                   const bool* const* cal_flag, const double* scale,
                                                   const double* src, double* right_T, double* left_T,
                                                   int expected_mask_uses);

void build_two_source_transposed_panels_and_masks(int bxyz, int na_grid, int LD_pool, const int* block_index,
                                                  const bool* const* cal_flag, const double* scale,
                                                  const double* right_src, const double* left_src,
                                                  double* right_T, double* left_T, int expected_mask_uses);

void build_scaled_transposed_panel_and_masks(int bxyz, int na_grid, int LD_pool, const int* block_index,
                                             const bool* const* cal_flag, const double* scale,
                                             const double* src, double* dst_T, int expected_mask_uses);

void build_transposed_panel(int bxyz, int na_grid, int LD_pool, const int* block_index,
                            const bool* const* cal_flag, const double* src, double* dst_T);

void build_same_source_transposed_panels_without_masks(int bxyz, int na_grid, int LD_pool,
                                                       const int* block_index, const bool* const* cal_flag,
                                                       const double* scale, const double* src,
                                                       double* right_T, double* left_T);

} // namespace GintVlocalFusion

namespace
{

/**
 * @brief 每个 OpenMP 线程复用的两个扁平转置面板工作区。
 *
 * 使用普通 std::vector，不指定额外对齐，也不填充行距。每条轨道行
 * 的实际长度固定为 bxyz。行指针表仅用于兼容 cal_meshball_vlocal
 * 的现有二维指针接口，转置热循环仍直接使用扁平连续缓冲区。
 */
class TransposedPanelWorkspace
{
public:
    void prepare(int orbital_rows, int bxyz)
    {
        rows_ = orbital_rows;
        bxyz_ = bxyz;
        const std::size_t required = static_cast<std::size_t>(rows_) * bxyz_;

        if (first_.size() < required)
        {
            first_.resize(required);
        }

        if (second_.size() < required)
        {
            second_.resize(required);
        }

        first_rows_.resize(rows_);
        second_rows_.resize(rows_);

        for (int iorb = 0; iorb < rows_; ++iorb)
        {
            first_rows_[iorb] = first_.data() + static_cast<std::size_t>(iorb) * bxyz_;
            second_rows_[iorb] = second_.data() + static_cast<std::size_t>(iorb) * bxyz_;
        }
    }

    double* first_data()
    {
        return first_.data();
    }

    double* second_data()
    {
        return second_.data();
    }

    const double* const* first_rows() const
    {
        return first_rows_.data();
    }

    const double* const* second_rows() const
    {
        return second_rows_.data();
    }

private:
    int rows_ = 0;
    int bxyz_ = 0;
    std::vector<double> first_;
    std::vector<double> second_;
    std::vector<const double*> first_rows_;
    std::vector<const double*> second_rows_;
};

} // 匿名命名空间

void Gint::gint_kernel_vlocal(Gint_inout* inout)
{
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

    hamilt::HContainer<double>* hRGint_kernel = PARAM.inp.nspin != 4
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

        TransposedPanelWorkspace panel_workspace;

#pragma omp for schedule(dynamic)
        for (int grid_index = 0;
             grid_index < this->nbxx;
             ++grid_index)
        {
            const int na_grid = this->gridt
                      ->how_many_atoms[grid_index];

            if (na_grid == 0)
            {
                continue;
            }

            // -------------------------------------------------------------
            // A. 网格块准备
            // -------------------------------------------------------------
            t_start = std::chrono::steady_clock::now();

            ModuleBase::Array_Pool<bool>
                cal_flag(this->bxyz, max_size);

            Gint_Tools::get_gint_vldr3(vldr3.data(), inout->vl, this->bxyz, this->bx, this->by, this->bz, this->nplane, this->gridt ->start_ind[grid_index], ncyz, dv);

            Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());

            const int LD_pool = block_index[na_grid];

            ModuleBase::Array_Pool<double>
                psir_ylm(this->bxyz, LD_pool);

            t_end = std::chrono::steady_clock::now();

            thread_time_prep += std::chrono::duration<double>(t_end - t_start).count();

            // -------------------------------------------------------------
            // B. 基函数计算
            // -------------------------------------------------------------
            t_start = std::chrono::steady_clock::now();

            Gint_Tools::cal_psir_ylm(*this->gridt, this->bxyz, na_grid, grid_index, delta_r, block_index.data(), block_size.data(), cal_flag.get_ptr_2D(), psir_ylm.get_ptr_2D());

            // 保留原代码对 psir_func_2 的求值语义；当前 vlocal 收缩仍不使用其结果。
            const ModuleBase::Array_Pool<double>& psir_ylm_2 =
                (!this->psir_func_2)
                    ? psir_ylm
                    : this->psir_func_2(psir_ylm, *this->gridt, grid_index, 0, block_iw, block_size, block_index, cal_flag);
            (void)psir_ylm_2;

            t_end = std::chrono::steady_clock::now();
            thread_time_psir += std::chrono::duration<double>(t_end - t_start).count();

            // -------------------------------------------------------------
            // C1. 同源使用同源单循环；不同源使用双源单循环
            // -------------------------------------------------------------
            t_start = std::chrono::steady_clock::now();

            panel_workspace.prepare(LD_pool, this->bxyz);
            double* right_T = panel_workspace.first_data();
            double* left_T = panel_workspace.second_data();
            const double* psir_ylm_flat = psir_ylm.get_ptr_2D()[0];

            if (!this->psir_func_1)
            {
                GintVlocalFusion::build_same_source_transposed_panels_and_masks(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vldr3.data(), psir_ylm_flat, right_T, left_T, 1);
            }
            else
            {
                const ModuleBase::Array_Pool<double>& psir_ylm_1 = this->psir_func_1(psir_ylm, *this->gridt, grid_index, 0, block_iw, block_size, block_index, cal_flag);

                GintVlocalFusion::build_two_source_transposed_panels_and_masks(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vldr3.data(), psir_ylm_flat, psir_ylm_1.get_ptr_2D()[0], right_T, left_T, 1);
            }

            t_end = std::chrono::steady_clock::now();
            thread_time_panels += std::chrono::duration<double>(t_end - t_start).count();

            // -------------------------------------------------------------
            // C2. 直接消费两个转置面板完成收缩
            // -------------------------------------------------------------
            t_start = std::chrono::steady_clock::now();

            ModuleBase::timer::tick("Handy_Test", "cal_meshball_vlocal");

            this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(), panel_workspace.first_rows(), panel_workspace.second_rows(), &hRGint_thread);

            ModuleBase::timer::tick("Handy_Test", "cal_meshball_vlocal");

            t_end = std::chrono::steady_clock::now();

            thread_time_meshball += std::chrono::duration<double>(t_end - t_start).count();
        }

#pragma omp critical
        {
            BlasConnector::axpy(hRGint_thread.get_nnr(), 1.0, hRGint_thread.get_wrapper(), 1, hRGint_kernel->get_wrapper(), 1);

            total_time_prep += thread_time_prep;

            total_time_psir += thread_time_psir;

            total_time_panels += thread_time_panels;

            total_time_meshball += thread_time_meshball;
        }
    }

    std::printf("  [Total CPU Core-Time] prep: %8.4f s" " | psir: %8.4f s" " | panels: %8.4f s" " | meshball: %8.4f s\n", total_time_prep, total_time_psir, total_time_panels, total_time_meshball);

    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal");

    ModuleBase::TITLE("Handy_Test", "test_cal_gint_vlocal");
    ModuleBase::timer::tick("Handy_Test", "test_cal_gint_vlocal");
}

void Gint::gint_kernel_dvlocal(Gint_inout* inout)
{
    ModuleBase::TITLE("Gint_interface", "cal_gint_dvlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_dvlocal");

    const UnitCell& ucell = *this->ucell;

    const int max_size = this->gridt->max_atom;

    const int lgd = this->gridt->lgd;

    const int nnrg = pvdpRx_reduced[
              inout->ispin].get_nnr();

    const int ncyz = this->ny * this->nplane;

    const double dv = ucell.omega / this->ncxyz;

    const double delta_r = this->gridt->dr_uniform;

    (void)lgd;

    if (PARAM.globalv.gamma_only_local)
    {
        ModuleBase::WARNING_QUIT("Gint_interface::cal_gint", "dvlocal only for k point!");
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
            pvdpRx_thread(pvdpRx_reduced[inout->ispin]);

        hamilt::HContainer<double>
            pvdpRy_thread(pvdpRy_reduced[inout->ispin]);

        hamilt::HContainer<double>
            pvdpRz_thread(pvdpRz_reduced[inout->ispin]);

        std::vector<int>
            block_iw(max_size, 0);

        std::vector<int>
            block_index(max_size + 1, 0);

        std::vector<int>
            block_size(max_size, 0);

        std::vector<double>
            vldr3(this->bxyz, 0.0);

        TransposedPanelWorkspace panel_workspace;

#pragma omp for schedule(dynamic)
        for (int grid_index = 0;
             grid_index < this->nbxx;
             ++grid_index)
        {
            const int na_grid = this->gridt
                      ->how_many_atoms[grid_index];

            if (na_grid == 0)
            {
                continue;
            }

            Gint_Tools::get_gint_vldr3(vldr3.data(), inout->vl, this->bxyz, this->bx, this->by, this->bz, this->nplane, this->gridt ->start_ind[grid_index], ncyz, dv);

            ModuleBase::Array_Pool<bool>
                cal_flag(this->bxyz, max_size);

            Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());

            const int LD_pool = block_index[na_grid];

            ModuleBase::Array_Pool<double>
                psir_ylm(this->bxyz, LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_x(this->bxyz, LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_y(this->bxyz, LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_z(this->bxyz, LD_pool);

            Gint_Tools::cal_dpsir_ylm(*this->gridt, this->bxyz, na_grid, grid_index, delta_r, block_index.data(), block_size.data(), cal_flag.get_ptr_2D(), psir_ylm.get_ptr_2D(), dpsir_ylm_x.get_ptr_2D(), dpsir_ylm_y.get_ptr_2D(), dpsir_ylm_z.get_ptr_2D());

            /*
             * 第二个缓冲区保存可被x/y/z复用的左面板，第一个缓冲区依次
             * 保存三个方向的导数右面板。
             */
            panel_workspace.prepare(LD_pool, this->bxyz);
            double* derivative_T = panel_workspace.first_data();
            double* psir_vlbr3_T = panel_workspace.second_data();

            GintVlocalFusion::build_scaled_transposed_panel_and_masks(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vldr3.data(), psir_ylm.get_ptr_2D()[0], psir_vlbr3_T, 3);

            GintVlocalFusion::build_transposed_panel(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), dpsir_ylm_x.get_ptr_2D()[0], derivative_T);
            this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(), panel_workspace.first_rows(), panel_workspace.second_rows(), &pvdpRx_thread);

            GintVlocalFusion::build_transposed_panel(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), dpsir_ylm_y.get_ptr_2D()[0], derivative_T);
            this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(), panel_workspace.first_rows(), panel_workspace.second_rows(), &pvdpRy_thread);

            GintVlocalFusion::build_transposed_panel(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), dpsir_ylm_z.get_ptr_2D()[0], derivative_T);
            this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(), panel_workspace.first_rows(), panel_workspace.second_rows(), &pvdpRz_thread);
        }

#pragma omp critical(gint_k)
        {
            BlasConnector::axpy(nnrg, 1.0, pvdpRx_thread.get_wrapper(), 1, this->pvdpRx_reduced[inout->ispin] .get_wrapper(), 1);

            BlasConnector::axpy(nnrg, 1.0, pvdpRy_thread.get_wrapper(), 1, this->pvdpRy_reduced[inout->ispin] .get_wrapper(), 1);

            BlasConnector::axpy(nnrg, 1.0, pvdpRz_thread.get_wrapper(), 1, this->pvdpRz_reduced[inout->ispin] .get_wrapper(), 1);
        }
    }

    ModuleBase::TITLE("Gint_interface", "cal_gint_dvlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_dvlocal");
}

void Gint::gint_kernel_vlocal_meta(Gint_inout* inout)
{
    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal_meta");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal_meta");

    const UnitCell& ucell = *this->ucell;

    const int max_size = this->gridt->max_atom;

    const int lgd = this->gridt->lgd;

    const int ncyz = this->ny * this->nplane;

    const double dv = ucell.omega / this->ncxyz;

    const double delta_r = this->gridt->dr_uniform;

    hamilt::HContainer<double>* hRGint_kernel = PARAM.inp.nspin != 4
        ? this->hRGint
        : this->hRGint_tmp[inout->ispin];

    hRGint_kernel->set_zero();

    const int nnrg = hRGint_kernel->get_nnr();

    (void)lgd;

#pragma omp parallel
    {
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

        std::vector<double>
            vkdr3(this->bxyz, 0.0);

        TransposedPanelWorkspace panel_workspace;

#pragma omp for schedule(dynamic)
        for (int grid_index = 0;
             grid_index < this->nbxx;
             ++grid_index)
        {
            const int na_grid = this->gridt
                      ->how_many_atoms[grid_index];

            if (na_grid == 0)
            {
                continue;
            }

            Gint_Tools::get_gint_vldr3(vldr3.data(), inout->vl, this->bxyz, this->bx, this->by, this->bz, this->nplane, this->gridt ->start_ind[grid_index], ncyz, dv);

            Gint_Tools::get_gint_vldr3(vkdr3.data(), inout->vofk, this->bxyz, this->bx, this->by, this->bz, this->nplane, this->gridt ->start_ind[grid_index], ncyz, dv);

            ModuleBase::Array_Pool<bool>
                cal_flag(this->bxyz, max_size);

            Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());

            const int LD_pool = block_index[na_grid];

            ModuleBase::Array_Pool<double>
                psir_ylm(this->bxyz, LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_x(this->bxyz, LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_y(this->bxyz, LD_pool);

            ModuleBase::Array_Pool<double>
                dpsir_ylm_z(this->bxyz, LD_pool);

            Gint_Tools::cal_dpsir_ylm(*this->gridt, this->bxyz, na_grid, grid_index, delta_r, block_index.data(), block_size.data(), cal_flag.get_ptr_2D(), psir_ylm.get_ptr_2D(), dpsir_ylm_x.get_ptr_2D(), dpsir_ylm_y.get_ptr_2D(), dpsir_ylm_z.get_ptr_2D());

            panel_workspace.prepare(LD_pool, this->bxyz);
            double* right_T = panel_workspace.first_data();
            double* left_T = panel_workspace.second_data();

            GintVlocalFusion::build_same_source_transposed_panels_and_masks(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vldr3.data(), psir_ylm.get_ptr_2D()[0], right_T, left_T, 4);
            this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(), panel_workspace.first_rows(), panel_workspace.second_rows(), &hRGint_thread);

            GintVlocalFusion::build_same_source_transposed_panels_without_masks(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vkdr3.data(), dpsir_ylm_x.get_ptr_2D()[0], right_T, left_T);
            this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(), panel_workspace.first_rows(), panel_workspace.second_rows(), &hRGint_thread);

            GintVlocalFusion::build_same_source_transposed_panels_without_masks(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vkdr3.data(), dpsir_ylm_y.get_ptr_2D()[0], right_T, left_T);
            this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(), panel_workspace.first_rows(), panel_workspace.second_rows(), &hRGint_thread);

            GintVlocalFusion::build_same_source_transposed_panels_without_masks(this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vkdr3.data(), dpsir_ylm_z.get_ptr_2D()[0], right_T, left_T);
            this->cal_meshball_vlocal(na_grid, LD_pool, block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(), panel_workspace.first_rows(), panel_workspace.second_rows(), &hRGint_thread);
        }

#pragma omp critical
        {
            BlasConnector::axpy(nnrg, 1.0, hRGint_thread.get_wrapper(), 1, hRGint_kernel->get_wrapper(), 1);
        }
    }

    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal_meta");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal_meta");
}
