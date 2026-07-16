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

#include <algorithm>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __MKL
#include <mkl_service.h>
#endif

namespace
{

/**
 * @brief Transpose two matrices from [bxyz][LD_pool] to [LD_pool][bxyz].
 *
 * The transpose is blocked in two dimensions:
 * 1. orbitals are processed atom by atom;
 * 2. grid points are processed in tiles of GRID_TILE.
 *
 * No complete intermediate matrix is created. Both source matrices are
 * transposed in the same traversal.
 */
inline void transpose_two_matrices_atom_blocked(
    const int na_grid,
    const int bxyz,
    const int ldt,
    const int* const block_index,
    const int* const block_size,
    const double* const* const src_vlbr3,
    const double* const* const src_ylm,
    double* const dst_vlbr3,
    double* const dst_ylm)
{
    constexpr int GRID_TILE = 16;
    constexpr int ORB_TILE = 16;

    for (int ia = 0; ia < na_grid; ++ia)
    {
        const int orbital_begin = block_index[ia];
        const int orbital_count = block_size[ia];

        for (int io0 = 0; io0 < orbital_count; io0 += ORB_TILE)
        {
            const int local_orbitals
                = std::min(ORB_TILE, orbital_count - io0);

            // Cache the destination row pointers so that the inner loop does
            // not repeatedly evaluate (global_orbital * ldt).
            double* dst_vlbr3_rows[ORB_TILE];
            double* dst_ylm_rows[ORB_TILE];

            for (int io = 0; io < local_orbitals; ++io)
            {
                const std::size_t global_orbital
                    = static_cast<std::size_t>(orbital_begin + io0 + io);

                dst_vlbr3_rows[io]
                    = dst_vlbr3 + global_orbital * static_cast<std::size_t>(ldt);
                dst_ylm_rows[io]
                    = dst_ylm + global_orbital * static_cast<std::size_t>(ldt);
            }

            for (int ib0 = 0; ib0 < bxyz; ib0 += GRID_TILE)
            {
                const int ib_end = std::min(ib0 + GRID_TILE, bxyz);

                for (int ib = ib0; ib < ib_end; ++ib)
                {
                    const double* const src_vlbr3_row
                        = src_vlbr3[ib] + orbital_begin + io0;
                    const double* const src_ylm_row
                        = src_ylm[ib] + orbital_begin + io0;

                    // The source values of the current atom are contiguous.
                    // The destination touches only local_orbitals rows, which
                    // keeps the active write set small enough for L1 cache.
                    for (int io = 0; io < local_orbitals; ++io)
                    {
                        dst_vlbr3_rows[io][ib] = src_vlbr3_row[io];
                        dst_ylm_rows[io][ib] = src_ylm_row[io];
                    }
                }
            }
        }
    }
}

} // namespace

void Gint::cal_meshball_vlocal(
    const int na_grid,                         // number of atoms on this grid block
    const int LD_pool,
    const int* const block_size,               // block_size[na_grid]
    const int* const block_index,              // block_index[na_grid + 1]
    const int grid_index,
    const bool* const* const cal_flag,          // cal_flag[bxyz][na_grid]
    const double* const* const psir_ylm,        // psir_ylm[bxyz][LD_pool]
    const double* const* const psir_vlbr3,      // psir_vlbr3[bxyz][LD_pool]
    hamilt::HContainer<double>* hR)
{
    const int bxyz_local = this->bxyz;

    // Keep ldt equal to bxyz in this experiment so that the only change from
    // the previous transpose version is the transpose implementation itself.
    const int ldt = bxyz_local;

    thread_local std::vector<double> vlbr3_T;
    thread_local std::vector<double> ylm_T;

    const std::size_t required_size
        = static_cast<std::size_t>(LD_pool)
        * static_cast<std::size_t>(ldt);

    // The per-thread buffers only grow. Their contents are fully overwritten
    // for the active [0, LD_pool) orbital range on every call.
    if (vlbr3_T.size() < required_size)
    {
        vlbr3_T.resize(required_size);
    }
    if (ylm_T.size() < required_size)
    {
        ylm_T.resize(required_size);
    }

    transpose_two_matrices_atom_blocked(
        na_grid,
        bxyz_local,
        ldt,
        block_index,
        block_size,
        psir_vlbr3,
        psir_ylm,
        vlbr3_T.data(),
        ylm_T.data());

    // After transposition, each orbital contains a contiguous sequence of
    // grid-point values. In Fortran column-major interpretation, each input
    // panel has shape [ib_length][number_of_orbitals].
    const char transa = 'T';
    const char transb = 'N';
    const double alpha = 1.0;
    const double beta = 1.0;

    const int mcell_index = this->gridt->bcell_start[grid_index];

    for (int ia1 = 0; ia1 < na_grid; ++ia1)
    {
        const int bcell1 = mcell_index + ia1;
        const int iat1 = this->gridt->which_atom[bcell1];
        const int id1 = this->gridt->which_unitcell[bcell1];
        const ModuleBase::Vector3<int> r1
            = this->gridt->get_ucell_coords(id1);

        for (int ia2 = 0; ia2 < na_grid; ++ia2)
        {
            const int bcell2 = mcell_index + ia2;
            const int iat2 = this->gridt->which_atom[bcell2];
            const int id2 = this->gridt->which_unitcell[bcell2];
            const ModuleBase::Vector3<int> r2
                = this->gridt->get_ucell_coords(id2);

            if (iat1 > iat2)
            {
                continue;
            }

            // Preserve the original first/last search exactly.
            int first_ib = 0;
            for (int ib = 0; ib < bxyz_local; ++ib)
            {
                if (cal_flag[ib][ia1] && cal_flag[ib][ia2])
                {
                    first_ib = ib;
                    break;
                }
            }

            int last_ib = 0;
            for (int ib = bxyz_local - 1; ib >= 0; --ib)
            {
                if (cal_flag[ib][ia1] && cal_flag[ib][ia2])
                {
                    last_ib = ib + 1;
                    break;
                }
            }

            const int ib_length = last_ib - first_ib;
            if (ib_length <= 0)
            {
                continue;
            }

            const auto tmp_matrix
                = hR->find_matrix(iat1, iat2, r1 - r2);
            if (tmp_matrix == nullptr)
            {
                continue;
            }

            const int m = tmp_matrix->get_row_size();
            const int n = tmp_matrix->get_col_size();

            // Preserve the original sparse/dense decision exactly.
            int cal_pair_num = 0;
            for (int ib = first_ib; ib < last_ib; ++ib)
            {
                cal_pair_num += cal_flag[ib][ia1] && cal_flag[ib][ia2];
            }

            const std::size_t orbital_offset_a
                = static_cast<std::size_t>(block_index[ia2])
                * static_cast<std::size_t>(ldt);
            const std::size_t orbital_offset_b
                = static_cast<std::size_t>(block_index[ia1])
                * static_cast<std::size_t>(ldt);

            if (cal_pair_num > ib_length / 4)
            {
                // Dense path: exactly one DGEMM over [first_ib, last_ib).
                const double* const ptr_a
                    = vlbr3_T.data() + orbital_offset_a + first_ib;
                const double* const ptr_b
                    = ylm_T.data() + orbital_offset_b + first_ib;

                dgemm_(
                    &transa,
                    &transb,
                    &n,
                    &m,
                    &ib_length,
                    &alpha,
                    ptr_a,
                    &ldt,
                    ptr_b,
                    &ldt,
                    &beta,
                    tmp_matrix->get_pointer(),
                    &n);
            }
            else
            {
                // Sparse path: preserve the original k=1 DGEMM for every
                // valid grid point. Only the input layout is changed.
                const int k = 1;

                for (int ib = first_ib; ib < last_ib; ++ib)
                {
                    if (!(cal_flag[ib][ia1] && cal_flag[ib][ia2]))
                    {
                        continue;
                    }

                    const double* const ptr_a
                        = vlbr3_T.data() + orbital_offset_a + ib;
                    const double* const ptr_b
                        = ylm_T.data() + orbital_offset_b + ib;

                    dgemm_(
                        &transa,
                        &transb,
                        &n,
                        &m,
                        &k,
                        &alpha,
                        ptr_a,
                        &ldt,
                        ptr_b,
                        &ldt,
                        &beta,
                        tmp_matrix->get_pointer(),
                        &n);
                }
            }
        }
    }
}
