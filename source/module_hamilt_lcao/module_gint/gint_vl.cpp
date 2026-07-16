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

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __MKL
#include <mkl_service.h>
#endif

namespace
{

/**
 * @brief 统计一个64位整数中值为1的比特数量。
 *
 * 例如：
 * value = 0b10110000，则返回3。
 *
 * 这里使用编译器内建函数 __builtin_popcountll。
 * 在支持 POPCNT 指令的处理器上，编译器通常会将其直接翻译为
 * 一条硬件人口计数指令，避免逐位循环统计。
 *
 * 本函数用于计算两个原子共同有效的网格点数量 cal_pair_num。
 */
inline int popcount64(const std::uint64_t value)
{
    return __builtin_popcountll(
        static_cast<unsigned long long>(value));
}

/**
 * @brief 统计一个非零64位整数末尾连续0的数量。
 *
 * “末尾”指最低有效位方向。例如：
 * value = 0b00101000，最低的1位于第3位，因此返回3。
 *
 * 本函数用于定位位掩码中第一个共同有效网格点。
 *
 * 注意：
 * __builtin_ctzll(0) 的行为未定义，因此调用本函数之前
 * 必须保证 value != 0。
 */
inline int ctz64(const std::uint64_t value)
{
    return __builtin_ctzll(
        static_cast<unsigned long long>(value));
}

/**
 * @brief 统计一个非零64位整数开头连续0的数量。
 *
 * “开头”指最高有效位方向。例如：
 * 如果最高的1位于第20位，则可以通过 63 - clz 得到20。
 *
 * 本函数用于定位位掩码中最后一个共同有效网格点。
 *
 * 注意：
 * __builtin_clzll(0) 的行为未定义，因此调用本函数之前
 * 必须保证 value != 0。
 */
inline int clz64(const std::uint64_t value)
{
    return __builtin_clzll(
        static_cast<unsigned long long>(value));
}

/**
 * @brief 为当前网格块中的每个原子构造动态长度位掩码。
 *
 * 原代码使用：
 *
 *     cal_flag[ib][ia]
 *
 * 表示网格点 ib 是否位于原子 ia 的轨道截断范围内。
 *
 * 本函数将每个原子的 bxyz 个 bool 压缩为若干个 uint64_t。
 * 一个 uint64_t 可表示64个网格点，因此：
 *
 *     mask_words = (bxyz + 63) / 64
 *
 * 例如当前算例 bxyz=75：
 *
 *     mask_words = 2
 *
 * 第0个64位字表示 ib=0~63，
 * 第1个64位字表示 ib=64~74。
 *
 * atom_masks 的逻辑布局为：
 *
 *     atom_masks[na_grid][mask_words]
 *
 * 但实际采用一维连续数组保存：
 *
 *     atom_masks[ia * mask_words + word_index]
 *
 * 这种布局的优点：
 * 1. 不需要为每个原子单独分配 vector；
 * 2. 所有掩码连续存储，缓存局部性更好；
 * 3. 后续原子对求交只需要按位与操作；
 * 4. 自动兼容 bxyz>128 的情况。
 */
inline void build_atom_masks(
    const int na_grid,
    const int bxyz,
    const int mask_words,
    const bool* const* const cal_flag,
    std::uint64_t* const atom_masks)
{
    // 位掩码总字数 = 原子数 × 每个原子所需的64位字数。
    const std::size_t total_words
        = static_cast<std::size_t>(na_grid) * mask_words;

    // thread_local 缓冲区会在不同网格块之间复用，因此每次构造前
    // 必须将当前有效区域清零，避免残留上一个网格块的置位比特。
    std::fill(
        atom_masks,
        atom_masks + total_words,
        std::uint64_t{0});

    // 逐网格点读取 cal_flag。由于 cal_flag 的第一维是 ib，
    // 该循环顺序保持了原数据在原子方向上的连续访问。
    for (int ib = 0; ib < bxyz; ++ib)
    {
        // 当前网格点属于第几个64位字。
        const int word_index = ib / 64;

        // 当前网格点在该64位字中的比特位置。
        const int bit_index = ib % 64;

        // 构造仅有第 bit_index 位为1的比特掩码。
        const std::uint64_t bit
            = std::uint64_t{1} << bit_index;

        for (int ia = 0; ia < na_grid; ++ia)
        {
            if (cal_flag[ib][ia])
            {
                // 将原子 ia 对应掩码中的当前网格点位置置1。
                atom_masks[
                    static_cast<std::size_t>(ia) * mask_words
                    + word_index] |= bit;
            }
        }
    }
}

/**
 * @brief 使用位掩码分析两个原子的共同有效网格点。
 *
 * 对两个原子 ia1、ia2，其共同有效网格点集合为：
 *
 *     pair_mask = atom_mask[ia1] & atom_mask[ia2]
 *
 * 一个按位与操作可以同时处理64个网格点。
 *
 * 本函数计算：
 * 1. first_ib：第一个共同有效网格点；
 * 2. last_ib：最后一个共同有效网格点的下一位置；
 * 3. cal_pair_num：共同有效网格点总数。
 *
 * 这些量与原代码通过三次 cal_flag 扫描得到的结果完全一致。
 *
 * 本轮优化在上一版“位掩码统计”的基础上进一步扩展：
 * - 稠密/稀疏判断阈值不变；
 * - 稠密分支完全不变；
 * - 稀疏分支直接遍历原子对交集掩码中的置位位；
 * - 稀疏分支仍然对每个有效网格点调用一次 k=1 的 DGEMM；
 * - 有效网格点集合、DGEMM 调用次数和调用顺序保持不变。
 */
inline bool analyse_pair_mask(
    const std::uint64_t* const mask1,
    const std::uint64_t* const mask2,
    const int mask_words,
    int& first_ib,
    int& last_ib,
    int& cal_pair_num)
{
    // 记录第一个和最后一个非零交集字的位置。
    int first_word = -1;
    int last_word = -1;

    // 保存第一个和最后一个非零交集字的具体比特值，
    // 后续分别使用 ctz 和 clz 定位首尾有效网格点。
    std::uint64_t first_intersection = 0;
    std::uint64_t last_intersection = 0;

    cal_pair_num = 0;

    // 逐64位字分析两个原子掩码的交集。
    // 当前 bxyz=75 时该循环仅执行2次。
    for (int iw = 0; iw < mask_words; ++iw)
    {
        // 按位与后，值为1的比特即两个原子共同有效的网格点。
        const std::uint64_t intersection
            = mask1[iw] & mask2[iw];

        // 当前64位范围内不存在共同有效点。
        if (intersection == 0)
        {
            continue;
        }

        // 一次 popcount 统计当前64位范围内的共同有效点数量。
        cal_pair_num += popcount64(intersection);

        // 第一次遇到非零交集字时，记录其位置和值。
        if (first_word < 0)
        {
            first_word = iw;
            first_intersection = intersection;
        }

        // 每次遇到非零交集字都更新，因此循环结束后
        // last_word 指向最后一个非零交集字。
        last_word = iw;
        last_intersection = intersection;
    }

    // 没有发现任何非零交集字，说明两个原子没有共同有效网格点。
    if (first_word < 0)
    {
        first_ib = 0;
        last_ib = 0;
        cal_pair_num = 0;
        return false;
    }

    // ctz 返回第一个非零交集字中最低置位比特的位置。
    // 加上该64位字在全局网格点序列中的起始偏移，
    // 得到第一个共同有效网格点 first_ib。
    first_ib
        = first_word * 64
        + ctz64(first_intersection);

    // clz 用于定位最后一个非零交集字中的最高置位比特。
    const int highest_set_bit
        = 63 - clz64(last_intersection);

    // 保持与原代码一致，last_ib 使用半开区间右端点，
    // 即“最后一个有效网格点索引 + 1”。
    last_ib
        = last_word * 64
        + highest_set_bit
        + 1;

    return true;
}

} // 匿名命名空间

void Gint::cal_meshball_vlocal(
    const int na_grid,                          // 当前网格块中的原子数
    const int LD_pool,
    const int* const block_size,                // 各原子的轨道数
    const int* const block_index,               // 各原子轨道在轨道池中的起始位置
    const int grid_index,
    const bool* const* const cal_flag,           // cal_flag[bxyz][na_grid]
    const double* const* const psir_right_T,     // 已转置右面板 [LD_pool][bxyz]
    const double* const* const psir_left_T,      // 已转置左面板 [LD_pool][bxyz]
    hamilt::HContainer<double>* hR)
{
    const int bxyz_local = this->bxyz;

    if (na_grid <= 0 || LD_pool <= 0 || bxyz_local <= 0)
    {
        return;
    }

    // ---------------------------------------------------------------------
    // 第一步：直接消费上层生成的两个转置面板
    // ---------------------------------------------------------------------
    //
    // psir_right_T 和 psir_left_T 的逻辑布局均为：
    //
    //     [LD_pool][bxyz]
    //
    // 即每条轨道在网格点方向连续存储。调用者已经完成：
    //
    // 1. 右面板的布局转换；
    // 2. 左面板的局域势缩放与布局转换。
    //
    // 因此本函数中不再分配完整转置缓冲区，也不再执行显式转置。
    const int ldt = bxyz_local;

    // ---------------------------------------------------------------------
    // 第二步：构造动态长度位掩码
    // ---------------------------------------------------------------------
    //
    // 向上取整计算每个原子需要多少个64位字：
    //
    // bxyz=1~64    -> mask_words=1
    // bxyz=65~128  -> mask_words=2
    // bxyz=129~192 -> mask_words=3
    //
    // 当前 Si512 算例 bxyz=75，因此每个原子使用2个 uint64_t。
    const int mask_words
        = (bxyz_local + 63) / 64;

    // 每个 OpenMP 线程拥有独立的位掩码缓冲区，
    // 避免线程之间的数据竞争。
    // 缓冲区只在容量不足时扩容，避免每个网格块反复申请内存。
    thread_local std::vector<std::uint64_t> atom_masks;

    const std::size_t required_mask_words
        = static_cast<std::size_t>(na_grid) * mask_words;

    if (atom_masks.size() < required_mask_words)
    {
        atom_masks.resize(required_mask_words);
    }

    build_atom_masks(
        na_grid,
        bxyz_local,
        mask_words,
        cal_flag,
        atom_masks.data());

    // ---------------------------------------------------------------------
    // 第三步：直接使用两个转置面板执行原有稠密/稀疏路径
    // ---------------------------------------------------------------------
    const char transa = 'T';
    const char transb = 'N';
    const double alpha = 1.0;
    const double beta = 1.0;

    const int mcell_index
        = this->gridt->bcell_start[grid_index];

    for (int ia1 = 0; ia1 < na_grid; ++ia1)
    {
        const int bcell1 = mcell_index + ia1;
        const int iat1 = this->gridt->which_atom[bcell1];
        const int id1 = this->gridt->which_unitcell[bcell1];
        const ModuleBase::Vector3<int> r1
            = this->gridt->get_ucell_coords(id1);

        // 当前原子 ia1 的位掩码起始地址。
        // mask1[0...mask_words-1] 表示该原子的全部网格点有效性。
        const std::uint64_t* const mask1
            = atom_masks.data()
            + static_cast<std::size_t>(ia1) * mask_words;

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

            // 当前原子 ia2 的位掩码起始地址。
            const std::uint64_t* const mask2
                = atom_masks.data()
                + static_cast<std::size_t>(ia2) * mask_words;

            int first_ib = 0;
            int last_ib = 0;
            int cal_pair_num = 0;

            // 使用位掩码一次性得到：
            // 1. 是否存在共同有效网格点；
            // 2. 首个有效网格点；
            // 3. 末尾有效网格点的下一位置；
            // 4. 有效网格点总数。
            //
            // 这替代了原代码针对每个原子对的多次 cal_flag 扫描。
            const bool has_overlap
                = analyse_pair_mask(
                    mask1,
                    mask2,
                    mask_words,
                    first_ib,
                    last_ib,
                    cal_pair_num);

            if (!has_overlap)
            {
                continue;
            }

            const int ib_length
                = last_ib - first_ib;

            const auto tmp_matrix
                = hR->find_matrix(
                    iat1,
                    iat2,
                    r1 - r2);

            if (tmp_matrix == nullptr)
            {
                continue;
            }

            const int m
                = tmp_matrix->get_row_size();

            const int n
                = tmp_matrix->get_col_size();

            // 两个输入均已经是 [LD_pool][bxyz] 布局。
            // block_index 直接定位相应原子的第一条轨道行。
            const double* const left_block
                = psir_left_T[block_index[ia2]];

            const double* const right_block
                = psir_right_T[block_index[ia1]];

            // 完全保留原代码的稀疏/稠密判断条件。
            //
            // 位掩码只负责更快地获得 cal_pair_num 和 ib_length，
            // 不改变原有算法选择策略。
            if (cal_pair_num > ib_length / 4)
            {
                // 稠密分支：一次 DGEMM 计算整个首尾区间。
                const double* const ptr_a
                    = left_block
                    + first_ib;

                const double* const ptr_b
                    = right_block
                    + first_ib;

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
                /*
                 * 稀疏分支：
                 *
                 * 不再扫描 [first_ib, last_ib) 区间，也不再逐点读取
                 * cal_flag[ib][ia1] 和 cal_flag[ib][ia2]。
                 *
                 * 对每个64位字计算两个原子掩码的交集：
                 *
                 *     pair_word = mask1[iw] & mask2[iw]
                 *
                 * pair_word 中值为1的比特就是共同有效的网格点。
                 * 每次使用 ctz64 找到最低置位比特，并通过：
                 *
                 *     pair_word &= pair_word - 1
                 *
                 * 清除该置位比特。
                 *
                 * mask_words 按递增顺序遍历，每个字内部也始终先处理
                 * 最低置位比特，所以有效网格点 ib 的处理顺序仍然严格递增，
                 * 与原来从 first_ib 扫描到 last_ib 的调用顺序一致。
                 */
                for (int iw = 0; iw < mask_words; ++iw)
                {
                    std::uint64_t pair_word
                        = mask1[iw] & mask2[iw];

                    while (pair_word != 0)
                    {
                        // 找到当前64位字中最低的置位比特。
                        const int bit_index
                            = ctz64(pair_word);

                        // 转换为当前网格块中的全局网格点编号。
                        const int ib
                            = iw * 64 + bit_index;

                        const int k = 1;

                        const double* const ptr_a
                            = left_block
                            + ib;

                        const double* const ptr_b
                            = right_block
                            + ib;

                        /*
                         * 保持原稀疏分支的计算方式：
                         * 每个共同有效网格点仍执行一次 k=1 的 DGEMM，
                         * 并直接累加到 tmp_matrix。
                         */
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

                        // 清除最低置位比特，继续处理下一个有效网格点。
                        pair_word &= pair_word - 1;
                    }
                }
            }
        }
    }
}
