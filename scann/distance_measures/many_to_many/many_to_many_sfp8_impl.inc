

struct ManyToManySFP8Args_DotProduct {
  size_t datapoint_bytes;
  size_t query_block_bytes;
  const int8_t* int8_queries;
  Int8QueryExpander* int8_query_expander;
  const int8_t* database_block0;
  const int8_t* database_block1;
  const float* query_scales;
  const int32_t* query_sums;
  const float* db_scales;
  size_t first_q_idx;
  size_t first_dp_idx;
  size_t num_datapoints;
};

struct ManyToManySFP8Args_SquaredL2 : ManyToManySFP8Args_DotProduct {
  const float* query_squared_l2_norms;
  const float* database_squared_l2_norms;
};

struct ManyToManySFP8Args_OrthogonalityAmplified
    : ManyToManySFP8Args_SquaredL2 {
  const int8_t* int8_residuals;
  Int8QueryExpander* int8_residuals_expander;
  const float* residual_scales;
  const int32_t* residual_sums;
  const float* r_dot_qs;
  float lambda;
};

template <
    typename Args, typename CallbackT,
    typename Base = DenseManyToManyTransposedBase<
        float, std::is_same_v<Args, ManyToManySFP8Args_OrthogonalityAmplified>>>
class DenseManyToManySFP8Pretransposed final : public Base {
 public:
  using Base::kElementsPerRegister;
  using Base::kResultsSize;
  using Base::kSmallQueryStride;

  static constexpr bool kNeedsSquaredL2 =
      std::is_base_of_v<ManyToManySFP8Args_SquaredL2, Args>;
  static constexpr bool kIsOrthogonalityAmplified =
      std::is_same_v<ManyToManySFP8Args_OrthogonalityAmplified, Args>;

  SCANN_INLINE DenseManyToManySFP8Pretransposed(
      const SFP8SimdBlockTransposedDatabase& queries,
      const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
      CallbackT callback)
      : Base(queries, database.size(), pool),
        queries_(queries),
        database_(database),
        callback_(std::move(callback)) {
    static_assert(Int8QueryTile::kDims == Int8DatabaseTile::kDims);
    static_assert(kSmallQueryStride % Int8QueryTile::kPoints == 0);
    static_assert(kElementsPerRegister == Int8DatabaseTile::kPoints);
    CHECK_EQ(queries.dimensionality(), database.dimensionality());
    CHECK_EQ(queries.datapoint_bytes(), database.datapoint_bytes());
    CHECK_EQ(queries.platform_generation(), kPlatformGeneration);
    CHECK_EQ(database.platform_generation(), kPlatformGeneration);
  }

  SCANN_INLINE DenseManyToManySFP8Pretransposed(
      const SFP8SimdBlockTransposedDatabase& queries,
      const SFP8SimdBlockTransposedDatabase& residuals, float lambda,
      const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
      CallbackT callback)
      : DenseManyToManySFP8Pretransposed(queries, database, pool,
                                         std::move(callback)) {
    residuals_ = &residuals;
    lambda_ = lambda;
    static_assert(kIsOrthogonalityAmplified);
    CHECK_EQ(residuals.dimensionality(), queries.dimensionality());
    CHECK_EQ(residuals.datapoint_bytes(), queries.datapoint_bytes());
    CHECK_EQ(residuals.platform_generation(), kPlatformGeneration);
    CHECK_EQ(residuals.size(), queries.size());

    std::vector<uint8_t> q(queries.hashed_space_bytes());
    std::vector<uint8_t> r(residuals.hashed_space_bytes());
    r_dot_qs_ = std::make_unique<float[]>(queries.size());
    const size_t dims = queries.dimensionality();
    for (const size_t i : IndicesOf(queries)) {
      CHECK_OK(residuals.ReconstructDatapoint(i, MakeMutableSpan(r)));
      const float r_scale = residuals.scales()[i];
      CHECK_OK(queries.ReconstructDatapoint(i, MakeMutableSpan(q)));
      const float q_scale = queries.scales()[i];
      r_dot_qs_[i] = SFP8DotProduct(dims, r, r_scale, q, q_scale);
    }
  }

  SCANN_SIMD_OUTLINE void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                                        size_t first_dp_idx,
                                        size_t num_datapoints) final {
    const size_t q_idx_end = first_q_idx + num_queries;
    Args args;
    args.datapoint_bytes = this->queries_.datapoint_bytes();
    args.query_block_bytes = this->queries_.block_bytes();
    args.int8_queries = this->queries_.payload().data();
    args.database_block0 =
        this->database_.payload().data() + first_dp_idx * args.datapoint_bytes;
    args.database_block1 = args.database_block0 + this->database_.block_bytes();
    args.query_scales = this->queries_.scales().data();
    args.query_sums = this->queries_.sums().data();
    args.db_scales = this->database_.scales().data() + first_dp_idx;
    args.first_q_idx = first_q_idx;
    args.first_dp_idx = first_dp_idx;
    args.num_datapoints = num_datapoints;
    CallbackT& callback = this->callback_;

    constexpr size_t kMaxNumQueryBlocks =
        DivRoundUp(kSmallQueryStride, Int8QueryTile::kPoints);
    thread_local Int8QueryExpander query_expander;
    query_expander.Ensure(kMaxNumQueryBlocks * args.query_block_bytes);
    args.int8_query_expander = &query_expander;
    if constexpr (kNeedsSquaredL2) {
      args.query_squared_l2_norms = this->queries_.squared_l2_norms().data();
      args.database_squared_l2_norms =
          this->database_.squared_l2_norms().data() + first_dp_idx;
    }
    thread_local Int8QueryExpander residuals_expander;
    if constexpr (kIsOrthogonalityAmplified) {
      residuals_expander.Ensure(kMaxNumQueryBlocks * args.query_block_bytes);
      args.int8_residuals_expander = &residuals_expander;
      args.int8_residuals = this->residuals_->payload().data();
      args.residual_scales = this->residuals_->scales().data();
      args.residual_sums = this->residuals_->sums().data();
      args.r_dot_qs = this->r_dot_qs_.get();
      args.lambda = this->lambda_;
    }
    for (; args.first_q_idx + kSmallQueryStride <= q_idx_end;
         args.first_q_idx += kSmallQueryStride) {
      BottomLevelBatch<kSmallQueryStride>(args, callback);
    }

    const size_t final_batch_size = q_idx_end - args.first_q_idx;
    SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE(kSmallQueryStride, final_batch_size,
                                         BottomLevelBatch, args, callback);
  }

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static void BottomLevelBatch(Args args,
                                                 CallbackT& callback) {
    auto accumulators = DoAccumulationTransposedTemplate<kNumQueries>(args);
    Base::template PassResultsToCallback<kNumQueries>(
        accumulators, callback, args.first_q_idx, args.first_dp_idx,
        args.num_datapoints);
  }

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static Simd<float, kNumQueries, 2>
  DoAccumulationTransposedTemplate(Args args) {
    constexpr size_t kNumQueryBlocks =
        DivRoundUp(kNumQueries, Int8QueryTile::kPoints);

    using QueryT = typename Int8QueryExpander::ExpandedQueryT;
    const QueryT* queries = args.int8_query_expander->Expand(MakeConstSpan(
        args.int8_queries + args.first_q_idx * args.datapoint_bytes,
        kNumQueryBlocks * args.query_block_bytes));

    const QueryT* volatile query_blocks_vol[kNumQueryBlocks];
    const QueryT* query_blocks[kNumQueryBlocks];
    for (size_t q_b : Seq(kNumQueryBlocks)) {
      query_blocks_vol[q_b] = queries + q_b * args.query_block_bytes;
      query_blocks[q_b] = query_blocks_vol[q_b];
    }

    const QueryT* residuals;
    const QueryT* volatile residual_blocks_vol[kNumQueryBlocks];
    const QueryT* residual_blocks[kNumQueryBlocks];
    if constexpr (kIsOrthogonalityAmplified) {
      residuals = args.int8_residuals_expander->Expand(MakeConstSpan(
          args.int8_residuals + args.first_q_idx * args.datapoint_bytes,
          kNumQueryBlocks * args.query_block_bytes));
      for (size_t r_b : Seq(kNumQueryBlocks)) {
        residual_blocks_vol[r_b] = residuals + r_b * args.query_block_bytes;
        residual_blocks[r_b] = residual_blocks_vol[r_b];
      }
    }

    Int32AccumulatorTile accumulators[2][kNumQueryBlocks];
    Int32AccumulatorTile residual_accumulators[2][kNumQueryBlocks];

    constexpr size_t kQueryTileBytes = Int8TileBytes<Int8QueryTile>();
    constexpr size_t kDatapointTileBytes = Int8TileBytes<Int8DatabaseTile>();
    constexpr size_t kTileBytesRatio = kDatapointTileBytes / kQueryTileBytes;
    static_assert(kDatapointTileBytes % kQueryTileBytes == 0);

    for (size_t query_pos :
         SeqWithStride<kQueryTileBytes>(0, args.query_block_bytes)) {
      Int8DatabaseTile database_tile[2];
      database_tile[0].Load(args.database_block0 + query_pos * kTileBytesRatio);
      database_tile[1].Load(args.database_block1 + query_pos * kTileBytesRatio);

      for (size_t q_b : Seq(kNumQueryBlocks)) {
        Int8QueryTile query_tile;
        query_tile.Load(query_blocks[q_b] + query_pos);

        accumulators[0][q_b].AccumulateDotProducts(query_tile,
                                                   database_tile[0]);
        accumulators[1][q_b].AccumulateDotProducts(query_tile,
                                                   database_tile[1]);

        if constexpr (kIsOrthogonalityAmplified) {
          Int8QueryTile residual_tile;
          residual_tile.Load(residual_blocks[q_b] + query_pos);
          residual_accumulators[0][q_b].AccumulateDotProducts(residual_tile,
                                                              database_tile[0]);
          residual_accumulators[1][q_b].AccumulateDotProducts(residual_tile,
                                                              database_tile[1]);
        }
      }
    }

    Simd<int32_t, 2, kNumQueryBlocks, Int8QueryTile::kPoints> int32_results;
    auto query_sums = args.query_sums + args.first_q_idx;
    for (size_t q_b : Seq(kNumQueryBlocks)) {
      accumulators[0][q_b].GetResult(int32_results[0][q_b], query_sums);
      accumulators[1][q_b].GetResult(int32_results[1][q_b], query_sums);
      query_sums += Int8QueryTile::kPoints;
    }
    Simd<int32_t, 2, kNumQueryBlocks, Int8QueryTile::kPoints>
        residual_int32_results;
    if constexpr (kIsOrthogonalityAmplified) {
      auto residual_sums = args.residual_sums + args.first_q_idx;
      for (size_t q_b : Seq(kNumQueryBlocks)) {
        residual_accumulators[0][q_b].GetResult(residual_int32_results[0][q_b],
                                                residual_sums);
        residual_accumulators[1][q_b].GetResult(residual_int32_results[1][q_b],
                                                residual_sums);
        residual_sums += Int8QueryTile::kPoints;
      }
    }

    Simd<float, kNumQueries, 2> float_results;
    auto db_scales = Simd<float, 2>::Load(args.db_scales);

    auto db_multipliers = Simd<float>::Zeros() - db_scales;
    Simd<float, 2> db_squared_l2_norms;
    if constexpr (kNeedsSquaredL2) {
      db_squared_l2_norms =
          Simd<float, 2>::Load(args.database_squared_l2_norms);

      db_multipliers += db_multipliers;
    }

    Simd<float> lambdas;
    if constexpr (kIsOrthogonalityAmplified) {
      lambdas = Simd<float>::Broadcast(args.lambda);
    }
    size_t query_idx = 0;
    for (size_t q_b : Seq(kNumQueryBlocks)) {
      for (size_t p : Seq(Int8QueryTile::kPoints)) {
        Simd<int32_t, 2> int32_result;
        int32_result[0] = int32_results[0][q_b][p];
        int32_result[1] = int32_results[1][q_b][p];
        const auto query_scale = Simd<float>::Broadcast(
            args.query_scales[args.first_q_idx + query_idx]);
        auto float_result =
            int32_result.ConvertTo<float>() * db_multipliers * query_scale;
        if constexpr (kNeedsSquaredL2) {
          float_result +=
              db_squared_l2_norms +
              Simd<float>::Broadcast(
                  args.query_squared_l2_norms[args.first_q_idx + query_idx]);
        }

        if constexpr (kIsOrthogonalityAmplified) {
          Simd<int32_t, 2> residual_int32_result;
          residual_int32_result[0] = residual_int32_results[0][q_b][p];
          residual_int32_result[1] = residual_int32_results[1][q_b][p];
          const auto residual_scale = Simd<float>::Broadcast(
              args.residual_scales[args.first_q_idx + query_idx]);
          auto r_q = Simd<float>::Broadcast(
              args.r_dot_qs[args.first_q_idx + query_idx]);
          auto r_d = residual_int32_result.ConvertTo<float>() * db_scales *
                     residual_scale;

          auto r_q_r_d = r_q * r_d;
          float_result +=
              (r_q * r_q + r_d * r_d - (r_q_r_d + r_q_r_d)) * lambdas;
        }
        float_results[query_idx++] = float_result;
        if (query_idx == kNumQueries) break;
      }
    }
    return float_results;
  }

 private:
  const SFP8SimdBlockTransposedDatabase& queries_;
  const SFP8SimdBlockTransposedDatabase& database_;
  CallbackT callback_;

  const SFP8SimdBlockTransposedDatabase* residuals_ = nullptr;
  std::unique_ptr<float[]> r_dot_qs_;
  float lambda_ = 0.0f;
};

template <typename CallbackT>
SCANN_OUTLINE void DenseManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  const bool is_squared_l2 =
      (DistanceMeasure::SQUARED_L2 == dist.specially_optimized_distance_tag());
  if (is_squared_l2) {
    DenseManyToManySFP8Pretransposed<ManyToManySFP8Args_SquaredL2, CallbackT>(
        queries, database, pool, std::move(callback))
        .TopLevelBatch();
  } else {
    DCHECK_EQ(dist.specially_optimized_distance_tag(),
              DistanceMeasure::DOT_PRODUCT);
    DenseManyToManySFP8Pretransposed<ManyToManySFP8Args_DotProduct, CallbackT>(
        queries, database, pool, std::move(callback))
        .TopLevelBatch();
  }
}

template <typename CallbackT>
SCANN_OUTLINE Status DenseManyToManySFP8OrthogonalityAmplifiedImpl(
    const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& normalized_residuals, float lambda,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  SCANN_RET_CHECK_EQ(queries.side(), Int8TileSide::kQuery);
  SCANN_RET_CHECK_EQ(normalized_residuals.side(), Int8TileSide::kQuery);
  SCANN_RET_CHECK_EQ(database.side(), Int8TileSide::kDatabase);
  SCANN_RET_CHECK_EQ(queries.size(), normalized_residuals.size());
  SCANN_RET_CHECK_EQ(queries.dimensionality(),
                     normalized_residuals.dimensionality());
  SCANN_RET_CHECK_EQ(queries.dimensionality(), database.dimensionality());
  SCANN_RET_CHECK_EQ(queries.platform_generation(), kPlatformGeneration);
  SCANN_RET_CHECK_EQ(normalized_residuals.platform_generation(),
                     kPlatformGeneration);
  SCANN_RET_CHECK_EQ(database.platform_generation(), kPlatformGeneration);

  if (queries.empty()) return OkStatus();
  if (database.empty()) return OkStatus();

  DenseManyToManySFP8Pretransposed<ManyToManySFP8Args_OrthogonalityAmplified,
                                   CallbackT>(queries, normalized_residuals,
                                              lambda, database, pool,
                                              std::move(callback))
      .TopLevelBatch();
  return OkStatus();
}
