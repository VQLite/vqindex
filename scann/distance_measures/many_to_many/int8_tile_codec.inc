template <Int8TileSide kSide>
class Int8TileCodecImpl : public Int8TileCodec {
  using Int8Tile = std::conditional_t<kSide == Int8TileSide::kQuery,
                                      Int8QueryTile, Int8DatabaseTile>;
  static constexpr bool kUnsigned =
      IsSame<Int8Tile, avx512_vnni::Int8DatabaseTile>();
  static constexpr int kUnsignedOffset = kUnsigned ? 128 : 0;
  static constexpr uint32_t kUnsignedOffsetsX4 =
      kUnsignedOffset | (kUnsignedOffset << 8) | (kUnsignedOffset << 16) |
      (kUnsignedOffset << 24);

  static constexpr size_t kContiguousDims = []() -> size_t {
    if (IsSame<Int8Tile, amx::Int8DatabaseTile>()) return 4;
    if (Int8Tile::kPoints == 1) return 0;
    return Int8Tile::kDims;
  }();

 public:
  explicit Int8TileCodecImpl(uint32_t dimensionality)
      : Int8TileCodec(dimensionality,
                      NextMultipleOf(dimensionality, Int8Tile::kDims),
                      Int8Tile::kPoints, Simd<int8_t>::kRegisterBytes,
                      kPlatformGeneration, kSide) {}

  void EncodeDatapoint(ConstSpan<int8_t> datapoint, size_t dp_idx,
                       MutableSpan<int8_t> payload) const override {
    DCHECK_EQ(datapoint.size(), dimensionality());
    size_t block_start = BlockStart(dp_idx);
    auto block = payload.subspan(block_start, block_bytes_);
    for (size_t dim_idx : IndicesOf(datapoint)) {
      block[PayloadIndex(dp_idx, dim_idx)] =
          datapoint[dim_idx] + kUnsignedOffset;
    }
  }

  void ReconstructDatapoint(size_t dp_idx, ConstSpan<int8_t> payload,
                            MutableSpan<int8_t> encoded) const override {
    DCHECK_EQ(encoded.size(), dimensionality());
    auto block = payload.subspan(BlockStart(dp_idx), block_bytes_);
    const size_t dims = dimensionality_;
    if constexpr (kContiguousDims == 0) {
      static_assert(kUnsignedOffset == 0);
      memcpy(encoded.data(), block.data(), dims);
      return;
    }
    DimensionIndex dim_idx = 0;
#pragma unroll 4
    for (; dim_idx + kContiguousDims <= dims; dim_idx += kContiguousDims) {
      memcpy(&encoded[dim_idx], &block[PayloadIndex(dp_idx, dim_idx)],
             kContiguousDims);
      if constexpr (kUnsignedOffset) {
        static_assert(kContiguousDims == 4);

        static_assert(kUnsignedOffset == 0b10000000);

        *reinterpret_cast<uint32_t*>(&encoded[dim_idx]) ^= kUnsignedOffsetsX4;
      }
    }
    for (; dim_idx < dims; ++dim_idx) {
      encoded[dim_idx] = block[PayloadIndex(dp_idx, dim_idx)] - kUnsignedOffset;
    }
  }

  void ReconstructFloatDatapoint(size_t dp_idx, ConstSpan<int8_t> payload,
                                 float scale,
                                 MutableSpan<float> encoded) const override {
    DCHECK_EQ(encoded.size(), dimensionality());
    auto block = payload.subspan(BlockStart(dp_idx), block_bytes_);
    for (DimensionIndex dim_idx : Seq(dimensionality_)) {
      int value = block[PayloadIndex(dp_idx, dim_idx)];
      if (kUnsigned) {
        value = static_cast<uint8_t>(value) - kUnsignedOffset;
      }
      encoded[dim_idx] = value * scale;
    }
  }

 private:
  size_t BlockStart(DatapointIndex dp_idx) const {
    return block_bytes_ * (dp_idx / Int8Tile::kPoints);
  }

  size_t PayloadIndex(DatapointIndex dp_idx, DimensionIndex dim_idx) const {
    const size_t tile_idx = dim_idx / Int8Tile::kDims;
    const size_t tile_dp_idx = dp_idx % Int8Tile::kPoints;
    const size_t tile_dim_idx = dim_idx % Int8Tile::kDims;

    size_t tile_row;
    size_t tile_col;
    if constexpr (IsSame<Int8Tile, amx::Int8DatabaseTile>()) {
      tile_row = tile_dim_idx / 4;
      tile_col = tile_dp_idx * 4 + tile_dim_idx % 4;
    } else {
      tile_row = tile_dp_idx;
      tile_col = tile_dim_idx;
    }
    return tile_idx * Int8TileBytes<Int8Tile>() + tile_row * Int8Tile::kDims +
           tile_col;
  }
};

inline std::unique_ptr<Int8TileCodec> NewInt8TileCodec(size_t dims,
                                                       Int8TileSide side) {
  switch (side) {
    case Int8TileSide::kQuery:
      return std::make_unique<Int8TileCodecImpl<Int8TileSide::kQuery>>(dims);
    case Int8TileSide::kDatabase:
      return std::make_unique<Int8TileCodecImpl<Int8TileSide::kDatabase>>(dims);
  }
}
