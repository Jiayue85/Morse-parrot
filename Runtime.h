#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "ModelData.h"

// No Arduino dependency: compile the SAME inference code on the Spark and Teensy.
namespace sd_student {
using namespace model_data;
constexpr int kPad = 256, kBos = 257, kEos = 258, kUnknown = 259;
constexpr int kEncSize = 2 * kEncoder;
constexpr int kGateSize = 3 * (kDecoder > kEncoder ? kDecoder : kEncoder);
constexpr int kInputSize = kEmbedding + kEncSize + kDecoder;

class Reader {
 public:
  virtual ~Reader() {}
  virtual bool read(uint32_t offset, void* output, size_t count) = 0;
  virtual uint32_t size() const = 0;
};

class WeightCache {
 public:
  explicit WeightCache(Reader& reader) : reader_(reader) { reset(); }
  void reset() { start_ = UINT32_MAX; valid_ = 0; physical_bytes = 0; misses = 0; }
  bool read(uint32_t offset, void* destination, size_t count) {
    auto* out = static_cast<uint8_t*>(destination);
    if (offset > reader_.size() || count > reader_.size() - offset) return false;
    while (count) {
      const uint8_t* bytes; size_t available;
      if (!window(offset, bytes, available)) return false;
      const size_t chunk = count < available ? count : available;
      memcpy(out, bytes, chunk); offset += chunk; out += chunk; count -= chunk;
    }
    return true;
  }
  bool window(uint32_t offset, const uint8_t*& bytes, size_t& available) {
    if (offset >= reader_.size()) return false;
    const uint32_t start = offset / kCacheBytes * kCacheBytes;
    if (start_ != start) {
      valid_ = reader_.size() - start;
      if (valid_ > kCacheBytes) valid_ = kCacheBytes;
      if (!reader_.read(start, block_, valid_)) { start_ = UINT32_MAX; return false; }
      start_ = start; physical_bytes += valid_; ++misses;
    }
    bytes = block_ + offset - start_;
    available = valid_ - (offset - start_);
    return available > 0;
  }
  bool verify() {
    uint8_t header[32];
    if (reader_.size() != kDataOffset + kWeightBytes || !read(0, header, sizeof(header))) return false;
    if (memcmp(header, "TSYSD01", 8) || u32(header + 8) != 1 || u32(header + 12) != kMetadataBytes ||
        u32(header + 16) != kDataOffset || u32(header + 20) != kWeightBytes ||
        u32(header + 24) != kCrc32 || u32(header + 28) != 0) return false;
    uint32_t crc = 0xFFFFFFFFu;
    for (uint32_t offset = 32; offset < reader_.size();) {
      const uint8_t* bytes; size_t available;
      if (!window(offset, bytes, available)) return false;
      for (size_t index = 0; index < available; ++index) {
        crc ^= bytes[index];
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
      }
      offset += available;
    }
    reset();
    return (crc ^ 0xFFFFFFFFu) == kCrc32;
  }
  uint64_t physical_bytes;
  uint32_t misses;
 private:
  static uint32_t u32(const uint8_t* b) { return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24; }
  Reader& reader_;
  uint32_t start_;
  size_t valid_;
  uint8_t block_[kCacheBytes];
};

struct Result {
  const char* status = "not_run";
  int category = 0, intent = 0, source_tokens = 0, generated_tokens = 0;
  float category_score = 0, intent_score = 0;
  bool complete = false;
};

struct Workspace {
  float encoded[kSource][kEncSize];
  float hidden[kLayers][kDecoder];
  float initial[kLayers * kDecoder];
  float encoder_hidden[kEncoder];
  float final_state[kEncSize], context[kEncSize], query[kEncSize];
  float embedded[kEmbedding], input[kInputSize];
  float gi[kGateSize], gh[kGateSize];
  float attention[kSource], probabilities[kVocab], generated[kVocab];
  uint16_t source[kSource], output[kTarget], bpe[2048];
  char summary[2049];
};

class Runtime {
 public:
  explicit Runtime(WeightCache& cache) : cache_(cache) {}
  Workspace work;

  Result run(const char* text) {
    Result result;
    work.summary[0] = 0;
    cache_.reset();
    const int count = tokenize(text);
    if (count <= 0) { result.status = count == 0 ? "empty_input" : "source_limit"; return result; }
    result.source_tokens = count;
    if (!encode(count)) { result.status = "read_error"; return result; }
    if (!matvec(m_category_weight, work.final_state, work.probabilities)) { result.status = "read_error"; return result; }
    softmax(work.probabilities, m_category_weight.rows);
    result.category = argmax(work.probabilities, m_category_weight.rows);
    result.category_score = work.probabilities[result.category];
    if (!matvec(m_intent_weight, work.final_state, work.probabilities)) { result.status = "read_error"; return result; }
    softmax(work.probabilities, m_intent_weight.rows);
    result.intent = argmax(work.probabilities, m_intent_weight.rows);
    result.intent_score = work.probabilities[result.intent];
    memset(work.context, 0, sizeof(work.context));
    int previous = kBos;
    bool ended = false;
    for (int position = 0; position < kTarget; ++position) {
      if (!step(previous, count)) { result.status = "read_error"; return result; }
      work.probabilities[kPad] = work.probabilities[kBos] = work.probabilities[kUnknown] = -1;
      previous = argmax(work.probabilities, kVocab);
      if (previous == kEos) { ended = true; break; }
      work.output[result.generated_tokens++] = uint16_t(previous);
    }
    size_t used = 0;
    for (int index = 0; index < result.generated_tokens; ++index) {
      if (!append_token(work.output[index], used, 0)) { work.summary[0] = 0; result.status = "decode_limit"; return result; }
    }
    work.summary[used] = 0;
    if (!valid_utf8(reinterpret_cast<const uint8_t*>(work.summary))) { work.summary[0] = 0; result.status = "invalid_utf8"; return result; }
    int words = 0; bool space = true;
    for (size_t index = 0; index < used; ++index) {
      const bool current = whitespace(uint8_t(work.summary[index]));
      if (!current && space) ++words;
      space = current;
    }
    result.complete = ended && words > 0 && words <= kWords;
    result.status = !ended ? "token_limit" : words > kWords ? "word_limit" : words == 0 ? "empty_summary" : "ok";
    return result;
  }

  int tokenize(const char* text) {
    const size_t size = strlen(text);
    if (size > 2048) return -1;
    int count = 0; size_t position = 0; bool have_word = false;
    while (position < size) {
      if (whitespace(uint8_t(text[position]))) {
        while (position < size && whitespace(uint8_t(text[position]))) ++position;
        if (position < size && have_word) {
          if (count >= kSource) return -1;
          work.source[count++] = 32;
        }
        continue;
      }
      int length = 0;
      while (position < size && !whitespace(uint8_t(text[position]))) {
        if (length >= 2048) return -1;
        work.bpe[length++] = uint8_t(text[position++]);
      }
      for (;;) {
        int best = kMergeCount, left = -1, right = -1;
        for (int index = 0; index + 1 < length; ++index) {
          for (int rank = 0; rank < best; ++rank) {
            if (work.bpe[index] == kMerges[rank][0] && work.bpe[index + 1] == kMerges[rank][1]) {
              best = rank; left = work.bpe[index]; right = work.bpe[index + 1]; break;
            }
          }
        }
        if (best == kMergeCount) break;
        int output = 0;
        for (int index = 0; index < length;) {
          if (index + 1 < length && work.bpe[index] == left && work.bpe[index + 1] == right) {
            work.bpe[output++] = uint16_t(260 + best); index += 2;
          } else work.bpe[output++] = work.bpe[index++];
        }
        length = output;
      }
      if (count + length > kSource) return -1;
      for (int index = 0; index < length; ++index) work.source[count++] = work.bpe[index];
      have_word = true;
    }
    return count;
  }

 private:
  WeightCache& cache_;
  static bool whitespace(uint8_t b) { return b == 32 || (b >= 9 && b <= 13); }
  static float sigmoid(float x) { if (x < -80) x = -80; if (x > 80) x = 80; return 1.0f / (1.0f + expf(-x)); }
  static int argmax(const float* values, int count) { int result = 0; for (int i = 1; i < count; ++i) if (values[i] > values[result]) result = i; return result; }
  static void softmax(float* values, int count) {
    const float maximum = values[argmax(values, count)]; float total = 0;
    for (int i = 0; i < count; ++i) { values[i] = expf(values[i] - maximum); total += values[i]; }
    for (int i = 0; i < count; ++i) values[i] /= total;
  }

  bool row(const Matrix& matrix, int index, const float* input, float& output, float* embedding = nullptr) {
    if (index < 0 || index >= matrix.rows) return false;
    uint32_t offset = kDataOffset + matrix.offset + uint32_t(index) * matrix.stride;
    float scale_bias[2];
    if (!cache_.read(offset, scale_bias, sizeof(scale_bias))) return false;
    offset += sizeof(scale_bias);
    float sum = 0; int used = 0;
    while (used < matrix.cols) {
      const uint8_t* bytes; size_t available;
      if (!cache_.window(offset, bytes, available)) return false;
      int chunk = matrix.cols - used;
      if (size_t(chunk) > available) chunk = int(available);
      for (int j = 0; j < chunk; ++j) {
        const float weight = float(static_cast<int8_t>(bytes[j]));
        if (embedding) embedding[used + j] = weight * scale_bias[0];
        else sum += weight * input[used + j];
      }
      offset += chunk; used += chunk;
    }
    output = sum * scale_bias[0] + scale_bias[1];
    return isfinite(output);
  }
  bool matvec(const Matrix& matrix, const float* input, float* output) {
    for (int index = 0; index < matrix.rows; ++index) if (!row(matrix, index, input, output[index])) return false;
    return true;
  }
  bool embedding(int token) { float unused; return row(m_embedding_weight, token, nullptr, unused, work.embedded); }
  bool gru(const float* input, float* hidden, int size, const Matrix& ih, const Matrix& hh) {
    if (!matvec(ih, input, work.gi) || !matvec(hh, hidden, work.gh)) return false;
    for (int i = 0; i < size; ++i) {
      const float reset = sigmoid(work.gi[i] + work.gh[i]);
      const float update = sigmoid(work.gi[size + i] + work.gh[size + i]);
      const float candidate = tanhf(work.gi[2 * size + i] + reset * work.gh[2 * size + i]);
      hidden[i] = (1 - update) * candidate + update * hidden[i];
    }
    return true;
  }
  bool encode(int count) {
    for (int reverse = 0; reverse < 2; ++reverse) {
      memset(work.encoder_hidden, 0, sizeof(work.encoder_hidden));
      for (int step = 0; step < count; ++step) {
        const int index = reverse ? count - 1 - step : step;
        if (!embedding(work.source[index])) return false;
        if (!gru(work.embedded, work.encoder_hidden, kEncoder,
                 reverse ? m_encoder_weight_ih_l0_reverse : m_encoder_weight_ih_l0,
                 reverse ? m_encoder_weight_hh_l0_reverse : m_encoder_weight_hh_l0)) return false;
        memcpy(work.encoded[index] + reverse * kEncoder, work.encoder_hidden, sizeof(work.encoder_hidden));
      }
    }
    memcpy(work.final_state, work.encoded[count - 1], sizeof(float) * kEncoder);
    memcpy(work.final_state + kEncoder, work.encoded[0] + kEncoder, sizeof(float) * kEncoder);
    if (!matvec(m_initial_weight, work.final_state, work.initial)) return false;
    for (int layer = 0; layer < kLayers; ++layer) for (int i = 0; i < kDecoder; ++i) work.hidden[layer][i] = tanhf(work.initial[layer * kDecoder + i]);
    return true;
  }
  bool step(int previous, int count) {
    if (!embedding(previous)) return false;
    memcpy(work.input, work.embedded, sizeof(work.embedded));
    memcpy(work.input + kEmbedding, work.context, sizeof(work.context));
    if (!gru(work.input, work.hidden[0], kDecoder, m_decoder_weight_ih_l0, m_decoder_weight_hh_l0)) return false;
    if (kLayers == 2 && !gru(work.hidden[0], work.hidden[1], kDecoder, m_decoder_weight_ih_l1, m_decoder_weight_hh_l1)) return false;
    const float* state = work.hidden[kLayers - 1];
    if (!matvec(m_query_weight, state, work.query)) return false;
    for (int index = 0; index < count; ++index) {
      float dot = 0; for (int i = 0; i < kEncSize; ++i) dot += work.encoded[index][i] * work.query[i];
      work.attention[index] = dot / sqrtf(float(kEncSize));
    }
    softmax(work.attention, count);
    memset(work.context, 0, sizeof(work.context));
    for (int index = 0; index < count; ++index) for (int i = 0; i < kEncSize; ++i) work.context[i] += work.attention[index] * work.encoded[index][i];
    memcpy(work.input, state, sizeof(float) * kDecoder);
    memcpy(work.input + kDecoder, work.context, sizeof(work.context));
    if (!matvec(m_output_weight, work.input, work.generated)) return false;
    softmax(work.generated, kVocab);
    memcpy(work.input + kDecoder + kEncSize, work.embedded, sizeof(work.embedded));
    float gate;
    if (!matvec(m_generation_gate_weight, work.input, &gate)) return false;
    gate = sigmoid(gate);
    for (int i = 0; i < kVocab; ++i) work.probabilities[i] = gate * work.generated[i];
    for (int index = 0; index < count; ++index) work.probabilities[work.source[index]] += (1 - gate) * work.attention[index];
    return true;
  }
  bool append_token(int token, size_t& used, int depth) {
    if (token < 0 || token >= kVocab || depth > 64) return false;
    if (token < 256) {
      if (used >= sizeof(work.summary) - 1 || token == 0) return false;
      work.summary[used++] = char(token); work.summary[used] = 0; return true;
    }
    if (token < 260) return false;
    const int rank = token - 260;
    return rank < kMergeCount && append_token(kMerges[rank][0], used, depth + 1) && append_token(kMerges[rank][1], used, depth + 1);
  }
  static bool valid_utf8(const uint8_t* text) {
    while (*text) {
      const uint8_t first = *text++;
      if (first < 128) continue;
      int remaining; uint32_t value, minimum;
      if (first >= 0xC2 && first <= 0xDF) { remaining = 1; value = first & 31; minimum = 0x80; }
      else if (first >= 0xE0 && first <= 0xEF) { remaining = 2; value = first & 15; minimum = 0x800; }
      else if (first >= 0xF0 && first <= 0xF4) { remaining = 3; value = first & 7; minimum = 0x10000; }
      else return false;
      for (int i = 0; i < remaining; ++i) { if ((*text & 0xC0) != 0x80) return false; value = (value << 6) | (*text++ & 63); }
      if (value < minimum || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) return false;
    }
    return true;
  }
};

static_assert(sizeof(Runtime) + sizeof(WeightCache) <= kWorkspaceLimit, "Model/cache exceeds the 256 KiB working arena");
}  // namespace sd_student
