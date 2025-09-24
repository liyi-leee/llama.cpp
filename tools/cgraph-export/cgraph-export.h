#ifndef CGRAPH_EXPORT_H
#define CGRAPH_EXPORT_H

#include <memory>
#include <cstdint>

// 修改后的序列化函数，返回FlatBuffer元数据和独立的数据buffer
struct SerializationResult {
    std::unique_ptr<uint8_t[]> metadata_buffer;
    size_t metadata_size;
    std::unique_ptr<uint8_t[]> data_buffer;
    size_t data_size;
};

SerializationResult serialize_cgraph_to_flatbuffer(const struct ggml_cgraph * cgraph);
void convert_to_flat_cgraph(const struct ggml_cgraph * cgraph);
void test_cgraph_serialization(const struct ggml_cgraph * original_cgraph);

#endif // CGRAPH_EXPORT_H