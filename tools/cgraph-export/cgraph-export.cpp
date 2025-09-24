#include "ggml-impl.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <inttypes.h>

#include "cgraph_generated.h"
#include "cgraph-export.h"

// Helper function to convert ggml_type to TensorType
GGML::Serialization::TensorType convert_ggml_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return GGML::Serialization::TensorType_F32;
        case GGML_TYPE_F16: return GGML::Serialization::TensorType_F16;
        case GGML_TYPE_Q4_0: return GGML::Serialization::TensorType_Q4_0;
        case GGML_TYPE_Q4_1: return GGML::Serialization::TensorType_Q4_1;
        case GGML_TYPE_Q8_0: return GGML::Serialization::TensorType_Q8_0;
        case GGML_TYPE_I8: return GGML::Serialization::TensorType_I8;
        case GGML_TYPE_I16: return GGML::Serialization::TensorType_I16;
        case GGML_TYPE_I32: return GGML::Serialization::TensorType_I32;
        default: return GGML::Serialization::TensorType_F32;
    }
}

// Helper function to convert ggml_op to TensorOp
GGML::Serialization::TensorOp convert_ggml_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_ADD: return GGML::Serialization::TensorOp_ADD;
        case GGML_OP_SUB: return GGML::Serialization::TensorOp_SUB;
        case GGML_OP_MUL: return GGML::Serialization::TensorOp_MUL;
        case GGML_OP_DIV: return GGML::Serialization::TensorOp_DIV;
        case GGML_OP_MUL_MAT: return GGML::Serialization::TensorOp_MUL_MAT;
        case GGML_OP_ROPE: return GGML::Serialization::TensorOp_ROPE;
        case GGML_OP_NORM: return GGML::Serialization::TensorOp_NORM;
        case GGML_OP_SOFT_MAX: return GGML::Serialization::TensorOp_SOFT_MAX;
        case GGML_OP_RESHAPE: return GGML::Serialization::TensorOp_RESHAPE;
        case GGML_OP_VIEW: return GGML::Serialization::TensorOp_VIEW;
        case GGML_OP_PERMUTE: return GGML::Serialization::TensorOp_PERMUTE;
        case GGML_OP_TRANSPOSE: return GGML::Serialization::TensorOp_TRANSPOSE;
        case GGML_OP_CONV_2D: return GGML::Serialization::TensorOp_CONV_2D;
        case GGML_OP_POOL_1D: return GGML::Serialization::TensorOp_POOL_1D;
        case GGML_OP_POOL_2D: return GGML::Serialization::TensorOp_POOL_2D;
        // For unary operations, we'll use specific cases when available
        case GGML_OP_UNARY: return GGML::Serialization::TensorOp_CUSTOM; // Handle unary ops specially
        default: return GGML::Serialization::TensorOp_CUSTOM;
    }
}

SerializationResult serialize_cgraph_to_flatbuffer(const struct ggml_cgraph * cgraph) {

    printf("n_nodes = %d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        printf(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                ggml_op_name(node->op), (node->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, node) ? "g" : " ");
    }

    printf("n_leafs = %d\n", cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * node = cgraph->leafs[i];

        printf(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                node->ne[0], node->ne[1],
                ggml_op_name(node->op),
                ggml_get_name(node));
    }

    flatbuffers::FlatBufferBuilder builder; // 使用默认大小，让它自动扩展

    // 创建一个映射来跟踪张量索引
    std::unordered_map<struct ggml_tensor*, uint32_t> tensor_index_map;
    
    // 收集所有唯一的张量 (nodes + leafs)
    std::vector<struct ggml_tensor*> all_tensors;
    
    // 添加所有计算节点
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * tensor = cgraph->nodes[i];
        if (tensor_index_map.find(tensor) == tensor_index_map.end()) {
            tensor_index_map[tensor] = all_tensors.size();
            all_tensors.push_back(tensor);
        }
    }

    // 添加所有叶子节点
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * tensor = cgraph->leafs[i];
        if (tensor_index_map.find(tensor) == tensor_index_map.end()) {
            tensor_index_map[tensor] = all_tensors.size();
            all_tensors.push_back(tensor);
        }
    }
    

    // 收集所有张量数据到独立的连续缓冲区（不受FlatBuffer限制）
    std::vector<uint8_t> continuous_data_buffer;
    std::vector<uint64_t> data_offsets; // 记录每个张量在缓冲区中的偏移
    
    size_t total_data_size = 0;
    
    for (size_t i = 0; i < all_tensors.size(); i++) {
        struct ggml_tensor * tensor = all_tensors[i];
        size_t data_size = ggml_nbytes(tensor);
        
        // 记录当前张量的数据偏移
        data_offsets.push_back(continuous_data_buffer.size());
        
        // 添加调试信息
        if (i < 5) {
            printf("Tensor %zu: data=%p, size=%zu, offset=%zu, name='%s'\n", 
                   i, tensor->data, data_size, continuous_data_buffer.size(), 
                   ggml_get_name(tensor) ? ggml_get_name(tensor) : "unnamed");
        }
        
        // 存储所有张量的数据到独立的buffer中
        if (tensor->data && data_size > 0) {
            const uint8_t* tensor_data = static_cast<const uint8_t*>(tensor->data);
            continuous_data_buffer.insert(continuous_data_buffer.end(), 
                                         tensor_data, tensor_data + data_size);
            total_data_size += data_size;
        } else {
            // 如果没有数据，添加占位符（零填充）
            continuous_data_buffer.resize(continuous_data_buffer.size() + data_size, 0);
            if (i < 5) {
                printf("  -> No data, zero-filled\n");
            }
        }
    }
    
    printf("Total data buffer size: %zu bytes\n", continuous_data_buffer.size());
    
    // 不再将数据存储在FlatBuffer中，FlatBuffer只存储元数据
    // 创建空的数据缓冲区向量（用于兼容性）
    std::vector<uint8_t> empty_data_buffer;
    auto data_buffer = builder.CreateVector(empty_data_buffer);
    
    // 创建序列化的张量
    std::vector<flatbuffers::Offset<GGML::Serialization::SerializedTensor>> tensor_offsets;
    
    for (size_t i = 0; i < all_tensors.size(); i++) {
        struct ggml_tensor * tensor = all_tensors[i];
        
        // 创建维度信息
        GGML::Serialization::TensorDimensions dims(
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]
        );
        
        // 创建步长信息
        GGML::Serialization::TensorStrides strides(
            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]
        );
        
        // 创建操作参数（如果有的话）
        std::vector<int32_t> op_params_vec;
        // TODO: 根据具体操作类型填充参数
        auto op_params = builder.CreateVector(op_params_vec);
        
        // 创建源张量索引
        std::vector<uint32_t> src_indices;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (tensor->src[j] != nullptr) {
                auto it = tensor_index_map.find(tensor->src[j]);
                if (it != tensor_index_map.end()) {
                    src_indices.push_back(it->second);
                }
            }
        }
        auto src = builder.CreateVector(src_indices);
        
        // 创建名称
        const char* tensor_name = ggml_get_name(tensor);
        auto name = builder.CreateString(tensor_name ? tensor_name : "");
        
        // 计算数据大小
        size_t data_size = ggml_nbytes(tensor);
        
        // 创建序列化张量 - 数据偏移指向独立buffer中的位置
        auto serialized_tensor = CreateSerializedTensor(
            builder,
            convert_ggml_type(tensor->type),    // type
            &dims,                              // TensorDimensions
            &strides,                           // TensorStrides  
            convert_ggml_op(tensor->op),        // op
            op_params,                          // op_params
            tensor->flags,                      // flags
            src,                                // src
            0,                                  // view_src (TODO: handle view tensors)
            0,                                  // view_offs
            data_offsets[i],                    // data_offset (指向独立buffer)
            data_size,                          // data_size
            name                                // name
        );
        
        tensor_offsets.push_back(serialized_tensor);
    }
    
    auto tensors = builder.CreateVector(tensor_offsets);
    
    // 创建节点索引数组
    std::vector<uint32_t> node_indices_vec;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto it = tensor_index_map.find(cgraph->nodes[i]);
        if (it != tensor_index_map.end()) {
            node_indices_vec.push_back(it->second);
        }
    }
    auto node_indices = builder.CreateVector(node_indices_vec);
    
    // 创建叶子索引数组
    std::vector<uint32_t> leaf_indices_vec;
    for (int i = 0; i < cgraph->n_leafs; i++) {
        auto it = tensor_index_map.find(cgraph->leafs[i]);
        if (it != tensor_index_map.end()) {
            leaf_indices_vec.push_back(it->second);
        }
    }
    auto leaf_indices = builder.CreateVector(leaf_indices_vec);
    
    // 创建梯度相关的索引数组（目前为空）
    auto grad_indices = builder.CreateVector<uint32_t>({});
    auto grad_acc_indices = builder.CreateVector<uint32_t>({});
    
    // 创建使用计数数组（目前为空）
    auto use_counts = builder.CreateVector<int32_t>({});

    // 创建HashSet（目前为空）
    auto visited_hash_set = GGML::Serialization::CreateHashSetDirect(
        builder,
        0,       // size
        nullptr, // used_bits
        nullptr  // keys
    );

    // 创建最终的序列化计算图
    auto serialized_cgraph = GGML::Serialization::CreateSerializedCGraph(
        builder,
        /* size */ (uint32_t)all_tensors.size(),
        /* n_nodes */ cgraph->n_nodes,
        /* n_leafs */ cgraph->n_leafs,
        tensors,
        node_indices,
        grad_indices,
        grad_acc_indices,
        leaf_indices,
        use_counts,
        visited_hash_set,
        GGML::Serialization::GraphEvalOrder_LEFT_TO_RIGHT,
        data_buffer
    );

    builder.Finish(serialized_cgraph);

    // 获取序列化数据
    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    printf("Serialized cgraph metadata: %p, size: %ld bytes\n", buf, size);
    printf("Total tensors: %zu, nodes: %d, leafs: %d\n", 
           all_tensors.size(), cgraph->n_nodes, cgraph->n_leafs);
    printf("Separate data buffer size: %zu bytes\n", continuous_data_buffer.size());
    
    // 复制FlatBuffer元数据到独立的内存块
    auto metadata_copy = std::make_unique<uint8_t[]>(size);
    memcpy(metadata_copy.get(), buf, size);
    
    // 复制数据buffer到独立的内存块
    auto data_copy = std::make_unique<uint8_t[]>(continuous_data_buffer.size());
    memcpy(data_copy.get(), continuous_data_buffer.data(), continuous_data_buffer.size());
    
    SerializationResult result;
    result.metadata_buffer = std::move(metadata_copy);
    result.metadata_size = size;
    result.data_buffer = std::move(data_copy);
    result.data_size = continuous_data_buffer.size();
    
    return result;
}

void convert_to_flat_cgraph(const struct ggml_cgraph * cgraph) {
    auto result = serialize_cgraph_to_flatbuffer(cgraph);
    // 写入元数据到文件
    std::ofstream meta_out("cgraph_metadata.bin", std::ios::binary);
    if (meta_out.is_open()) {
        meta_out.write(reinterpret_cast<const char*>(result.metadata_buffer.get()), result.metadata_size);
        meta_out.close();
        printf("Metadata written to cgraph_metadata.bin (%zu bytes)\n", result.metadata_size);
    } else {
        printf("Failed to open cgraph_metadata.bin for writing!\n");
    }

    // 写入数据到文件
    std::ofstream data_out("cgraph_data.bin", std::ios::binary);
    if (data_out.is_open()) {
        data_out.write(reinterpret_cast<const char*>(result.data_buffer.get()), result.data_size);
        data_out.close();
        printf("Data written to cgraph_data.bin (%zu bytes)\n", result.data_size);
    } else {
        printf("Failed to open cgraph_data.bin for writing!\n");
    }

    printf("Serialization completed:\n");
    printf("  Metadata size: %zu bytes\n", result.metadata_size);
    printf("  Data size: %zu bytes\n", result.data_size);
}

// 测试函数：序列化然后直接验证 FlatBuffer 数据
void test_cgraph_serialization(const struct ggml_cgraph * original_cgraph) {
    printf("\n=== Testing CGraph Serialization ===\n");
    
    // 序列化
    printf("1. Serializing original cgraph...\n");
    auto result = serialize_cgraph_to_flatbuffer(original_cgraph);
    
    // 验证 FlatBuffer 数据
    printf("\n2. Verifying FlatBuffer metadata...\n");
    
    // 验证 FlatBuffer 数据的有效性
    auto verifier = flatbuffers::Verifier(result.metadata_buffer.get(), result.metadata_size);
    if (!GGML::Serialization::VerifySerializedCGraphBuffer(verifier)) {
        printf("✗ FlatBuffer metadata verification failed!\n");
        return;
    }
    printf("✓ FlatBuffer metadata is valid\n");
    
    // 获取序列化的计算图
    auto serialized_cgraph = GGML::Serialization::GetSerializedCGraph(result.metadata_buffer.get());
    if (!serialized_cgraph) {
        printf("✗ Failed to read SerializedCGraph from buffer!\n");
        return;
    }
    
    auto tensors = serialized_cgraph->tensors();
    if (!tensors) {
        printf("✗ No tensors found in serialized data!\n");
        return;
    }
    
    printf("✓ Successfully read serialized data\n");
    
    // 比较基本结构信息
    printf("\n3. Comparing structure information...\n");
    printf("Original: %d nodes, %d leafs\n", original_cgraph->n_nodes, original_cgraph->n_leafs);
    printf("Serialized: %d nodes, %d leafs\n", serialized_cgraph->n_nodes(), serialized_cgraph->n_leafs());
    
    bool structure_matches = (original_cgraph->n_nodes == serialized_cgraph->n_nodes()) &&
                            (original_cgraph->n_leafs == serialized_cgraph->n_leafs());
    
    if (!structure_matches) {
        printf("✗ Graph structure doesn't match!\n");
        return;
    }
    printf("✓ Graph structure matches\n");
    
    // 创建原始张量到索引的映射（与序列化时相同的逻辑）
    std::unordered_map<struct ggml_tensor*, uint32_t> tensor_index_map;
    std::vector<struct ggml_tensor*> all_tensors;
    
    // 添加所有计算节点
    for (int i = 0; i < original_cgraph->n_nodes; i++) {
        struct ggml_tensor * tensor = original_cgraph->nodes[i];
        if (tensor_index_map.find(tensor) == tensor_index_map.end()) {
            tensor_index_map[tensor] = all_tensors.size();
            all_tensors.push_back(tensor);
        }
    }

    // 添加所有叶子节点
    for (int i = 0; i < original_cgraph->n_leafs; i++) {
        struct ggml_tensor * tensor = original_cgraph->leafs[i];
        if (tensor_index_map.find(tensor) == tensor_index_map.end()) {
            tensor_index_map[tensor] = all_tensors.size();
            all_tensors.push_back(tensor);
        }
    }
    
    printf("\n4. Comparing tensor data...\n");
    printf("Total tensors to compare: %zu\n", all_tensors.size());
    
    if (tensors->size() != all_tensors.size()) {
        printf("✗ Tensor count mismatch: original=%zu, serialized=%u\n", 
               all_tensors.size(), tensors->size());
        return;
    }
    
    int mismatches = 0;
    int perfect_matches = 0;
    
    // 逐个比较张量
    for (size_t i = 0; i < all_tensors.size(); i++) {
        struct ggml_tensor * orig_tensor = all_tensors[i];
        auto serialized_tensor = tensors->Get(i);
        
        bool tensor_matches = true;
        
        // 比较维度
        auto dims = serialized_tensor->ne();
        if (orig_tensor->ne[0] != dims->ne0() || 
            orig_tensor->ne[1] != dims->ne1() ||
            orig_tensor->ne[2] != dims->ne2() || 
            orig_tensor->ne[3] != dims->ne3()) {
            tensor_matches = false;
            printf("  Tensor %zu: dimension mismatch\n", i);
            printf("    Original: [%ld,%ld,%ld,%ld]\n", 
                   orig_tensor->ne[0], orig_tensor->ne[1], orig_tensor->ne[2], orig_tensor->ne[3]);
            printf("    Serialized: [%ld,%ld,%ld,%ld]\n", 
                   dims->ne0(), dims->ne1(), dims->ne2(), dims->ne3());
        }
        
        // 比较步长
        auto strides = serialized_tensor->nb();
        if (orig_tensor->nb[0] != strides->nb0() || 
            orig_tensor->nb[1] != strides->nb1() ||
            orig_tensor->nb[2] != strides->nb2() || 
            orig_tensor->nb[3] != strides->nb3()) {
            tensor_matches = false;
            printf("  Tensor %zu: stride mismatch\n", i);
        }
        
        // 比较类型
        if (convert_ggml_type(orig_tensor->type) != serialized_tensor->type()) {
            tensor_matches = false;
            printf("  Tensor %zu: type mismatch\n", i);
        }
        
        // 比较操作
        if (convert_ggml_op(orig_tensor->op) != serialized_tensor->op()) {
            tensor_matches = false;
            printf("  Tensor %zu: operation mismatch (orig=%s, serialized=%d)\n", 
                   i, ggml_op_name(orig_tensor->op), serialized_tensor->op());
        }
        
        // 比较标志
        if (orig_tensor->flags != serialized_tensor->flags()) {
            tensor_matches = false;
            printf("  Tensor %zu: flags mismatch\n", i);
        }
        
        // 比较名称
        const char* orig_name = ggml_get_name(orig_tensor);
        const char* serialized_name = serialized_tensor->name() ? serialized_tensor->name()->c_str() : "";
        if (strcmp(orig_name ? orig_name : "", serialized_name) != 0) {
            tensor_matches = false;
            printf("  Tensor %zu: name mismatch (orig='%s', serialized='%s')\n", 
                   i, orig_name ? orig_name : "", serialized_name);
        }
        
        // 比较源张量索引
        auto src_indices = serialized_tensor->src();
        std::vector<uint32_t> expected_src_indices;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (orig_tensor->src[j] != nullptr) {
                auto it = tensor_index_map.find(orig_tensor->src[j]);
                if (it != tensor_index_map.end()) {
                    expected_src_indices.push_back(it->second);
                }
            }
        }
        
        if (src_indices->size() != expected_src_indices.size()) {
            tensor_matches = false;
            printf("  Tensor %zu: source count mismatch\n", i);
        } else {
            for (size_t j = 0; j < expected_src_indices.size(); j++) {
                if (src_indices->Get(j) != expected_src_indices[j]) {
                    tensor_matches = false;
                    printf("  Tensor %zu: source index mismatch at position %zu\n", i, j);
                    break;
                }
            }
        }
        
        // 比较数据大小
        size_t orig_data_size = ggml_nbytes(orig_tensor);
        
        if (orig_data_size != serialized_tensor->data_size()) {
            tensor_matches = false;
            printf("  Tensor %zu: data size mismatch (orig=%zu, serialized=%lu)\n", 
                   i, orig_data_size, serialized_tensor->data_size());
        }
        
        // 比较实际数据内容 - 使用独立的数据buffer
        if (orig_data_size > 0 && serialized_tensor->data_size() > 0) {
            if (serialized_tensor->data_offset() + serialized_tensor->data_size() <= result.data_size) {
                const uint8_t* serialized_data = result.data_buffer.get() + serialized_tensor->data_offset();
                const uint8_t* orig_data = static_cast<const uint8_t*>(orig_tensor->data);
                
                if (orig_data && orig_tensor->data) {
                    // 比较前64字节或全部数据（取较小值）
                    size_t compare_size = std::min(orig_data_size, (size_t)64);
                    bool data_matches = memcmp(orig_data, serialized_data, compare_size) == 0;
                    
                    if (!data_matches) {
                        tensor_matches = false;
                        printf("  Tensor %zu: data content mismatch (compared first %zu bytes)\n", i, compare_size);
                        
                        // 显示前几个字节的详细对比（仅对小张量）
                        if (orig_data_size <= 32 && i < 5) {
                            printf("    Original:   ");
                            for (size_t k = 0; k < std::min(compare_size, (size_t)16); k++) {
                                printf("%02x ", orig_data[k]);
                            }
                            printf("\n    Serialized: ");
                            for (size_t k = 0; k < std::min(compare_size, (size_t)16); k++) {
                                printf("%02x ", serialized_data[k]);
                            }
                            printf("\n");
                        }
                    } else if (i < 5) {
                        printf("  Tensor %zu: data content matches (checked %zu bytes)\n", i, compare_size);
                    }
                } else {
                    if (i < 5) {
                        printf("  Tensor %zu: original data is null, skipping data comparison\n", i);
                    }
                }
            } else {
                tensor_matches = false;
                printf("  Tensor %zu: invalid data buffer offset or size\n", i);
                if (i < 5) {
                    printf("    data_buffer=%p, offset=%llu, size=%llu, buffer_size=%zu\n",
                           result.data_buffer.get(), serialized_tensor->data_offset(), 
                           serialized_tensor->data_size(), result.data_size);
                }
            }
        } else if (i < 5) {
            printf("  Tensor %zu: no data to compare (orig_size=%zu, ser_size=%llu)\n", 
                   i, orig_data_size, serialized_tensor->data_size());
        }
        
        if (tensor_matches) {
            perfect_matches++;
        } else {
            mismatches++;
        }
        
        // 显示前几个张量的详细信息
        if (i < 5) {
            printf("  Tensor %zu (%s): %s\n", i, 
                   orig_name ? orig_name : "unnamed",
                   tensor_matches ? "✓ MATCH" : "✗ MISMATCH");
            printf("    Original: data=%p, size=%zu\n", orig_tensor->data, orig_data_size);
            printf("    Serialized: offset=%llu, size=%llu\n", 
                   serialized_tensor->data_offset(), serialized_tensor->data_size());
            if (!tensor_matches) {
                printf("    Mismatch details logged above\n");
            }
        } else if (i < 3) {
            printf("  Tensor %zu (%s): %s\n", i, 
                   orig_name ? orig_name : "unnamed",
                   tensor_matches ? "✓ MATCH" : "✗ MISMATCH");
        }
    }
    
    printf("\n5. Final comparison results:\n");
    printf("  Perfect matches: %d/%zu\n", perfect_matches, all_tensors.size());
    printf("  Mismatches: %d/%zu\n", mismatches, all_tensors.size());
    
    if (mismatches == 0) {
        printf("✓ All tensors match perfectly!\n");
        printf("✓ Serialization/deserialization test PASSED!\n");
    } else {
        printf("✗ Found %d tensor mismatches\n", mismatches);
        printf("✗ Serialization test FAILED!\n");
    }
    
    printf("=== Serialization test completed ===\n\n");
}