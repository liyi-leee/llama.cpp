#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

// Helper function to fill a tensor with random data
static void fill_random_f32(struct ggml_tensor *tensor) {
    if (tensor == NULL) return;
    size_t n = ggml_nelements(tensor);
    float *data = (float *) tensor->data;
    for (size_t i = 0; i < n; i++) {
        data[i] = ((float) rand() / RAND_MAX - 0.5f) * 0.2f;
    }
}

int main() {
    // --- Initialize GGUF Context ---
    struct ggml_init_params params = { 128 * 1024 * 1024, NULL, false };
    struct ggml_context *ctx_data = ggml_init(params);
    struct gguf_context *ctx_gguf = gguf_init_empty();

    // --- Set Metadata ---
    gguf_set_val_str(ctx_gguf, "general.architecture", "llama");
    gguf_set_val_str(ctx_gguf, "general.name", "test-gguf");

    // --- Add Required Hyperparameters ---
    gguf_set_val_u32(ctx_gguf, "llama.vocab_size", 3);                // 词汇表大小
    gguf_set_val_u32(ctx_gguf, "llama.block_count", 1);              // Transformer block 数量
    gguf_set_val_u32(ctx_gguf, "llama.context_length", 512);         // 上下文长度
    gguf_set_val_u32(ctx_gguf, "llama.embedding_length", 128);       // 嵌入维度
    gguf_set_val_u32(ctx_gguf, "llama.feed_forward_length", 512);    // 前馈网络维度
    gguf_set_val_u32(ctx_gguf, "llama.attention.head_count", 8);     // 注意力头数量
    gguf_set_val_u32(ctx_gguf, "llama.attention.head_count_kv", 1);  // KV 注意力头数量
    gguf_set_val_f32(ctx_gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5f); // LayerNorm RMS Epsilon

    // --- Define Tokenizer ---
    const char *tokens[] = { "<unk>", "<s>", "</s>" };
    float scores[] = { 0.0f, 0.0f, 0.0f };
    int32_t toktypes[] = { 2, 3, 3 };

    gguf_set_val_str(ctx_gguf, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(ctx_gguf, "tokenizer.ggml.tokens", tokens, 3);
    gguf_set_arr_data(ctx_gguf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, 3);
    gguf_set_arr_data(ctx_gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, toktypes, 3);

    // --- Add token_embd.weight Tensor ---
    int vocab_size = 3; // 词汇表大小
    int embedding_dim = 128; // 嵌入维度

    // --- Helper for adding tensors ---
    auto add_tensor = [&](const char *name, struct ggml_tensor *tensor) {
        fill_random_f32(tensor);
        ggml_set_name(tensor, name);
        gguf_add_tensor(ctx_gguf, tensor);
    };

    printf("Defining model tensors...\n");
    add_tensor("token_embd.weight", ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim, vocab_size));
    add_tensor("output.weight", ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim, vocab_size));
    add_tensor("output_norm.weight", ggml_new_tensor_1d(ctx_data, GGML_TYPE_F32, embedding_dim));
    add_tensor("blk.0.attn_norm.weight", ggml_new_tensor_1d(ctx_data, GGML_TYPE_F32, embedding_dim));
    add_tensor("blk.0.attn_q.weight",    ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim, embedding_dim));
    add_tensor("blk.0.attn_k.weight",    ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim, embedding_dim/8));
    add_tensor("blk.0.attn_v.weight",    ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim, embedding_dim/8));
    add_tensor("blk.0.attn_output.weight",    ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim, embedding_dim));
    add_tensor("blk.0.ffn_norm.weight", ggml_new_tensor_1d(ctx_data, GGML_TYPE_F32, embedding_dim));
    add_tensor("blk.0.ffn_gate.weight", ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim, embedding_dim*4));
    add_tensor("blk.0.ffn_down.weight", ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim*4, embedding_dim));
    add_tensor("blk.0.ffn_up.weight",   ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, embedding_dim, embedding_dim*4));

    // --- Write GGUF File ---
    const char* filename = "test-gguf.gguf";
    printf("Writing to file '%s'...\n", filename);
    gguf_write_to_file(ctx_gguf, filename, false);

    printf("✅ GGUF test model written successfully.\n");
    ggml_free(ctx_data);
    gguf_free(ctx_gguf);

    return 0;
}