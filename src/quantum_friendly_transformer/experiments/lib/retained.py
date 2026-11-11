"""Manifests for result files retained in the current report scope."""

from __future__ import annotations


RETAINED_N_SCALING_INPUTS = [
    "experiments/outputs/raw/diag_llama32_1b_wikitext_N512_1024_2048_4096_8192_zonly.parquet",
    "experiments/outputs/raw/diag_mistral7b_wikitext_N512_1024_2048_4096_8192_zonly.parquet",
    "experiments/outputs/raw/diag_mistral7b_instruct_v02_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_qwen15_1p8b_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_qwen25_1p5b_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_qwen3_1p7b_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_phi4_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_gemma3_4b_pt_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_qwen25_0p5b_instruct_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_qwen3_0p6b_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_gemma3_1b_pt_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_deepseek_r1_distill_qwen_1p5b_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_phi3_mini_128k_instruct_wikitext_N512_1024_2048_4096_8192.parquet",
    "experiments/outputs/raw/diag_smollm3_3b_wikitext_N512_1024_2048_4096_8192.parquet",
]



RETAINED_RECOVERY_RUNS = [
    {
        "model": "roneneldan/TinyStories-1M",
        "family": "TinyStories",
        "hidden_size_d": 64,
        "protocol": "full_model_learnable_cap_strict_sst2",
        "source": "recovery_sst2_tinystories_1m_fn24_lr3e4_steps20000_fulltrain.json",
        "baseline_source": "baseline_sst2_tinystories_1m_strict_steps10000_fulltrain.json",
    },
    {
        "model": "roneneldan/TinyStories-3M",
        "family": "TinyStories",
        "hidden_size_d": 128,
        "protocol": "full_model_learnable_cap_strict_sst2",
        "source": "recovery_sst2_tinystories_3m_fn24_low_lr1e4_steps20000_fulltrain.json",
        "baseline_source": "baseline_sst2_tinystories_3m_strict_steps5000_fulltrain.json",
    },
    {
        "model": "roneneldan/TinyStories-8M",
        "family": "TinyStories",
        "hidden_size_d": 256,
        "protocol": "full_model_learnable_cap_strict_sst2",
        "source": "recovery_sst2_tinystories_8m_fn24_stricttarget833716_steps10000_fulltrain.json",
        "baseline_source": "baseline_sst2_tinystories_8m_strict_steps5000_fulltrain.json",
    },
    {
        "model": "roneneldan/TinyStories-28M",
        "family": "TinyStories",
        "hidden_size_d": 512,
        "protocol": "full_model_learnable_cap_strict_sst2",
        "source": "recovery_sst2_tinystories_28m_fn24_steps600_fulltrain.json",
        "baseline_source": "baseline_sst2_tinystories_28m_strict_steps5000_fulltrain.json",
        "baseline_target": 0.8256880733944955,
        "threshold": 0.8156880733944954,
    },
    {
        "model": "roneneldan/TinyStories-33M",
        "family": "TinyStories",
        "hidden_size_d": 768,
        "protocol": "full_model_learnable_cap_strict_sst2",
        "source": "recovery_sst2_tinystories_33m_fn24_steps3000_fulltrain_target847477.json",
        "baseline_source": "baseline_sst2_tinystories_33m_strict_steps5000_fulltrain.json",
        "baseline_target": 0.8451834862385321,
        "threshold": 0.8351834862385321,
    },
    {
        "model": "roneneldan/TinyStories-1Layer-21M",
        "family": "TinyStories",
        "hidden_size_d": 1024,
        "protocol": "full_model_learnable_cap_strict_sst2",
        "source": "recovery_sst2_tinystories_1layer_21m_d1024_fn24_steps3000_fulltrain.json",
    },
    {
        "model": "roneneldan/TinyStories-2Layers-33M",
        "family": "TinyStories",
        "hidden_size_d": 1024,
        "protocol": "full_model_learnable_cap_strict_sst2",
        "source": "recovery_sst2_tinystories_2layers_33m_d1024_fn24_steps5000_fulltrain_target857798.json",
    },
    {
        "model": "allenai/OLMo-1B-0724-hf",
        "family": "OLMo",
        "hidden_size_d": 2048,
        "protocol": "full_model_learnable_cap_sst2",
        "source": "recovery_sst2_olmo_1b_0724_fullfinetune_cap24_bsz32_lr2e5_3epochs_target933486.json",
        "baseline_source": "baseline_sst2_olmo_1b_0724_fullfinetune_bsz32_lr2e5_3epochs.json",
    },
    {
        "model": "mistralai/Mistral-7B-v0.1",
        "family": "Mistral",
        "hidden_size_d": 4096,
        "protocol": "head_caps_learnable_cap_sst2",
        "source": "dscale_sst2_mistral7b_N128_caps24_48_96_steps500_headcaps.json",
    },
]


RETAINED_TRAINABLE_ATTENTION_RECOVERY_RUNS = [
    {
        "model": "roneneldan/TinyStories-1M",
        "source": "raw/recovery_sst2_tinystories_1m_fn24_trainable_alpha_attention_steps20000.json",
    },
    {
        "model": "roneneldan/TinyStories-3M",
        "source": "raw/recovery_sst2_tinystories_3m_fn24_trainable_alpha_attention_steps20000.json",
    },
    {
        "model": "roneneldan/TinyStories-8M",
        "source": "raw/recovery_sst2_tinystories_8m_fn24_trainable_alpha_attention_retry_shared_scalelr3e2_steps5000.json",
    },
    {
        "model": "roneneldan/TinyStories-28M",
        "source": "raw/recovery_sst2_tinystories_28m_fn24_trainable_alpha_attention_steps10000.json",
    },
    {
        "model": "roneneldan/TinyStories-33M",
        "source": "raw/recovery_sst2_tinystories_33m_fn24_trainable_alpha_attention_tasktrainable_rngneutral_steps3000.json",
    },
    {
        "model": "roneneldan/TinyStories-1Layer-21M",
        "source": "raw/recovery_sst2_tinystories_1layer_21m_fn24_trainable_alpha_attention_steps10000.json",
    },
    {
        "model": "roneneldan/TinyStories-2Layers-33M",
        "source": "raw/recovery_sst2_tinystories_2layers_33m_fn24_trainable_alpha_attention_tasktrainable_rngneutral_steps5000.json",
    },
    {
        "model": "allenai/OLMo-1B-0724-hf",
        "source": "raw/recovery_sst2_olmo_1b_fn24_trainable_alpha_attention_tasktrainable_steps6315.json",
    },
    {
        "model": "mistralai/Mistral-7B-v0.1",
        "source": "raw/recovery_sst2_mistral7b_fn24_trainable_alpha_attention_tasktrainable_steps500.json",
    },
]


RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS = [
    {
        "model": "roneneldan/TinyStories-1M",
        "source": "raw/recovery_sst2_tinystories_1m_fn24_bounded_alpha_attention_steps20000.json",
    },
    {
        "model": "roneneldan/TinyStories-3M",
        "source": "raw/recovery_sst2_tinystories_3m_fn24_bounded_alpha_attention_steps20000.json",
    },
    {
        "model": "roneneldan/TinyStories-8M",
        "source": "raw/recovery_sst2_tinystories_8m_fn24_bounded_alpha_attention_shared_scalelr3e2_steps5000.json",
    },
    {
        "model": "roneneldan/TinyStories-28M",
        "source": "raw/recovery_sst2_tinystories_28m_fn24_bounded_alpha_attention_steps10000.json",
    },
    {
        "model": "roneneldan/TinyStories-33M",
        "source": "raw/recovery_sst2_tinystories_33m_fn24_bounded_alpha_attention_shared_steps3000.json",
    },
    {
        "model": "roneneldan/TinyStories-1Layer-21M",
        "source": "raw/recovery_sst2_tinystories_1layer_21m_fn24_bounded_alpha_attention_steps10000.json",
    },
    {
        "model": "roneneldan/TinyStories-2Layers-33M",
        "source": "raw/recovery_sst2_tinystories_2layers_33m_fn24_bounded_alpha_attention_shared_steps5000.json",
    },
    {
        "model": "allenai/OLMo-1B-0724-hf",
        "source": "raw/recovery_sst2_olmo_1b_fn24_bounded_alpha_attention_shared_steps6315.json",
    },
    {
        "model": "mistralai/Mistral-7B-v0.1",
        "source": "raw/recovery_sst2_mistral7b_fn24_bounded_alpha_attention_shared_steps500.json",
    },
]



RECOVERY_FIELDNAMES = [
    "model",
    "family",
    "hidden_size_d",
    "task",
    "metric",
    "tensor_N",
    "protocol",
    "train_scope",
    "optimizer",
    "learning_rate",
    "batch_size",
    "grad_accum_steps",
    "max_steps",
    "baseline_target",
    "baseline_best_steps",
    "baseline_best_metric",
    "baseline_final_steps",
    "baseline_final_metric",
    "threshold",
    "fn_cap",
    "fn24_recovered",
    "fn24_first_recovery_steps",
    "fn24_first_recovery_metric",
    "fn24_best_steps",
    "fn24_best_metric",
    "fn24_final_steps",
    "fn24_final_metric",
    "fn24_cap_over_d",
    "source_file",
    "baseline_source_file",
]


RETAINED_RAW_OUTPUTS = {
    "raw/diag_llama32_1b_wikitext_N512_1024_2048_4096_8192_zonly.parquet",
    "raw/diag_mistral7b_wikitext_N512_1024_2048_4096_8192_zonly.parquet",
    "raw/diag_mistral7b_instruct_v02_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_qwen15_1p8b_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_qwen25_1p5b_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_qwen3_1p7b_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_phi4_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_gemma3_4b_pt_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_qwen25_0p5b_instruct_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_qwen3_0p6b_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_gemma3_1b_pt_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_deepseek_r1_distill_qwen_1p5b_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_phi3_mini_128k_instruct_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_smollm3_3b_wikitext_N512_1024_2048_4096_8192.parquet",
    "raw/diag_layernorm_dscale_sst2_N128.csv",
    "raw/diag_layernorm_dscale_gpt2_wikitext_N512.metadata.json",
    "raw/diag_layernorm_dscale_opt_wikitext_N512.metadata.json",
    "raw/diag_layernorm_dscale_opt_wikitext_N512_large.metadata.json",
    "raw/baseline_sst2_olmo_1b_0724_fullfinetune_bsz32_lr2e5_3epochs.json",
    "raw/recovery_sst2_olmo_1b_0724_fullfinetune_cap24_bsz32_lr2e5_3epochs_target933486.json",
    "raw/dscale_sst2_mistral7b_N128_caps24_48_96_steps500_headcaps.json",
    "raw/baseline_sst2_tinystories_1m_strict_steps10000_fulltrain.json",
    "raw/baseline_sst2_tinystories_3m_strict_steps5000_fulltrain.json",
    "raw/baseline_sst2_tinystories_8m_strict_steps5000_fulltrain.json",
    "raw/baseline_sst2_tinystories_28m_strict_steps5000_fulltrain.json",
    "raw/baseline_sst2_tinystories_33m_strict_steps5000_fulltrain.json",
    "raw/recovery_sst2_tinystories_1m_fn24_stricttarget823394_steps10000_fulltrain.json",
    "raw/recovery_sst2_tinystories_1m_fn24_low_lr1e4_steps20000_fulltrain.json",
    "raw/recovery_sst2_tinystories_1m_fn24_lr3e4_steps20000_fulltrain.json",
    "raw/recovery_sst2_tinystories_3m_fn24_stricttarget818807_steps10000_fulltrain.json",
    "raw/recovery_sst2_tinystories_3m_fn24_low_lr1e4_steps20000_fulltrain.json",
    "raw/recovery_sst2_tinystories_8m_fn24_stricttarget833716_steps10000_fulltrain.json",
    "raw/recovery_sst2_tinystories_28m_fn24_steps600_fulltrain.json",
    "raw/recovery_sst2_tinystories_33m_fn24_steps3000_fulltrain_target847477.json",
    "raw/recovery_sst2_tinystories_1layer_21m_d1024_fn24_steps3000_fulltrain.json",
    "raw/recovery_sst2_tinystories_2layers_33m_d1024_fn24_steps5000_fulltrain_target857798.json",
    *(
        run["source"]
        for run in RETAINED_TRAINABLE_ATTENTION_RECOVERY_RUNS
    ),
    *(
        run["source"].removesuffix(".json") + ".fn24.recovery.pt"
        for run in RETAINED_TRAINABLE_ATTENTION_RECOVERY_RUNS
    ),
    *(
        run["source"]
        for run in RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS
    ),
    *(
        run["source"].removesuffix(".json") + ".fn24.recovery.pt"
        for run in RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS
    ),
}



RETAINED_OUTPUT_DIRECTORIES = {
    "aggregate_n_scaling_extended",
    "aggregate_dscale_layernorm",
    "aggregate_recovery_fn_vs_d",
    "aggregate_trainable_attention_recovery",
    "aggregate_bounded_attention_recovery",
}


RETAINED_FIGURE_PREFIXES = (
    "figure_N_scaling_extended_",
    "figure_layernorm_factor_vs_d_",
)
