"""
示例：使用 RotationTrainer 进行 SpinQuant / QuaRot 训练

展示了 RotationTrainer 的多种使用模式，以及与 AutoRound 的集成方式。
"""

import sys
sys.path.insert(0, "/data/lkk/quarot/auto-round")

from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import (
    RotationTrainer,
    RotationTrainerConfig,
    LossLogger,
    OrthogonalityMonitor,
)
from auto_round.calib_dataset import get_dataloader


# ============================================================================
# 模式 1：RotationTrainer 基础用法（推荐）
# ============================================================================

def demo_trainer_basic():
    """最简用法：Trainer 训练 → fuse → AutoRound 量化。"""
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloader = get_dataloader(
        tokenizer, 1024,
        nsamples=128, dataset_name="NeelNanda/pile-10k"
    )

    # 创建 Trainer
    trainer = RotationTrainer(
        model,
        config=RotationTrainerConfig(
            r1=True, r2=True, r3=True, r4=True,
            trainable_rotation=True,
            trainable_smooth=True,
            iters=200,
            lr=1e-4,
            smooth_lr=1e-3,
            loss_type="kl_top",
            log_interval=50,
        ),
        callbacks=[
            LossLogger(log_interval=50),
            OrthogonalityMonitor(threshold=1e-3, log_interval=50),
        ],
    )

    # 训练
    metrics = trainer.train(dataloader)
    print(f"Training complete: best_loss={metrics['best_loss']:.6f}, steps={metrics['steps']}")

    # 融合
    model = trainer.fuse()

    # AutoRound 量化（此时模型已完成旋转融合）
    autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
    autoround.quantize()
    autoround.save_quantized("./Qwen3-0.6B-spinquant")


# ============================================================================
# 模式 2：QuaRot 模式（固定 Hadamard，不训练）
# ============================================================================

def demo_quarot_fixed():
    """QuaRot：固定 Hadamard，无训练，直接融合。"""
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    from auto_round.algorithms.transforms.spinquant import (
        SpinQuantConfig,
        SpinQuantPreprocessor,
    )

    # QuaRot: 固定 Hadamard，无训练
    config = SpinQuantConfig(
        r1=True, r2=True, r3=True, r4=True,
        trainable_rotation=False,    # ❌ 不训练
        trainable_smooth=False,
        iters=0,
    )

    preprocessor = SpinQuantPreprocessor(model, config)
    preprocessor.preprocess(dataloader=None)  # 无需 dataloader

    autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
    autoround.quantize()


# ============================================================================
# 模式 3：带评估的检查点保存/恢复
# ============================================================================

def demo_checkpoint_resume():
    """训练 100 步 → 保存 → 恢复 → 继续训练 100 步。"""
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloader = get_dataloader(tokenizer, seqlen=2048, dataset_name="NeelNanda/pile-10k", nsamples=128)

    config = RotationTrainerConfig(
        r1=True, r2=True,
        trainable_rotation=True,
        iters=100,
        lr=1e-4,
        save_interval=50,
        checkpoint_dir="./checkpoints",
    )
    trainer = RotationTrainer(model, config=config)

    # 第一轮训练
    metrics_1 = trainer.train(dataloader)
    print(f"Round 1: loss={metrics_1['loss_history'][-1]:.6f}")

    # 保存检查点
    ckpt_path = trainer.save_checkpoint()

    # ... 评估效果，不满意则继续 ...

    # 加载检查点，继续训练
    trainer.load_checkpoint(ckpt_path)
    config.iters = 200  # 延长到 200 步
    metrics_2 = trainer.train(dataloader)
    print(f"Round 2: loss={metrics_2['loss_history'][-1]:.6f}")

    model = trainer.fuse()


# ============================================================================
# 模式 4：自定义损失函数
# ============================================================================

def demo_custom_loss():
    """使用自定义损失函数训练旋转矩阵。"""
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloader = get_dataloader(tokenizer, seqlen=2048, dataset_name="NeelNanda/pile-10k", nsamples=128)

    def my_loss_fn(logits, ori_logits, config):
        """自定义：MSE + cosine similarity 混合损失。"""
        import torch.nn.functional as F
        mse = F.mse_loss(logits, ori_logits)
        cos = 1 - F.cosine_similarity(
            logits.flatten(0, -2), ori_logits.flatten(0, -2), dim=-1
        ).mean()
        return mse + 0.1 * cos

    trainer = RotationTrainer(
        model,
        config=RotationTrainerConfig(iters=200, lr=1e-4),
        compute_loss_fn=my_loss_fn,
    )
    trainer.train(dataloader)
    model = trainer.fuse()


# ============================================================================
# 模式 5：与 AutoRound 入口集成（未来 API）
# ============================================================================

def demo_autoround_integration():
    """
    理想情况下 AutoRound 的 API 可以这样设计：
    
    autoround = AutoRound(
        model, tokenizer,
        bits=4, group_size=128,
        enable_spinquant=True,
        spinquant_config=RotationTrainerConfig(
            r1=True, r2=True,
            trainable_rotation=True,
            iters=200,
        ),
    )
    autoround.quantize()  # 内部自动: SpinQuant 训练 → fuse → AutoRound 量化
    """
    pass


if __name__ == "__main__":
    demo_trainer_basic()

