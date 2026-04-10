#!/root/miniconda3/bin/python
"""
Qwen2.5-Instruct 交互式评估脚本

加载训练好的 LoRA 权重，支持通过命令行进行金融文本分类推理。

用法:
    python eval_chat.py --lora_model ./output/lora_qwen2.5_20260224_160335/lora_model

交互命令:
    - 输入金融文本             → 模型对文本进行分类
    - clear                   → 清空对话历史
    - exit / quit             → 退出
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import cli

# ── 默认配置 ──────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "./Qwen2.5-3B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 100


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Qwen2.5-Instruct 交互式评估")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"基础模型路径（默认 {DEFAULT_BASE_MODEL}）")
    parser.add_argument("--lora_model", type=str, required=True,
                        help="LoRA 模型权重路径（如 ./output/lora_qwen2.5_20260224_160335/lora_model）")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help=f"最大生成 token 数（默认 {DEFAULT_MAX_NEW_TOKENS}）")
    return parser.parse_args()


def build_prompt(text: str) -> str:
    """构建金融文本分类的提示模板。

    Args:
        text: 金融文本内容。

    Returns:
        格式化后的提示文本。
    """
    prompt = (
        "请根据以下金融文本内容,判断其所属的标签类别。标签包括:国际、经济活动、市场、金属、政策、中立、积极、煤炭等。\n"
        "输出格式:标签1,标签2,...\n"
        f"文本内容:{text}\n"
        "标签:"
    )
    return prompt


def build_chat_messages(text: str) -> list[dict]:
    """构建聊天消息格式。

    Args:
        text: 金融文本内容。

    Returns:
        聊天消息列表。
    """
    return [
        {
            "role": "user",
            "content": build_prompt(text)
        }
    ]


def decode_response(output_ids: torch.Tensor, tokenizer) -> str:
    """解码生成的 token ID 为文本，去除特殊 token。

    Args:
        output_ids: 模型 generate 返回的 token ID [1, N]。
        tokenizer: Qwen2.5 分词器。

    Returns:
        解码后的纯文本字符串。
    """
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


def main():
    """交互式评估主函数。

    流程：
    1. 加载 Qwen2.5-Instruct 模型并加载训练好的 LoRA 权重
    2. 进入交互循环：支持输入金融文本进行分类、清空对话
    3. 每轮推理：构建提示 → 模型生成 → 显示结果
    """
    args = parse_args()

    cli.print_header("Qwen2.5-Instruct 交互式评估")

    # ── 设备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cli.print_kv("设备", str(device))

    # ── 加载模型 ──────────────────────────────────────────────────────
    cli.print_loading("Qwen2.5-Instruct 基础模型")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 启用flash attention
    if hasattr(base_model.config, 'use_flash_attention_2'):
        base_model.config.use_flash_attention_2 = True

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 加载 LoRA 权重
    cli.print_loading(f"加载 LoRA 权重: {args.lora_model}")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
        device_map="auto",
        dtype=torch.bfloat16
    )
    
    model.to(device)
    model.eval()

    cli.print_kv("最大生成长度", args.max_new_tokens)
    cli.print_divider()

    # ── 交互循环 ──────────────────────────────────────────────────────
    cli.print_info("交互命令:")
    cli.print_info("  - 输入金融文本             → 模型对文本进行分类")
    cli.print_info("  - clear                   → 清空对话历史")
    cli.print_info("  - exit / quit             → 退出")
    cli.print_divider()

    messages: list[dict] = []           # 对话历史
    round_num = 0

    while True:
        try:
            user_input = input("输入: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            cli.print_info("退出程序")
            break

        if not user_input:
            continue

        # ── 退出 ──
        if user_input.lower() in ("exit", "quit"):
            cli.print_info("退出程序")
            break

        # ── 清空对话 ──
        if user_input.lower() == "clear":
            messages.clear()
            round_num = 0
            cli.print_success("对话历史已清空")
            cli.print_divider()
            continue

        # ── 文本分类 ──
        round_num += 1
        cli.print_header(f"第 {round_num} 轮")

        # 构建聊天消息
        chat_messages = build_chat_messages(user_input)
        messages.extend(chat_messages)

        # 应用 chat template
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)

        # 模型推理
        cli.print_loading("分类中")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                temperature=0.0
            )

        response = decode_response(output_ids, tokenizer)

        # 追加助手回复
        messages.append({"role": "assistant", "content": response})

        # 打印回复
        cli.print_success("分类结果:")
        print(f"  {response}")
        cli.print_divider()


if __name__ == "__main__":
    main()