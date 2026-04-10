import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_model_files(model_path):
    print(f"检查模型目录: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型目录不存在: {model_path}")
        return False
    
    required_files = [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'model.safetensors.index.json'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
            print(f"❌ 缺少文件: {file}")
        else:
            print(f"✓ 找到文件: {file}")
    
    if missing_files:
        print(f"\n❌ 模型不完整，缺少以下文件: {missing_files}")
        return False
    
    print(f"\n✓ 所有必需文件都存在")
    return True

def load_and_test_model(model_path):
    print(f"\n正在加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✓ Tokenizer 加载成功")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ 模型加载成功")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def test_conversation(model, tokenizer):
    print(f"\n开始对话测试...")
    
    test_messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "你好，请介绍一下你自己。"
        }
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        print(f"\n用户输入: {test_messages[1]['content']}")
        print("模型正在生成回答...")
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n模型回答: {response}")
        print("\n✓ 对话测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 对话测试失败: {e}")
        return False

def main():
    model_path = "./Qwen2.5-3B-Instruct"
    
    print("=" * 60)
    print("Qwen2.5-3B-Instruct 模型测试脚本")
    print("=" * 60)
    
    if not check_model_files(model_path):
        print("\n❌ 模型文件检查失败，请先下载完整的模型")
        sys.exit(1)
    
    model, tokenizer = load_and_test_model(model_path)
    if model is None or tokenizer is None:
        print("\n❌ 模型加载失败")
        sys.exit(1)
    
    if test_conversation(model, tokenizer):
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！模型可以正常使用")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 对话测试失败")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()