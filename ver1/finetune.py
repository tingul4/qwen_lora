import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info

class QwenSAMDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item["conversations"]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(conversations)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return inputs

def collate_fn(batch):
    # Simplified collate for batch_size=1
    return batch[0] 

def train():
    # Configuration
    model_id = "Qwen/Qwen2-VL-7B-Instruct" # Or local path
    jsonl_path = "/raid/mystery-project/qwen_lora/sam3_qwen_pipeline/qwen_finetune_data.jsonl"
    output_dir = "/raid/mystery-project/qwen_lora/sam3_qwen_pipeline/checkpoints"
    
    # Hyperparameters
    batch_size = 1 # User requested small batch size
    grad_accum_steps = 16 # User requested large accumulation
    num_epochs = 10
    learning_rate = 1e-5
    
    # Load Processor
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1024*28*28)
    
    # Load Model with QLoRA (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    # User requested: c_attn, w1, w2. Mapping to Qwen2 names.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Dataset
    dataset = QwenSAMDataset(jsonl_path, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10, 
        num_training_steps=len(dataloader) * num_epochs // grad_accum_steps
    )
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, inputs in enumerate(progress_bar):
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * grad_accum_steps
            progress_bar.set_postfix({"loss": loss.item() * grad_accum_steps})
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader)}")
        
        # Save checkpoint
        model.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))

if __name__ == "__main__":
    train()
