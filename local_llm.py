"""Local GPU inference for Qwen3.5-4B, no API keys required."""
import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LocalQwen:
    def __init__(self, model_id='Qwen/Qwen3.5-4B', max_new_tokens=512):
        if not torch.cuda.is_available():raise RuntimeError('Kaggle GPU is required: Settings -> Accelerator -> GPU')
        self.max_new_tokens=max_new_tokens
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_id,device_map='auto',quantization_config=quant,torch_dtype=torch.float16,trust_remote_code=True).eval()
    @torch.inference_mode()
    def generate(self, messages):
        try:prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError:prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(prompt,return_tensors='pt').to(next(self.model.parameters()).device)
        input_tokens=int(inputs['input_ids'].shape[1])
        if input_tokens+self.max_new_tokens>8192:raise ValueError(f'Prompt too long: {input_tokens} tokens')
        output=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
        tokens=output[0,inputs['input_ids'].shape[1]:]
        result=self.tokenizer.decode(tokens,skip_special_tokens=True)
        if '</think>' in result:result=result.split('</think>')[-1].strip()
        return result,input_tokens,int(tokens.shape[0])
