import pyrootutils
pyrootutils.setup_root(search_from = __file__,
                       indicator   = "README.md",
                       pythonpath  = True)
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils.configs import ParamConfig


class LlmPipeline():
    
    def __init__(self, model_path, user_id) -> None:
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map        = "auto",
                                                          trust_remote_code = False,
                                                          revision          = "main")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                                       use_fast=True)
        
              
        self.config = ParamConfig().load(Path(__file__).parent.parent / 'user_data' / user_id / 'template' / 'configs.yaml')
        
        
        
    
    def load(self):
        pipe =  pipeline(
                    task                 = "text-generation",
                    model                = self.model,
                    tokenizer            = self.tokenizer,
                    torch_dtype          = torch.bfloat16,
                    device_map           = "auto",
                    do_sample            = True,
                    eos_token_id         = self.tokenizer.eos_token_id,
                    **self.config.llama.params
                    )
        
        return pipe