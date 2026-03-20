import pyrootutils
pyrootutils.setup_root(search_from = __file__,
                       indicator   = "README.md",
                       pythonpath  = True)

import os
import pandas as pd
from addict import Dict
from pathlib import Path

from utils.configs import ParamConfig 


class SetTemplate():
    def __init__(self, 
                 user_id: str, 
                 ) -> None:
        
        self.user_id        = user_id
        self._BASE_SAVE_DIR = Path(__file__).parent.parent.parent / 'user_data' / user_id / 'template'
        self.params         = ParamConfig()

        
    @property
    def base_save_dir(self):
        
        return self._BASE_SAVE_DIR
    
    @base_save_dir.setter
    def set_base_dir(self, new_base_dir):
        
        self._BASE_SAVE_DIR = new_base_dir
    
    
    def load(self, llm:str, template_type:str):
        config = self.params.load(self.base_save_dir/ 'configs.yaml')
        
        return config[llm]['templates'][template_type]
    
    
    def load_template(self, llm:str, template_type:str):
        config = self.load(llm, template_type)
        
        return getattr(self,f'{template_type}_template')(config)
        
        
    def edit(self, llm:str, template_type:str, **kwargs:Dict): # SetTemplate 초기화 시 입력한 llm을 통해 llama 또는 chatgpt template설정 가능. 
        config = self.params.load(self.base_save_dir / 'configs.yaml')
        for key, item in kwargs.items():
            
            config[llm]['templates'][template_type][key] = item
        
        self.params.save(config, self.base_save_dir)
        
            
    # def check_llm_attr(self, target_llm):
    #     is_dir = [element.stem for element in self._BASE_SAVE_DIR.parent.parent.iterdir()]
    #     if target_llm in is_dir:
            
    #         return True
    #     else:    
            
    #         return {"status" : f'does not exist in the list. {is_dir}'}
        
        
    def crawl_template(self, kwargs:Dict) -> str: # 크롤러 분류 시 필요한 템플릿
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            
            if len(kwargs.system) < 6:
                system = kwargs.system_default
                
            else:
                system = kwargs.system
                
            if len(kwargs.prompt) < 6 :
                prompt = kwargs.prompt_default
                
            else:
                prompt = kwargs.prompt
            
            if (kwargs.company_info == '') and (kwargs.product_info == ''):
                info_template = f"\n\n{kwargs.company_info_default}\n\n{kwargs.product_info_default}"
                
            else:
                info_template = f"\n\n{kwargs.company_info}\n\n{kwargs.product_info}"
                
            SYSTEM_PROMPT = B_SYS + info_template + f'\n{system}' + E_SYS
            crawl_template =  B_INST + SYSTEM_PROMPT + "'User: {user_input}' " + prompt + E_INST
            
            return crawl_template
        
        
    def conversation_template(self, kwargs:Dict): ## 다시 보기
        
        if (len(kwargs.prompt) < 6) and (len(kwargs.system) < 6):
            system_prompt = kwargs.prompt_default
            
        else:
            system_prompt = kwargs.prompt
        
        converation_template =  system_prompt + ": {context}" + "\nQuestion: {question}"
        
        
        return converation_template
    
    
    def report_template(self, kwargs:Dict):
        if (len(kwargs.prompt) < 6):
            report_template = f"{kwargs.prompt_default}"
            
        else:
            report_template = f"{kwargs.prompt}"
            
        return report_template
    
    
    def standalone_template(self, kwargs:Dict):
        if (len(kwargs.system) < 6):
            standalone_template = f"{kwargs.system_default}"
            
        else:
            standalone_template = f"{kwargs.system}"
            
        return standalone_template
    
    
    def document_template(self, kwargs:Dict): ## 다시 보기
        
        if (len(kwargs.prompt) < 6):
            system_prompt = kwargs.prompt_default
            
        else:
            system_prompt = kwargs.prompt
        
        document_template =  system_prompt
        
        
        return document_template
                    
    
    def set_initial_templates(self,):
        dst = self.base_save_dir
        
        if dst.is_dir() == False:
            os.makedirs(name     = dst, 
                        exist_ok = True)
        
        init_config = self.params.load()
        self.params.save(init_config, dst)

        
if __name__ =="__main__":
    st = SetTemplate('asdf1234')
    print(st.load('chatgpt', 'params').model)
    # st.edit('chatgpt','params',**{'model': 'gpt-3.5-turbo'})
    
    
    
    
    
    
        