import pyrootutils
pyrootutils.setup_root(search_from = __file__,
                       indicator   = "README.md",
                       pythonpath  = True)

from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from server.modules.set_template import SetTemplate
from utils.configs import ParamConfig


class FilterChain():

    def __init__(self,
                 user_id,
                 model_path=None):

        self.user_id  = user_id
        self.template = SetTemplate(user_id)
        self.params   = ParamConfig()

        llm = ChatOpenAI(model       = self.template.load('chatgpt', 'params').model,
                         temperature = 0)

        prompt = PromptTemplate(
            input_variables = ["user_input"],
            template        = self.template.load_template('llama', 'crawl')
        )

        self._chain = LLMChain(llm=llm, prompt=prompt, verbose=False)



    def chain(self, question):
        return self._chain.predict(user_input=question)


    async def async_chain(self, question):
        return await self._chain.apredict(user_input=question)
    
        
if __name__ == "__main__":
    # model_path = "TheBloke/Llama-2-13B-Chat-GPTQ"
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    
    model_name = model_path.split('/')[-1]
    
    question = """부의 입장에서 생각을 자주해봅니다. 농업은 진짜 천한일인가? 사회의 악인가? 이런식의 이야기를 하더군요 나라의국부를 축내는 기생충같은 산업군으로... 전 정권에선 귀한대접이었지만, 지금 정권에선 때쓰는 나쁜놈들처럼 언론에서\xa0 이야기를 합니다. 그래서 최근 구토 방토 나온다고 기사로 엄중히 심판한거 같다는 생각합니다.\xa0 본론을 이야기하자면, 농업을 포기한 순간, 농업은 생각보다 다시 돌이키기가 어렵습니다. \n    석유야 없는거고 모 든 나라가 중동과 몇몇의 나라를 바라보니, 그들의 입장에따라 좌지 우지 되는데 또 다같이 중동을 압박하죠. 일부만 없는경우가 아니기에 중동도 미국과 중국의 눈치를 봐가며 석유 의 가격을 책정합니다. 석유를 무기화 하면 순식간에 중동의 이라크를 없앴던 미국의 사례를 찾아보면 알수 있습니다. \xa0 \n    하지만 농업은 어느 나라도 자기 스스로 해결 할 수 있는 건 해야 됩니다. 국제적 연대로 같이 싸워주지 않습니다. \n    저도 모든 농업을 다 지킬 필요가 있다고 생각하진 않지만, 모든 농업이 최소한의 경쟁력으로 준비를 해둬서 비상시국에 대비를 해야 된다고 합니다. 100년에 한번 엄청난 대기근이 세계적으로 온다고 하면 형제 자매를 버려가며 살게 할수 없으니까요... 석유 가격 상승은 경기가 침체되겠지만, 농업은 1년~2년안에 해결 못하면 많은 숫자가 죽어야 됩니다. 매년 얼마나 나올지 정확히 알수 없는것도 농사의 맹점이기도 하고요. 그래서 미국이 세계에서 가장 농업 보조금을 많 이주면서 FTA를 위반하면서까지 농업을 살리고 있습니다. \n    진짜 세상일 어떻게 될지 모르고, 어떻게 될지 모를때 진짜 문제가 생기면 모두가 굶어죽습니다. 이걸 선진국은 압니 다. \n    식량이 부족할지 모른다라는 불안감은 인류의 DNA에 있는겁니다. 한국도 경신대기근 같은 진짜 기아가 있었습니다. 세대가 멸절하는... \n    그리고 농사를 해보시면 정말 많이 심는다고 많이 나오지도 않고요. 기후위기에 대응해서 식물이 버티다가 임계점을 넘어가면 정말 말도 안되는 양의 소출이 나올가능성이 많다고 생각합니다. 그럴때 1년은 버티 지만 2년차에 갑자기 다시 모두 농지를 만들어서 농사를 짓는건 불가능한 일입니다. \n    정말 어려운 문제이지만, 농업이 아니고도 잘먹고 잘사는 미국, 영국, 프랑스 등의 선진국 들은 농촌을 여러가지 버퍼의 의미에서 계속 지키고 있습니다. 크게 생각하면 다음과 같습니다. \n    1. 위기때 생존, 죽은 사람은 나중에 식량이 많아져도 돌아올수 없는 비가역적 인 문제 \n    2. 농업인구 200만이 실업자로 된다면 나라가 거의 망하기 전까지 돌아가겠죠 \n    굳이 안 사오고 먹어도 100조 이상의 산업에서 먹고사는 농민 200만, 유통인 200만의 인구들이 다른 직업으로 가면 나라가 망할정도로 침체가 올 겁니다. 이분들을 전부 다 받아줄 산업이 어디있는지 궁금합니다. \n    3. 매년 100조가량 외화를 식량을 사느라 사와야 될겁니다. 삼성이 한 3개정도 있어서 최대 수익률 내는 모든 현금을 외국에 사야되니, 굳이 그러지 않는거죠... 이정도의 외화를 국내에 내수로 돌리는게 더 경제적이지 않을까요? \n    4. 비슷한 의미로 스타벅스가 있는데, 왜 자영업자 카페를 가야 되나, 미군이 있는데 왜 자국민들이 군대를 육성해야 되느냐, \n
                """
    # question = "hi"
    
  
    lp = FilterChain(user_id='asdf1234', model_path=model_path)
   
    print(lp.chain(question))
    