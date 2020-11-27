""" 모델 모듈 """
from importlib import import_module

model = ""

def createModel(model_name, opt):
    """ 데이터 셋 생성 함수 """
    # 전역함수 등록
    global model
    
    # 해당되는 모듈 불러오기
    module = import_module("models." + model_name + "Model")

    model = getattr(module, model_name + "Model")(opt)

def getModel():
    """ 생성된 데이터 로더를 반환하는 함수 """
    return model