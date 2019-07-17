1. 完善train.md 把操作手册里的问答对添加到文件
2. 完善NLU.json 把问题及意图添加到文件
3. 修改stories.md 完善场景

项目目录
pips install -r requirements.txt
pip3 install pip install bert-serving-server

训练
1. 启动Bert-as-service
bert-serving-start -model_dir bert_mode/chinese_L-12_H-768_A-12/ -num_worker=4 
2. 训练QAService 模型
python3 actons/qa_service.py
3. 训练rasa模型
make train


模型验证
1. 启动Bert-as-service 
bert-serving-start -model_dir bert_mode/chinese_L-12_H-768_A-12/ -num_worker=4 
2. 启动action服务
rasa run actions --actions actions.actions --debug
3. 启动rasa
rasa shell --endpoints configs/endpoints.yml -m models --debug

