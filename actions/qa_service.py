# -*- coding: utf-8 -*-
import time
from concurrent import futures
from bert_serving.client import ConcurrentBertClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import logging
import pickle

logger = logging.getLogger(__name__)

THRESHOLD = 0.7

topk = 5

prefix_item = '##### **ST:** '
prefix_q = '##### **Q:** '
prefix_a = '##### **A:** '

defaults = {
        "ip": 'localhost',
        "port": 5555,
        "port_out": 5556,
        "show_server_config": False,
        "output_fmt": 'ndarray',
        "check_version": False,
        "timeout": 5000,
        "identity": None,
        "batch_size": 128,
        "model_path": 'qs_model.data'
    }

class QAService:

    @classmethod
    def initBertClient(cls,bert_clinet_config=None):
        if bert_clinet_config:
            component_config = component_config
        else:
            component_config = defaults

        ip = component_config['ip']
        port = component_config['port']
        port_out = component_config['port_out']
        show_server_config = component_config['show_server_config']
        output_fmt = component_config['output_fmt']
        check_version = component_config['check_version']
        timeout = component_config['timeout']
        identity = component_config['identity']

        return ConcurrentBertClient(
            ip=ip,
            port=int(port),
            port_out=int(port_out),
            show_server_config=show_server_config,
            output_fmt=output_fmt,
            check_version=check_version,
            timeout=int(timeout),
            identity=identity
        )


    @classmethod
    def train(cls,data_path='actions/train.md',out_model_path='actions/qs_model.pk',bert_clinet_config=None):
        logger.debug('input path {} out model path {}'.format(data_path,out_model_path))

        if not data_path:
            raise Exception("输入的模型文件不能为空")
        elif not os.path.exists(data_path):
            raise Exception("文件不存在 '{}'.".format(data_path))
        elif not os.path.isfile(data_path):
            raise Exception("'{}' 不是一个文件.".format(data_path))
        
        qs_list = []
        qs_arr = []
        qs_id2tag = {}
        qs_tag2answer = {}

        logger.info('开始解析文件')
        with open(data_path,'r',encoding='utf8') as fin:
            qa_count = 0
            qs_count = 0
            tag = ''
            lines = fin.readlines()#读取全部内容  
            for line in lines:
                if line.strip():
                    if line.startswith(prefix_item) and not tag:
                        qa_count += 1
                        tag = 'qs_tag_' + str(qa_count)
                    elif line.startswith(prefix_q) and tag:
                        qs_list.append(line[len(prefix_q):])
                        qs_id2tag[qs_count] = tag
                        qs_count += 1
                    elif line.startswith(prefix_a) and tag:
                        qs_tag2answer[tag] = line[len(prefix_a):]
                        tag = ''

        logger.info('创建bert客户端')
        bc = cls.initBertClient(bert_clinet_config)
        choice_list = [' '.join(list(i)) for i in qs_list]

        logger.info('encode 问题')
        qs_arr = bc.encode(choice_list)

        logger.info('保存模型到文件:{}'.format(out_model_path))
        model = {}
        model['qs_list'] = qs_list
        model['qs_arr'] = qs_arr
        model['qs_id2tag'] = qs_id2tag
        model['qs_tag2answer'] = qs_tag2answer

        output_directory = os.path.dirname(os.path.abspath(out_model_path))
        print('out dir {}'.format(output_directory))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        with open(out_model_path,'wb') as fout:
            pickle.dump(model,fout)


    
    def loadModel(self,model_path='actions/qs_model.pk'):
        logger.debug('model path {}'.format(model_path))
        if not model_path:
            raise Exception("输入的模型文件不能为空")
        elif not os.path.exists(model_path):
            raise Exception("文件不存在 '{}'.".format(model_path))
        elif not os.path.isfile(model_path):
            raise Exception("'{}' 不是一个文件.".format(model_path))

        with open(model_path, 'rb') as fo:
            model = pickle.load(fo, encoding='bytes')
            self.qs_list = model['qs_list']
            self.qs_arr = model['qs_arr']
            self.qs_id2tag = model['qs_id2tag']
            self.qs_tag2answer = model['qs_tag2answer']

        if not self.qs_list:
            raise Exception('模型文件不完整，缺少:qs_list')
        if not self.qs_id2tag:
            raise Exception('模型文件不完整，缺少:qs_id2tag')
        if not self.qs_tag2answer:
            raise Exception('模型文件不完整，缺少:qs_tag2answer')
        
        index_size,*others = self.qs_arr.shape
        qs_size = len(self.qs_list)

        logger.info('qs_list size :{} qs_arr shape ({},)'.format(qs_size,index_size))

        if index_size != qs_size:
            raise Exception('模型文件有问题，qs_arr和qs_list大小不一致')


    def __init__(self, component_config=None):
        bert_client_config = component_config['bert_clinet_config'] if component_config else None
        model_path = component_config['model_path'] if component_config else None
        if bert_client_config:
            self.bc = self.initBertClient(bert_client_config)
        else:
            self.bc = self.initBertClient()

        if model_path :
            self.loadModel(model_path)
        else:
            self.loadModel()

        logger.info('start qs matching service ...')

    def get_similarity(self, qs):
        input_seg = ' '.join(list(qs))
        input_arr = self.bc.encode([input_seg])
        max_score = 0
        max_i = -1
        for i in range(len(self.qs_list)):
            sim = cosine_similarity(self.qs_arr[i].reshape(-1, 768), input_arr)
            logger.debug('[{} | {}]\tsimilarity\t:\t{}'.format(qs, self.qs_list[i], sim[0][0]))
            if sim[0][0] > max_score:
                max_score = sim[0][0]
                max_i = i
        if max_score > THRESHOLD:
            return max_score, self.qs_list[max_i], max_i
        else:
            return 0, '不知道你说的啥', -1

    def get_similarity_np(self, qs):
        query_seg = ' '.join(list(qs))
        query_vec = self.bc.encode([query_seg])[0]

        # compute normalized dot product as score
        score = np.sum(query_vec * self.qs_arr, axis=1) / np.linalg.norm(self.qs_arr, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]

        if logger.isEnabledFor(logging.DEBUG):
            for idx in topk_idx:
                logger.debug('输入:{} 匹配:{} 相似度:{}'.format(qs,self.qs_list[idx],score[idx]))

        max_i = topk_idx[0]
        max_score = score[max_i]
        
        return max_score, self.qs_list[max_i], max_i
        

    def getanswer(self, qs):
        score, res, index = self.get_similarity_np(qs)
        logger.debug('输入:{} 最佳匹配:{} 相似度:{}'.format(qs,res,score))
        if index >= 0 and index < len(self.qs_list):
            tag = self.qs_id2tag[index]
            result = self.qs_tag2answer[tag]
        else:
            result = None
        return result


def test_service():
    ser = QAService()
    qs = "有几种类型的模版"
    result = ser.getanswer(qs)
    print('{}\t{}'.format(qs, result))

if __name__ == '__main__':
    QAService.train()
    test_service()