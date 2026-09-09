import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self,model_name,num_labels,dropout_rate):
        """
        参数：
        model_name：预训练模型名称或本地路径。
        num_labels：分类类别数量。本项目为15。
        dropout_rate：Dropout比例，用于减轻过拟合。
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)       #加载预训练BERT模型
        self.dropout = nn.Dropout(dropout_rate)     #定义Dropout层
        self.classifier = nn.Linear(self.bert.config.hidden_size,num_labels)        #分类层

    def forward(self,input_ids,attention_mask,token_type_ids=None):
        """
        前向传播。
        input_ids：文本经过Tokenizer后得到的词编号。
        attention_mask：标记哪些位置是真实文本，哪些位置是Padding。
        token_type_ids：句子类型编号。单句分类任务中通常全为0。
        """
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output       #取BERT的句子级别表示
        pooled_output = self.dropout(pooled_output)      #使用Dropout防止模型过拟合
        logits = self.classifier(pooled_output)      #通过全连接层得到每个类别的分数

        return logits