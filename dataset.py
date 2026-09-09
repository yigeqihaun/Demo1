from pathlib import Path
import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_len,label2id):
        """
        初始化数据集对象。
        参数：
        data_path：数据文件路径
        tokenizer：BERT分词器
        max_len：文本最大长度
        label2id：类别名称到数字编号的映射字典
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.texts = []     #保存所有新闻文本
        self.labels = []    #保存每条新闻对应的数字标签
        self._load_data()

    def _load_data(self):
        """
        读取和解析数据文件。
        提取：parts[2]：类别名称;parts[3]：新闻标题
        """
        lines = Path(self.data_path).read_text(encoding="utf-8").splitlines()        #一次性读取文件中的所有行
        for line in lines:
            line = line.strip()     #删除行首和行尾的空格

            if not line:
                continue    #空行跳过

            parts = line.split("_!_")   #根据_!_分割每条数据

            if len(parts) < 4:
                continue

            label_name = parts[2].strip()
            text = parts[3].strip()

            if label_name not in self.label2id:     #类别不在标签映射字典中就跳过
                continue

            self.texts.append(text)     #保存新闻标题
            self.labels.append(self.label2id[label_name])   #将类别名称转换为数字编号后保存

    def __len__(self):
        """
        返回数据集大小。
        DataLoader会调用这个方法，
        判断数据集中一共有多少条样本。
        """
        return len(self.texts)

    def __getitem__(self, index):
        """
        根据下标返回一条原始文本和标签。
        这里暂时不分词，
        分词工作交给 collate_fn统一处理。
        """
        text = self.texts[index]
        label = self.labels[index]
        return text, label

    def collate_fn(self, batch):
        """
        将多条样本整理成一个 batch。
        batch 的形式类似：
        [
            ("新闻标题1", 0),
            ("新闻标题2", 3),
            ...
        ]
        这个函数完成：
        1. 提取文本；
        2. 提取标签；
        3. 批量分词；
        4. Padding；
        5. 截断；
        6. 转换为 PyTorch Tensor。
        """
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        encoding = self.tokenizer(texts,max_length=self.max_len,padding=True,truncation=True,return_tensors="pt")
        encoding["labels"] = torch.tensor(labels,dtype=torch.long)

        return encoding

def build_label_mapping(data_path):
    """
    扫描数据文件，建立标签映射。
    返回两个对象：
    label2id：
        {
            "news_car": 0,
            "news_edu": 1
        }
    label_names：
        [
            "news_car",
            "news_edu"
        ]
    """
    labels = set()      #使用集合去重

    with open(data_path,"r",encoding="utf-8") as f:      #打开数据文件
        for line in f:
            parts = line.strip().split("_!_")       #读取并分割字段

            if len(parts) >= 3:      #至少要有类别字段
                label_name = parts[2].strip()       #提取类别名称
                labels.add(label_name)      #添加到集合中

    label_names = sorted(labels)        #排序后再编号，保证每次运行编号一致
    label2id = {}       #创建类别名称到数字编号的映射

    for index, label_name in enumerate(label_names):
        label2id[label_name] = index

    return label2id, label_names