import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import NewsDataset,build_label_mapping
from model import BertClassifier
from train import train_model,evaluate_model

def set_seed(seed):
    """
    固定随机种子，保证实验尽量可以复现。
    训练中的数据打乱、Dropout和模型参数初始化都可能包含随机过程。
    """
    random.seed(seed)       #固定Python自带的随机数
    np.random.seed(seed)        #固定NumPy的随机数
    torch.manual_seed(seed)     #固定PyTorch CPU随机数
    torch.cuda.manual_seed_all(seed)        #固定PyTorch GPU随机数
    torch.backends.cudnn.deterministic = True       #让cuDNN尽量使用确定性算法
    torch.backends.cudnn.benchmark = False      #关闭自动寻找最快算法


def main():
    parser = argparse.ArgumentParser()      #创建命令行参数解析器
    parser.add_argument("--config",default="config.json")       #设置配置文件参数，默认读取config.json
    args = parser.parse_args()      #读取命令行参数
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))      #读取JSON配置文件
    set_seed(config["seed"])        #设置随机种子
    device = torch.device("cuda" if torch.cuda.is_available()       #如果有可用GPU，就使用GPU；否则使用CPU
        else "cpu"
    )

    print("使用设备：", device)

    label2id, label_names = build_label_mapping(config["train_path"])       #根据训练集建立标签映射

    print("类别数量：", len(label_names))
    print("类别名称：", label_names)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])     #加载BERT分词器
    train_dataset = NewsDataset(config["train_path"],tokenizer,config["max_len"],label2id)      #创建训练集
    dev_dataset = NewsDataset(config["dev_path"],tokenizer,config["max_len"],label2id)      #创建验证集
    test_dataset = NewsDataset(config["test_path"],tokenizer,config["max_len"],label2id)        #创建测试集
    train_loader = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True,num_workers=config["num_workers"],collate_fn=train_dataset.collate_fn)     #创建训练集DataLoader，每轮训练前打乱训练数据，使用Dataset类中的自定义批处理函数
    dev_loader = DataLoader(dev_dataset,batch_size=config["batch_size"],shuffle=False,num_workers=config["num_workers"],collate_fn=dev_dataset.collate_fn)      #创建验证集DataLoader
    test_loader = DataLoader(test_dataset,batch_size=config["batch_size"],shuffle=False,num_workers=config["num_workers"],collate_fn=test_dataset.collate_fn)       #创建测试集DataLoader

    print("训练集数量：", len(train_dataset))
    print("验证集数量：", len(dev_dataset))
    print("测试集数量：", len(test_dataset))

    model = BertClassifier(                 #创建BERT分类模型
        model_name=config["model_name"],
        num_labels=len(label_names),
        dropout_rate=config["dropout_rate"]
    )

    model.to(device)
    swanlab_run = None

    if config["use_swanlab"]:       #根据配置决定是否启用SwanLab
        try:
            import swanlab
            swanlab_run = swanlab.init(project="bert-news-classification",config=config)        #创建一次实验记录
        except Exception as error:          #SwanLab失败不影响本地训练
            print("SwanLab 初始化失败，继续训练")
            print(error)

    train_model(model,train_loader,dev_loader,config,device,swanlab_run)
    best_model_path = Path(config["output_dir"]) / "best_model.pt"          #最佳模型保存路径
    model.load_state_dict(torch.load(best_model_path,map_location=device))          #加载验证集上表现最好的模型参数
    test_accuracy,test_report = evaluate_model(model,test_loader,device,label_names)        #在测试集上进行最终评估
    if swanlab_run is not None:
        swanlab_run.log({
            "test_accuracy": test_accuracy
        })


if __name__ == "__main__":
    main()