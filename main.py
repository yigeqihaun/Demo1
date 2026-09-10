import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from dataset import NewsDataset
from dataset import build_label_mapping
from model import BertClassifier
from train import Trainer
from train import evaluate_model

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
    parser.add_argument("--config",default="config/config_base.json")       #设置配置文件参数，默认读取config_base.json
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

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"],local_files_only=True)     #加载BERT分词器
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

    model.to(device)        #将模型移动到CPU或GPU

    optimizer = AdamW(          #在main.py中创建AdamW优化器
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    total_steps = (len(train_loader) * config["epochs"])        #计算总训练步数

    #在main.py中创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=config["warmup_steps"],num_training_steps=total_steps)

    #创建模型输出目录
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True,exist_ok=True)

    best_accuracy = 0.0     #初始化训练过程中的状态变量
    history = []

    swanlab_run = None      #初始化SwanLab

    if config["use_swanlab"]:
        try:
            import swanlab

            swanlab_run = swanlab.init(project="bert-news-classification",config=config)

        except Exception as error:
            print("SwanLab 初始化失败，继续本地训练。")
            print(error)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        output_dir=output_dir,
        best_accuracy=best_accuracy,
        history=history,
        label2id=label2id,
        label_names=label_names,
        swanlab_run=swanlab_run
    )

    # 开始训练
    trainer.train()

    # 最佳模型路径
    best_model_path = ( output_dir/ "best_model.pt")

    # 加载完整 checkpoint
    checkpoint = torch.load(best_model_path,map_location=device)

    # 从 checkpoint 中读取模型权重
    model.load_state_dict(checkpoint["model_state_dict"])
    print("最佳验证集准确率：",checkpoint["best_dev_accuracy"])
    print("标签映射：",checkpoint["label2id"])

    # 使用最佳模型在测试集上评估
    test_accuracy, test_report = evaluate_model(model,test_loader,device,checkpoint["label_names"])

    # 将测试集准确率上传到 SwanLab
    if swanlab_run is not None:
        swanlab_run.log({"test_accuracy": test_accuracy})


if __name__ == "__main__":
    main()