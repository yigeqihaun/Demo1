from pathlib import Path
import json
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def calculate_accuracy(true_labels, pred_labels):
    """手动计算分类准确率。"""
    correct_count = 0
    total_count = len(true_labels)
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label == pred_label:
            correct_count += 1
    if total_count == 0:
        return 0.0
    accuracy = correct_count / total_count
    return accuracy

class Trainer:
    """
    训练器类。
    这个类将原来的两个训练相关函数统一管理：
    1. run_epoch：执行一轮训练或验证
    2. train：执行多轮训练
    """

    def __init__(self,model,train_loader,dev_loader,optimizer,scheduler,config,
        device,output_dir,best_accuracy,history,label2id,label_names,swanlab_run=None):
        """
        初始化训练器。
        optimizer、scheduler、output_dir、best_accuracy和history都在main.py中创建，再传入这里。
        """
        self.model = model      #保存模型
        self.train_loader = train_loader        #保存训练集DataLoader
        self.dev_loader = dev_loader        #保存验证集DataLoader
        self.optimizer = optimizer      #保存优化器
        self.scheduler = scheduler      #保存学习率调度器
        self.config = config        #保存训练配置
        self.device = device        #保存运行设备
        self.output_dir = output_dir        #保存输出目录
        self.best_accuracy = best_accuracy      #保存当前最佳验证集准确率
        self.history = history      #保存训练历史
        self.label2id = label2id        #保存标签映射
        self.label_names = label_names
        self.swanlab_run = swanlab_run      #保存SwanLab实验对象
        self.criterion = torch.nn.CrossEntropyLoss()        #多分类任务使用交叉熵损失

    def run_epoch(self, data_loader, training=True):
        """
        执行一轮训练或验证。training=True：训练模式，计算梯度并更新参数
        training=False：验证模式，只进行前向计算，不更新参数。
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        true_labels = []
        pred_labels = []

        for batch in tqdm(data_loader):
            batch = {
                key: value.to(self.device)
                for key, value in batch.items()     #将batch中的Tensor移动到CPU或GPU
            }

            model_inputs = {key: batch[key]for key in ["input_ids","attention_mask","token_type_ids"]if key in batch}

            with torch.set_grad_enabled(training):      #训练阶段允许计算梯度，验证阶段不计算梯度
                logits = self.model(**model_inputs)
                loss = self.criterion(logits,batch["labels"])

                if training:
                    self.optimizer.zero_grad()      #清空上一个batch的梯度
                    loss.backward()     # 反向传播
                    torch.nn.utils.clip_grad_norm_(         #梯度裁剪
                        self.model.parameters(),
                        self.config["grad_clip"]
                    )
                    self.optimizer.step()       #更新模型参数
                    self.scheduler.step()       #更新学习率

            total_loss += (loss.item() * batch["labels"].size(0))       #累加当前batch的损失
            predictions = logits.argmax(dim=-1)         #取分数最大的类别作为预测类别
            true_labels.extend(batch["labels"].detach().cpu().tolist())     #保存真实标签和预测标签
            pred_labels.extend(predictions.detach().cpu().tolist())

        # 计算平均损失
        average_loss = (total_loss / len(data_loader.dataset))

        # 手动计算准确率
        accuracy = calculate_accuracy(true_labels,pred_labels)

        return average_loss, accuracy

    def train(self):
        """
        执行完整的多轮训练。
        每轮训练结束后：
        1. 在验证集上评估；
        2. 记录 Loss 和 Accuracy；
        3. 如果验证集准确率提高，则保存模型。
        """

        for epoch in range(1 , self.config["epochs"] + 1):
            print(f"开始第 {epoch} 轮训练")
            train_loss, train_accuracy = self.run_epoch(self.train_loader,training=True)        #训练集训练
            dev_loss, dev_accuracy = self.run_epoch(self.dev_loader,training=False)         #验证集评估
            result = {                          #保存本轮结果
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "dev_loss": dev_loss,
                "dev_accuracy": dev_accuracy
            }
            self.history.append(result)
            print(result)
            if self.swanlab_run is not None:        #记录到SwanLab
                self.swanlab_run.log(result)

            if dev_accuracy > self.best_accuracy:       #只保存验证集表现最好的模型
                self.best_accuracy = dev_accuracy

                checkpoint = {
                    "model_state_dict": self.model.state_dict(),        #模型参数
                    "label2id": self.label2id,          #标签映射
                    "label_names": self.label_names,
                    "model_name": self.config["model_name"],        #模型配置信息
                    "num_labels": len(self.label_names),
                    "max_len": self.config["max_len"],
                    "dropout_rate": self.config["dropout_rate"],
                    "best_dev_accuracy": self.best_accuracy         #最佳验证集结果
                }

                torch.save(checkpoint,self.output_dir / "best_model.pt")

                print("已保存最佳模型、标签映射和配置信息")

        # 保存所有轮次的训练记录
        with open(self.output_dir / "history.json","w",encoding="utf-8") as file:
            json.dump(self.history,file,ensure_ascii=False,indent=2)



def evaluate_model(model,data_loader,device,label_names):
    model.eval()        #切换到测试模式
    true_labels = []     #保存真实标签
    pred_labels = []         #保存预测标签
    with torch.no_grad():       #测试阶段不计算梯度
        for batch in tqdm(data_loader):
            batch = {key: value.to(device)for key, value in batch.items()}
            model_inputs = {key: batch[key]for key in ["input_ids","attention_mask","token_type_ids"]if key in batch}
            logits = model(**model_inputs)
            predictions = logits.argmax(dim=-1)
            true_labels.extend(batch["labels"].cpu().tolist())       #保存真实标签
            pred_labels.extend(predictions.cpu().tolist())      #保存预测标签

    accuracy = calculate_accuracy(true_labels,pred_labels)
    report = classification_report(     #生成详细分类报告
        true_labels,
        pred_labels,
        target_names=label_names,
        digits = 4,
        zero_division = 0
    )

    print(f"测试集准确率：{accuracy:.4f}")
    print(report)

    return accuracy, report