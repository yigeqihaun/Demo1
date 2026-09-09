from pathlib import Path
import json
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def run_epoch(model,data_loader,optimizer,scheduler,device,training=True,grad_clip=1.0):
    """
    执行一轮训练或验证。
    training=True：训练模式，需要反向传播和更新参数。
    training=False：验证模式，只计算结果，不更新参数。
    """
    if training:
        model.train()
    else:
        model.eval()         #训练模式启用Dropout，验证模式会关闭Dropout

    criterion = torch.nn.CrossEntropyLoss()      #多分类任务使用交叉熵损失
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for batch in tqdm(data_loader):
        batch = {key: value.to(device)for key, value in batch.items()}      #将batch中的Tensor移动到CPU或GPU
        model_inputs = {key: batch[key]for key in ["input_ids","attention_mask","token_type_ids"]if key in batch}        #只将模型需要的字段传入模型

        with torch.set_grad_enabled(training):       # training=True 时允许计算梯度；training=False 时不计算梯度。
            logits = model(**model_inputs)      #前向传播，得到分类分数
            loss = criterion(logits,batch["labels"])         #计算预测结果与真实标签之间的损失
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
                optimizer.step()
                scheduler.step()

        total_loss += (loss.item()* batch["labels"].size(0))        #将当前batch的损失乘以样本数，后面用于计算整个数据集的平均损失
        predictions = logits.argmax(dim=-1)     #取分数最大的类别作为预测类别
        all_labels.extend(batch["labels"].detach().cpu().tolist())      #保存真实标签
        all_predictions.extend(predictions.detach().cpu().tolist())     #保存预测标签

    average_loss = (total_loss / len(data_loader.dataset))      #计算整个数据集的平均损失
    accuracy = accuracy_score(all_labels,all_predictions)       #计算准确率
    return average_loss, accuracy


def train_model(model,train_loader,dev_loader,config,device,swanlab_run=None):
    """
    完整训练模型。
    每轮训练后都会在验证集上评估，如果验证集准确率更高，就保存当前模型。
    """
    optimizer = AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])       #创建AdamW优化器
    total_steps = (len(train_loader) * config["epochs"])        #总训练步数：每轮batch数×训练轮数
    warmup_steps = config["warmup_steps"]       #预热步数，表示前150个训练batch逐渐增大学习率。
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps)     #创建线性学习率调度器
    output_dir = Path(config["output_dir"])     #创建模型输出目录
    output_dir.mkdir(exist_ok=True)
    best_accuracy = 0.0     #当前最好的验证集准确率
    history = []        #保存每轮训练记录

    for epoch in range(1,config["epochs"] + 1):
        print(f"开始第 {epoch} 轮训练")
        train_loss, train_accuracy = run_epoch(model,train_loader,optimizer,scheduler,device,training=True,grad_clip=config["grad_clip"])
        dev_loss, dev_accuracy = run_epoch(model,dev_loader,optimizer,scheduler,device,training=False,grad_clip=config["grad_clip"])
        result = {                               #保存本轮结果
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_accuracy
        }

        history.append(result)      #加入历史记录
        print(result)

        if swanlab_run is not None:     #如果连接了SwanLab，就上传本轮指标
            swanlab_run.log(result)

        if dev_accuracy > best_accuracy:        #如果当前验证集准确率更高，就保存当前模型参数
            best_accuracy = dev_accuracy
            torch.save(model.state_dict(),output_dir / "best_model.pt")
            print("已保存最佳模型")

    with open(output_dir / "history.json","w",encoding="utf-8") as file:         #保存所有轮次的训练历史
        json.dump(history,file,ensure_ascii=False,indent=2)

def evaluate_model(model,data_loader,device,label_names):
    from sklearn.metrics import classification_report       #延迟导入分类报告函数
    model.eval()        #切换到测试模式
    all_labels = []     #保存真实标签
    all_predictions = []         #保存预测标签
    with torch.no_grad():       #测试阶段不计算梯度
        for batch in tqdm(data_loader):
            batch = {key: value.to(device)for key, value in batch.items()}
            model_inputs = {key: batch[key]for key in ["input_ids","attention_mask","token_type_ids"]if key in batch}
            logits = model(**model_inputs)
            predictions = logits.argmax(dim=-1)
            all_labels.extend(batch["labels"].cpu().tolist())       #保存真实标签
            all_predictions.extend(predictions.cpu().tolist())      #保存预测标签

    accuracy = accuracy_score(all_labels,all_predictions)
    report = classification_report(     #生成详细分类报告
        all_labels,
        all_predictions,
        target_names=label_names,
        digits = 4,
        zero_division = 0
    )

    print(f"测试集准确率：{accuracy:.4f}")
    print(report)

    return accuracy, report