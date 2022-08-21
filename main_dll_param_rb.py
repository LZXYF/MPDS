import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AdamW, BertConfig
from loss import AutomaticWeightedLoss
from transformers import BertForSequenceClassification, AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification, DataCollatorForTokenClassification
import torch.nn.functional as F
import os
from textCNN_rand import textCNNModel as textCNNModel_rand
from textCNN import textCNNModel as textCNNModel_non
from textCNN_mu import textCNNModel as textCNNModel_mu
import numpy as np
import random

# 所有的参数都定义在这：
data_set = [{"name": "oh", "classes": 23, "test_num": 4043.0, "sentence_len": 128, "seed": 1},
            {"name": "mr", "classes": 2, "test_num": 3554.0, "sentence_len": 64, "seed": 200},
            {"name": "r52", "classes": 52, "test_num": 2569.0, "sentence_len": 128, "seed": 300},
            {"name": "r8", "classes": 8, "test_num": 2189.0, "sentence_len": 128, "seed": 400},
            {"name": "20ng", "classes": 20, "test_num": 7532.0, "sentence_len": 200, "seed": 500}]
selected = 0
cnnmodel = "non" # static non mu

# 模型 bert or roberta
BertModellist = ["bert-base-uncased", "roberta-base"]
selected_model = 1
BertModel = None

textCNNModel = textCNNModel_rand
if cnnmodel == "non":
    textCNNModel = textCNNModel_non
if cnnmodel == "mu":
    textCNNModel = textCNNModel_mu

# textcnn的参数
embedding_size = 300 # 300
kernel = 100 # 
num_classes = data_set[selected]["classes"]
dict_num = 0
if selected_model == 1:
  dict_num = 50265 # 50265 #  roberta: 50265 # bert: 30522
else:
  dict_num = 30522

en_tokenizer = None

sentence_len = data_set[selected]["sentence_len"]

max_score = 0.0

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

if selected_model == 0:
  en_tokenizer = AutoTokenizer.from_pretrained(BertModellist[selected_model])
  BertModel = BertForSequenceClassification
else:
  en_tokenizer = RobertaTokenizer.from_pretrained(BertModellist[selected_model])
  BertModel = RobertaForSequenceClassification

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyDataset(Dataset):

  def __init__(self, train=True, number=None):
    super().__init__()
    # 非测试模式
    if number is None:
      if train:
        url = "./data/" + data_set[selected]["name"] + "_train.txt"
      else:
        url = "./data/" + data_set[selected]["name"] + "_test.txt"
      if train:
        self.lines = open(url, "r", encoding="utf-8").readlines()
      else:
        self.lines = open(url, "r", encoding="utf-8").readlines()[:900]
    # 测试模式
    else:
        self.lines = open("./data/" + data_set[selected]["name"] + "_test.txt", "r", encoding="utf-8").readlines()


  def __getitem__(self, index):
    text = self.lines[index].strip("\n").split("\t")[0].strip()
    label = int(self.lines[index].strip("\n").split("\t")[-1])
    return text, label

  def __len__(self):
    return len(self.lines)


def collate_fn(batch):
  en_texts = []
  zh_texts = []

  texts = []
  texts_bert = []
  labels = []
  for i in range(len(batch)):
    en_texts.append(batch[i][0])
    labels.append(batch[i][1])

  or_texts = [en_tokenizer(en_text, max_length=sentence_len, truncation=True)["input_ids"] for en_text in en_texts]
  bert_ids = [tokenizer(en_text, max_length=sentence_len, truncation=True)["input_ids"] for en_text in en_texts]
  texts.append(or_texts)
  texts.append(bert_ids)

  for i in range(len(texts)):
    texts_i = texts[i]
    for j in range(len(texts_i)):
        if len(texts_i[j]) < sentence_len:
            texts_i[j] += [0 for _ in range(sentence_len-len(texts_i[j]))]
  return texts, labels


def main(l):
  global max_score
  max_score = 0.0
  # 定义训练设备
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  GSE_model = BertModel.from_pretrained(BertModellist[selected_model], num_labels=data_set[selected]["classes"],
                                                             output_attentions=True,
                                                             output_hidden_states=False)

  LFF_model = textCNNModel("bert-base-uncased", kernel, num_classes)

  GSE_model.to(device)
  LFF_model.to(device)

  # 开启训练模式
  GSE_model.train()
  LFF_model.train()

  batch_size = 36

  epoch = 150 #55

  dataset = MyDataset()
  dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

  with open("./logs/" + data_set[selected]["name"] + "_" + cnnmodel + "_out.log", "a", encoding="utf-8") as f:
      f.write("=======================================================l=" + str(l) + "===============\n")

  lr1 = 2e-05
  lr2 = 1e-03
  millist = [14, 36, 53, 65, 70, 90]
  ga = 0.5

  optimizer_grouped_parameters = [
    {'params': GSE_model.parameters(), "lr": lr1, "weight_decay_rate": 0.00005},
    {'params': LFF_model.parameters(), "lr": lr2, "weight_decay_rate": 0.00005}]

  loss_list = []
  optimizer = AdamW(optimizer_grouped_parameters)

  loss_compute = torch.nn.CrossEntropyLoss()
  schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=millist, gamma=ga)
  sm = torch.nn.Softmax(dim=1)
  score_list = []

  for _ in range(epoch):
    loss_list.append(0.0)
    GSE_model.train()
    LFF_model.train()

    print("第 " + str(_ + 1) + " 遍训练开始 ============================= 》》\n")
    with open("./logs/" + data_set[selected]["name"] + "_" + cnnmodel + "_out.log", "a", encoding="utf-8") as f:
      f.write("第" + str(_ + 1) + "遍训练开始！！\n")
      f.write("参数为：mil:" + str(millist) + ", 学习率：" + str(lr1) + "，" + str(lr2) + "...gamma=" + str(ga) + "\n")

    total_loss = [0.0, 0.0] 
    p = 30
    for i,(texts, labels) in enumerate(tqdm(dataloader)):
      optimizer.zero_grad()

      if (i+1) % p == 0:
        with open("./logs/" + data_set[selected]["name"] + "_" + cnnmodel + "_out.log", "a", encoding="utf-8") as f:
            f.write("第" + str(i+1) + "批次的精度："
                  + str(total_loss[0] / (p * (batch_size)))
                  + ", loss：" + str(total_loss[1] / p)
                  + "\n")

        total_loss = [0.0, 0.0]

      Y = labels
      Y = torch.tensor(Y).to(device)
      outs = []
      berts = [GSE_model, LFF_model]
      for ik in range(2):
        X = texts[ik]
        X = torch.tensor(X).to(device)
        out = berts[ik](X)
        if ik == 0:
          outs.append(sm(out["logits"]))
        else:
          outs.append(sm(out))

      r = round(1.0 - l, 1)

      final_out = outs[0] * l + r * (outs[1]+1e-10)

      final_out = torch.log(final_out)

      loss = loss_compute(final_out, Y).to(device)

      total_loss[1] += loss.item()
      loss_list[_] += loss.item()

      correct = torch.eq(torch.max(final_out, dim=1)[1], Y.flatten()).float()  # eq里面的两个参数的shape=torch.Size([16])
      total_loss[0] += correct.sum().item()

      loss.backward()

      # 梯度更新
      optimizer.step()

    # 学习率更新
    schedule.step()

    if _ > -1:
      # 测试
      score_item = oeval(l=l,GSE_model=GSE_model,LFF_model=LFF_model)
      loss_list[_] = round(loss_list[_] / 3357.0, 6) 
      if max_score < score_item:
          max_score = score_item
          GSE_model.save_pretrained("./" + data_set[selected]["name"] + "_model/" + str(int(l * 10)) + "/maxscore/")
          torch.save(LFF_model.state_dict(),
                   "./" + data_set[selected]["name"] + "_model/" + str(int(l * 10)) + "/maxscore/" + "model_state_dict.pth")

      score_list.append(score_item)
      with open("./logs/" + data_set[selected]["name"] + "_" + cnnmodel + "_out.log", "a", encoding="utf-8") as f:
        f.write("第 " + str(_ + 1) + " 次训练后的分数：" + str(score_item) + "\n")
        f.write("目前的最成熟模型分数：【 " + str(max_score) + " 】\n")
        f.write("目前损失值列表：【" + str(loss_list) + "】\n")

  with open("./logs/" + data_set[selected]["name"] + "_" + cnnmodel + "_out.log", "a", encoding="utf-8") as f:
    f.write("l = " + str(l) + " 时，分数变化列表：" + str(score_list) + "\n")
    f.write("其中的最成熟模型分数：【 " + str(max_score) + " 】\n")


# 模型评估
def oeval(l,GSE_model, LFF_model):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # 评估模式
  GSE_model.eval()
  LFF_model.eval()

  GSE_model.to(device)
  LFF_model.to(device)

  batch_size = 36
  sm = torch.nn.Softmax(dim=1)
  loss_compute = torch.nn.CrossEntropyLoss()

  testset = MyDataset(train=False, number=3000)
  testloader = DataLoader(testset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

  eval_loss = 0.0
  eval_acc = 0.0
  for e_i, (e_texts, e_labels) in enumerate(tqdm(testloader)):
    eval_Y = e_labels
    eval_Y = torch.tensor(eval_Y).to(device)
    eval_outs = []
    berts = [GSE_model, LFF_model]

    # 计算三种语言的损失值
    for k in range(2):
      eval_X = e_texts[k]
      eval_X = torch.tensor(eval_X).to(device)
      out = berts[k](eval_X)
      if k == 0:
        eval_outs.append(sm(out["logits"]))
      else:
        eval_outs.append(sm(out))

    r = round(1.0 - l, 1)
    eval_final_out = eval_outs[0] * l + r * (eval_outs[1] + 1e-10)
    eval_final_out = torch.log(eval_final_out)

    eval_loss += loss_compute(eval_final_out, eval_Y).to(device).item()

    eval_acc += torch.eq(torch.max(eval_final_out, dim=1)[1], eval_Y.flatten()).float().sum().item()

  score = eval_acc / (data_set[selected]["test_num"])
  return round(score, 4)


if __name__ == '__main__':
    # l = [0.9,0.8,0.7,0.6,0.5,0.0,0.4,0.3,0.2,0.1]
    l = [0.6]
    if os.path.exists("./logs/" + data_set[selected]["name"] + "_" + cnnmodel + "_out.log"):
      with open("./logs/" + data_set[selected]["name"] + "_" + cnnmodel + "_out.log", "a", encoding="utf-8") as f:
        f.write("\n\n\n\n\t【 " + BertModellist[selected_model] + " 】\t模型训练==：\n\n")
    else:
      with open("./logs/" + data_set[selected]["name"] + "_" + cnnmodel + "_out.log", "w", encoding="utf-8") as f:
        f.write("\n\n\n\n\t【 " + BertModellist[selected_model] + " 】\t模型训练==：\n\n")
    # 设置随机数种子
    # set_seed(data_set[selected]["seed"])
    for i in range(len(l)):
      set_seed(data_set[selected]["seed"])
      main(l=l[i])