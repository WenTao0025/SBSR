"""
采用了ViT_L_16
"""
import json
import os
import random
import time
from sklearn.metrics import  average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from timm.data import Mixup
from Datasets.VitB.RenderDataloader import load_model_datasets
from Datasets.VitB.SketchModelDatasets import load_model_query_datasets
from Datasets.VitB.SketchNetDataloader import load_sketch_datasets
from Nets.VIT.CrossDomainNet2 import CrossDomainNet
from configs.Vit.config import config_argument
# class CLIPLoss(nn.Module):
#     def __init__(self):
#         super(CLIPLoss, self).__init__()
#         self.logit_scale = nn.Parameter(torch.tensor(1.0))  # 初始化 t 参数，可以进行优化
#
#     def forward(self, image_features, shape_features,label):
#         # Normalize features
#         image_features = F.normalize(image_features, dim=-1)  # 对图像特征向量进行归一化
#         shape_features = F.normalize(shape_features, dim=-1)    # 对文本特征向量进行归一化
#
#         # Compute logits using the exponential of the logit_scale
#         logit_scale = self.logit_scale.exp()  # 计算 logit_scale 的指数，得到标量 t
#         logits_per_image = logit_scale * torch.matmul(image_features, shape_features.t())  # 计算图像和文本之间的相似度矩阵
#         logits_per_shape = logits_per_image.t()  # 转置得到文本与图像之间的相似度矩阵
#
#         labels = (label.unsqueeze(1) == label.unsqueeze(0)).float()
#
#         # Calculate cross-entropy loss for image-to-text and text-to-image
#         loss_i = F.binary_cross_entropy_with_logits(logits_per_image, labels)
#         loss_t = F.binary_cross_entropy_with_logits(logits_per_shape, labels.t())
#         loss = (loss_i + loss_t) / 2  # 平均损失
#
#         return loss
arg = config_argument()

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.deterministic = True  # 固定网络结构
set_seeds(2)
mixup = Mixup(mixup_alpha=1.0, cutmix_alpha=1.0, prob=1.0,switch_prob=0.5, label_smoothing=0.3, num_classes=arg.class_num)
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, imgs):
        """
        Args:
            imgs (Tensor): Tensor batch of images of size (B, C, H, W).
        Returns:
            Tensor: Batch of images with n_holes of dimension length x length cut out of each.
        """
        batch_size, channels, height, width = imgs.size()

        for i in range(batch_size):
            for n in range(self.n_holes):
                y = np.random.randint(height)
                x = np.random.randint(width)

                y1 = np.clip(y - self.length // 2, 0, height)
                y2 = np.clip(y + self.length // 2, 0, height)
                x1 = np.clip(x - self.length // 2, 0, width)
                x2 = np.clip(x + self.length // 2, 0, width)

                imgs[i, :, y1:y2, x1:x2] = 0.

        return imgs
class weighted_entropy_ce(nn.Module):
    def __init__(self):
        super(weighted_entropy_ce, self).__init__()
        self.CE = nn.CrossEntropyLoss()
    def forward(self, x_input, y_target, weight, e_lambda = 1):
        weight = weight.reshape(-1, 1)
        # print(weight.shape)
        p = F.softmax(x_input,dim=-1)
        p = p.detach()
        entropy = - torch.sum(p * F.log_softmax(x_input,dim=-1), dim=1).reshape(-1, 1)
        # print(entropy.shape)
        # print(rank_input1)

        weight_beta = e_lambda * weight
        # weight_1 = torch.ones_like(weight_beta) - weight_beta
        entropy = weight_beta * entropy
        # print(entropy)


        loss = self.CE(x_input,y_target)
        loss = torch.mean(loss) - torch.mean(entropy)
        # print(loss)
        return loss
class TrainCrossNet(nn.Module):
    def __init__(self):
        super(TrainCrossNet, self).__init__()
        self.device = "cuda:3"
        self.best_acc = 0
        self.best_NN = 0
        self.best_FT = 0
        self.num_view = arg.num_views
        self.best_ST = 0
        self.channels = arg.channels
        self.pix_size = arg.img_size
        self.num_view = arg.num_views
        self.best_DCG = 0
        self.beta1 = arg.beta1
        self.beta2 = arg.beta2
        self.lr = arg.lr
        self.net = CrossDomainNet().to(self.device)
        self.epochs = 1000
        self.opt = torch.optim.Adam(self.net.parameters(),lr=self.lr,betas=(self.beta1,self.beta2))
        self.load2()
    def load2(self):
        net = self.net
        trained_weight = torch.load(arg.save_path, map_location=self.device)
        cross_net_status = self.net.state_dict()

        # 获取 cross_net_status 和 trained_weight 的键列表
        cross_net_keys = list(cross_net_status.keys())
        trained_weight_keys = list(trained_weight.keys())

        # 确保两个列表长度一致，如果不一致可能需要额外处理
        if len(cross_net_keys) != len(trained_weight_keys):
            raise ValueError("The number of keys in cross_net_status and trained_weight does not match.")

        # 通过列表序号来读取权重
        for i in range(len(cross_net_keys)):
            cross_net_status[cross_net_keys[i]] = trained_weight[trained_weight_keys[i]]

        net.load_state_dict(cross_net_status, strict=False)
        print('参数加载完成')
    def load(self):
        query_trained_weight = torch.load(arg.SketchNet_params,map_location=self.device)
        model_trained_weight = torch.load(arg.ModelNet_params,map_location=self.device)
        query_keys = query_trained_weight.keys()
        query_keys = list(query_keys)
        model_keys = model_trained_weight.keys()
        model_keys = list(model_keys)
        i=9
        j=0
        cross_net_status = self.net.state_dict()
        for names,_ in cross_net_status.items():
            if names.startswith('QueryNet'):
                cross_net_status[names] = query_trained_weight[query_keys[i]]
                i += 1
        for names,_ in cross_net_status.items():
            if names.startswith('ModelNet'):
                cross_net_status[names] = model_trained_weight[model_keys[j]]
                j += 1
        self.net.load_state_dict(cross_net_status,strict=True)
        print("参数加载完成")

    def Cross_Push_Pull(self, labels, query_emb, model_emb):
        # 归一化嵌入向量
        device = self.device
        temperature = arg.tau

        # 归一化查询和模型嵌入
        query_emb = F.normalize(query_emb, p=2, dim=1).to(device)
        model_emb = F.normalize(model_emb, p=2, dim=1).to(device)

        # 计算相似度矩阵
        logits = (query_emb @ model_emb.T) / temperature

        # 生成掩码
        labels = labels.contiguous().view(-1, 1).to(device)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 计算 exp(logits)
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 计算对比损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 计算异类推远项（推远不同类别的样本）
        # 计算查询样本和模型样本之间的余弦相似度
        cosine_distances = 1 - F.cosine_similarity(query_emb.unsqueeze(1), model_emb.unsqueeze(0), dim=2)  # 计算余弦距离

        # 创建标签不相等的掩码，只考虑不同类别的样本对
        label_diff_mask = 1 - torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)

        # 计算异类推远项（平方差和），仅考虑不同类别的样本对
        hetero_pushaway_term = (cosine_distances ** 2 * label_diff_mask).sum()

        # 最终损失：对比损失 + 异类推远项
        loss = -mean_log_prob_pos.mean() + arg.lambda_pushaway * hetero_pushaway_term

        return loss

    def run(self, x, left, right):
        self.min, self.max = 1e20, -1e10
        self.min = min(x.min(), self.min)
        self.max = max(x.max(), self.max)
        k = (right-left)/(self.max - self.min)
        return left+k*(x - self.min)
    def softmax(self,X):
        X -= X.max()
        X_exp = X.exp()
        max = X_exp.max() + 0.001
        return X_exp / max
    def Train(self):
        self.train_datasets = load_model_query_datasets('train')
        with open("per_model_num.json","r") as json_file:
            per_model_num = json.load(json_file)
        criterion = weighted_entropy_ce().to(self.device)
        maha_intermediate_dict = np.load(arg.maha_file, allow_pickle='TRUE')
        class_cov_invs = maha_intermediate_dict.item()['class_cov_invs']
        class_means = maha_intermediate_dict.item()['class_means']
        cov_invs = maha_intermediate_dict.item()['cov_inv']
        means = maha_intermediate_dict.item()['mean']
        for epoch in range(self.epochs):
            self.net.train()
            pbar = tqdm.tqdm(self.train_datasets)
            for meta in pbar:
                query_img = meta['query_img'].to(device = self.device)
                query_img = query_img.reshape(-1,self.channels,arg.img_size,arg.img_size)
                render_img = meta['render_img'].to(device = self.device)
                render_img= render_img.reshape(-1,self.channels,arg.img_size,arg.img_size).to(device = self.device)
                label = meta['label_cat'].to(device = self.device)
                class_model_num = list(per_model_num.values())
                counts_tensor = torch.tensor(class_model_num, dtype=torch.float32,device=self.device)
                weights = torch.log(torch.tensor(2.0,device=self.device)) - torch.log(counts_tensor)   # 取倒数
                weights /= weights.sum()  # 归一化，使权重总和为 1
                self.CE = nn.CrossEntropyLoss()
                query_img, labels = mixup(query_img, label)
                clf_emb1,clf_emb2,model_emb,model_emb2 = self.net(query_img,render_img)
                pre_feature = clf_emb2.cpu().data.numpy()
                maha_distance = get_relative_maha_distance(pre_feature, cov_invs, class_cov_invs, means, class_means,
                                                           label.cpu().data.numpy())
                maha_distance = torch.from_numpy(maha_distance)
                maha_distance_normalized = self.run(maha_distance, arg.left, arg.right)
                maha_distance_normalized = self.softmax(maha_distance_normalized / arg.T)

                maha_weight = maha_distance_normalized.to(self.device)
                loss1 = criterion(clf_emb2, labels, maha_weight, arg.e_lambda)
                loss2 = self.CE(model_emb2,labels)
                loss3 = self.Cross_Push_Pull(label,clf_emb1,model_emb) + self.Cross_Push_Pull(label,model_emb,clf_emb1)
                # loss3 = self.CLIPLoss(clf_emb1,model_emb,label)
                loss = loss1 + loss2
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                info_dict = {'loss' : '%.7f' % (loss.item())}
                pbar.set_postfix(info_dict)
                pbar.set_description('Epoch : %d ' % (epoch))
            if (epoch % 1 == 0):
                #torch.cuda.empty_cache()
                self.test()
            # main()
    def test_train(self):
        device = self.device
        self.net.eval()
        data_test_model = load_model_datasets(bs=1, data_type='train')
        data_test_query = load_sketch_datasets('train', bs=1)

        render_embs_list = []  # 存放特征图的list
        cat_label_model_list = []  # 存放模型的类别标签

        # 先将所有的model_render经过网络映射成特征图
        pbar = tqdm.tqdm(data_test_model)
        with torch.no_grad():
            for meta in pbar:
                render_img = meta['query_or_model_img'].to(device=self.device)
                cat_label_model = meta['label_cat'].to(device=self.device)

                render_img = render_img.reshape(-1, self.channels, arg.img_size, arg.img_size).to(device=self.device)
                render_emb = self.net.get_render_emb(render_img)
                cat_label_model_list.append(cat_label_model)
                render_embs_list.append(render_emb)
        render_embs = torch.concat(render_embs_list, dim=0)
        cat_label_model_list = torch.concat(cat_label_model_list, dim=0).reshape(-1)
        model_lenth = render_embs.size(0)

        results = []
        query_labels = []
        pbar = tqdm.tqdm(data_test_query)
        with torch.no_grad():
            for meta in pbar:
                query_img = meta['sketch_img'].to(device=self.device)
                cat_label_query = meta['label_cat'].to(device=self.device)
                bs = cat_label_query.size(0)
                query_labels.append(cat_label_query)
                query_emb = self.net.get_sketch_emb(query_img)

                query_emb = query_emb.repeat_interleave(model_lenth, dim=0).reshape(1, model_lenth, arg.embding_size)
                render_embs = render_embs.reshape(1, model_lenth, arg.embding_size)
                similarity_score = F.cosine_similarity(query_emb, render_embs, dim=2)[0]
                sorted_similarity_score, sorted_index = torch.sort(similarity_score, descending=True)
                query_result_model_label = cat_label_model_list[sorted_index]
                query_label = cat_label_query.repeat(model_lenth)

                results.append((query_label == query_result_model_label))
        results = torch.stack(results, dim=0)
        query_labels = torch.stack(query_labels, dim=0)

        # 计算指标
        NN = results[:, 0].float().mean().item()

        cls_counts = torch.bincount(cat_label_model_list)
        FT, ST = calculate_retrieval_scores(results, cls_counts, query_labels)

        temp123 = results.sum()
        s = results[:, :32].sum()
        p = s / (results.size(0) * 32)
        rr = s / temp123
        E_measure = 2 * p * rr / (p + rr )

        # DCG 和 mAP
        DCG, mAP = calculate_dcg_and_map(results, device)

        print(f"NN: {NN:.3f}, FT: {FT:.3f}, ST: {ST:.3f}, E: {E_measure:.3f}, DCG: {DCG:.3f}, mAP: {mAP:.3f}")

    def test(self):
        device = self.device
        self.net.eval()
        data_test_model = load_model_datasets(bs=1, data_type='test')
        data_test_query = load_sketch_datasets('test', bs=1)

        render_embs_list = []  # 存放特征图的list
        cat_label_model_list = []  # 存放模型的类别标签

        # 先将所有的model_render经过网络映射成特征图
        pbar = tqdm.tqdm(data_test_model)
        with torch.no_grad():
            for meta in pbar:
                render_img = meta['query_or_model_img'].to(device=self.device)
                cat_label_model = meta['label_cat'].to(device=self.device)

                render_img = render_img.reshape(-1, self.channels, arg.img_size, arg.img_size).to(device=self.device)
                render_emb = self.net.get_render_emb(render_img)
                cat_label_model_list.append(cat_label_model)
                render_embs_list.append(render_emb)

        render_embs = torch.concat(render_embs_list, dim=0)
        cat_label_model_list = torch.concat(cat_label_model_list, dim=0).reshape(-1)
        model_lenth = render_embs.size(0)

        results = []
        query_labels = []
        query_embs = []
        pbar = tqdm.tqdm(data_test_query)
        with torch.no_grad():
            for meta in pbar:
                query_img = meta['sketch_img'].to(device=self.device)
                cat_label_query = meta['label_cat'].to(device=self.device)
                bs = cat_label_query.size(0)
                query_labels.append(cat_label_query)
                query_emb = self.net.get_sketch_emb(query_img)
                query_embs.append(query_emb)

                query_emb = query_emb.repeat_interleave(model_lenth, dim=0).reshape(1, model_lenth, arg.embding_size)
                render_embs = render_embs.reshape(1, model_lenth, arg.embding_size)
                similarity_score = F.cosine_similarity(query_emb, render_embs, dim=2)[0]
                sorted_similarity_score, sorted_index = torch.sort(similarity_score, descending=True)
                query_result_model_label = cat_label_model_list[sorted_index]
                query_label = cat_label_query.repeat(model_lenth)

                results.append((query_label == query_result_model_label))

        results = torch.stack(results, dim=0)
        # 保存结果数据到文件
        torch.save(results, '../emb/result.pth')
        print(f"Results saved to {'../emb/result.pth'}")
        query_labels = torch.stack(query_labels, dim=0)
        query_embs = torch.stack(query_embs,dim=0)

        # 计算指标
        NN = results[:, 0].float().mean().item()

        cls_counts = torch.bincount(cat_label_model_list)
        FT, ST = calculate_retrieval_scores(results, cls_counts, query_labels)

        temp123 = results.sum()
        total_query = results.size(0)

        # 动态调整前k个检索结果的数量，最大为32
        E_list = torch.zeros((total_query, 1))

        for i in range(total_query):
            true_label = query_labels[i].item()
            # 获取该类别的相关模型数量，最多为32
            k = 32  # 选择相关模型数量（最多32个）

            s = results[i, :k].sum()  # 只考虑前k个结果

            # 计算精确率p和召回率rr
            p = s / (k * 1.0)  # 使用实际相关模型数
            rr = s / cls_counts[true_label] if cls_counts[true_label] > 0 else 0  # 总体相关模型数

            if (p == 0) and (rr == 0):
                E_list[i] = 0
            else:
                E_list[i] = 2 * p * rr / (p + rr + 1e-6)  # 防止除以零

        E_measure = E_list.mean()

        # DCG 和 mAP
        DCG, mAP = calculate_dcg_and_map(results, device)
        print(f"NN: {NN:.3f}, FT: {FT:.3f}, ST: {ST:.3f}, E: {E_measure:.3f}, DCG: {DCG:.3f}, mAP: {mAP:.3f}")
        if self.best_NN < NN:
            self.best_NN = NN
            self.best_FT = FT
            self.best_ST = ST
            self.best_F = E_measure
            self.best_DCG = DCG
            self.mAP = mAP
            # self.save()


        # 计算Precision-Recall曲线
        y_true = []
        y_scores = []

        # 遍历所有查询样本
        for i in range(total_query):
            true_label = query_labels[i].item()
            query_emb = query_embs[i]
            # 获取该查询样本的所有视图标签（模型标签）
            current_view_labels = cat_label_model_list.cpu().numpy()  # 需要将其移到CPU并转为numpy
            # 获取与当前查询样本相关的所有视图的相似度（相似度得分）
            query_emb = query_emb.repeat_interleave(model_lenth, dim=0).reshape(1, model_lenth, arg.embding_size)
            similarity_score = F.cosine_similarity(query_emb, render_embs, dim=2)[0]
            current_distances = similarity_score.cpu().numpy()  # 这里改为使用 similarity_score 而不是 results[i]

            # 创建二进制标签：相关为1，不相关为0
            binary_labels = (current_view_labels == true_label).astype(int)

            # 添加到列表
            y_true.extend(binary_labels)
            y_scores.extend(current_distances)

        # 将 y_true 和 y_scores 转换为 NumPy 数组，确保它们在 CPU 上
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # 计算 Precision-Recall 曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)

        # 绘制 PR 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='b', lw=2, label=f'PR Curve (AP = {average_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("SHREC'13 Precision-Recall Curve")
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig('pr_curve.png')
        plt.close()
    def PR_curve(self):
        device = self.device
        self.net.eval()
        data_test_model = load_model_datasets(bs=1, data_type='test')
        data_test_query = load_sketch_datasets('test', bs=1)

        render_embs_list = []  # 存放特征图的list
        cat_label_model_list = []  # 存放模型的类别标签

        # 先将所有的model_render经过网络映射成特征图
        pbar = tqdm.tqdm(data_test_model)
        with torch.no_grad():
            for meta in pbar:
                render_img = meta['query_or_model_img'].to(device=self.device)
                cat_label_model = meta['label_cat'].to(device=self.device)

                render_img = render_img.reshape(-1, self.channels, arg.img_size, arg.img_size).to(device=self.device)
                render_emb = self.net.get_render_emb(render_img)
                cat_label_model_list.append(cat_label_model)
                render_embs_list.append(render_emb)

        render_embs = torch.concat(render_embs_list, dim=0)
        cat_label_model_list = torch.concat(cat_label_model_list, dim=0).reshape(-1)
        model_lenth = render_embs.size(0)

        results = []
        query_labels = []
        query_embs = []
        pbar = tqdm.tqdm(data_test_query)
        with torch.no_grad():
            for meta in pbar:
                query_img = meta['sketch_img'].to(device=self.device)
                cat_label_query = meta['label_cat'].to(device=self.device)
                bs = cat_label_query.size(0)
                query_labels.append(cat_label_query)
                query_emb = self.net.get_sketch_emb(query_img)
                query_embs.append(query_emb)

                query_emb = query_emb.repeat_interleave(model_lenth, dim=0).reshape(1, model_lenth, arg.embding_size)
                render_embs = render_embs.reshape(1, model_lenth, arg.embding_size)
                similarity_score = F.cosine_similarity(query_emb, render_embs, dim=2)[0]
                sorted_similarity_score, sorted_index = torch.sort(similarity_score, descending=True)
                query_result_model_label = cat_label_model_list[sorted_index]
                query_label = cat_label_query.repeat(model_lenth)

                results.append((query_label == query_result_model_label))

        results = torch.stack(results, dim=0)
        # 保存结果数据到文件
        query_labels = torch.stack(query_labels, dim=0)
        query_embs = torch.stack(query_embs, dim=0)
        total_query = results.size(0)
        # 计算Precision-Recall曲线
        y_true = []
        y_scores = []

        # 遍历所有查询样本
        for i in range(total_query):
            true_label = query_labels[i].item()
            query_emb = query_embs[i]
            # 获取该查询样本的所有视图标签（模型标签）
            current_view_labels = cat_label_model_list.cpu().numpy()  # 需要将其移到CPU并转为numpy
            # 获取与当前查询样本相关的所有视图的相似度（相似度得分）
            query_emb = query_emb.repeat_interleave(model_lenth, dim=0).reshape(1, model_lenth, arg.embding_size)
            similarity_score = F.cosine_similarity(query_emb, render_embs, dim=2)[0]
            current_distances = similarity_score.cpu().numpy()  # 这里改为使用 similarity_score 而不是 results[i]

            # 创建二进制标签：相关为1，不相关为0
            binary_labels = (current_view_labels == true_label).astype(int)

            # 添加到列表
            y_true.extend(binary_labels)
            y_scores.extend(current_distances)

        # 将 y_true 和 y_scores 转换为 NumPy 数组，确保它们在 CPU 上
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # 计算 Precision-Recall 曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)

        # 绘制 PR 曲线
        # plt.figure(figsize=(8, 6))
        # plt.plot(recall, precision, color='r', lw=2, label=f'ours (AP = {average_precision:.4f})')
        return precision, recall,thresholds,average_precision
    def save(self):
        current_time = time.localtime()  # 将时间戳转换为本地时间
        # 格式化时间，提取月、日和小时
        formatted_time = time.strftime("%m-%d-%H", current_time)

        # 打印保存路径和文件名
        print(f"Saving model at {formatted_time}....")

        # 保存模型，使用格式化的时间作为文件名的一部分
        torch.save(self.net.state_dict(), arg.save_path)
        print('saving....')


def calculate_retrieval_scores(results, cls_counts, query_labels):
    FT = ST = 0
    for i, label in enumerate(query_labels):
        c = cls_counts[label].item()
        if c > 0:
            FT += results[i, :c].float().sum().item() / c
            ST += results[i, :2*c].float().sum().item() / c
    return FT / len(query_labels), ST / len(query_labels)

def calculate_dcg_and_map(results, device):
    NDCG = mAP = 0
    for i in range(results.size(0)):
        relevance = results[i].float().to(device)
        log_positions = torch.log2(torch.arange(2, relevance.size(0) + 1).float().to(device))
        if relevance.size(0) > 1:
            dcg_score = (relevance[1:] / log_positions).sum().item()
        else:
            dcg_score = 0
        dcg_score += relevance[0].item()  # Add the relevance of the first position

        ideal_relevance = torch.sort(relevance, descending=True)[0]
        if ideal_relevance.size(0) > 1:
            ideal_dcg_score = (ideal_relevance[1:] / log_positions).sum().item()
        else:
            ideal_dcg_score = 0
        ideal_dcg_score += ideal_relevance[0].item()

        if ideal_dcg_score > 0:
            ndcg_score = dcg_score / ideal_dcg_score
        else:
            ndcg_score = 0

        NDCG += ndcg_score

        positive_indices = relevance.nonzero(as_tuple=False).squeeze() + 1
        if positive_indices.numel() > 0:
            count_map = torch.arange(1, positive_indices.numel() + 1).float().to(device)
            mAP += (count_map / positive_indices.float()).mean().item()

    return NDCG / results.size(0), mAP / results.size(0)


def get_perclass_num(render: str) -> dict:
    per_class_dict = {}
    root = "../data"
    key_path = os.path.join(root, render)

    # 遍历每个子文件夹
    for class_name in os.listdir(key_path):
        class_path = os.path.join(key_path, class_name)
        if os.path.isdir(class_path):  # 确保是文件夹
            per_class_dict[class_name] = len(os.listdir(class_path))

    return per_class_dict
def maha(
        indist_train_embeds_in,
        indist_train_labels_in,
        subtract_mean=False,
        normalize_to_unity=False,
        indist_classes=arg.class_num,
):

    # storing the replication results
    maha_intermediate_dict = dict()

    description = ""

    all_train_mean = np.mean(indist_train_embeds_in, axis=0, keepdims=True)

    indist_train_embeds_in_touse = indist_train_embeds_in

    if subtract_mean:
        indist_train_embeds_in_touse -= all_train_mean
        description = description + " subtract mean,"

    if normalize_to_unity:
        indist_train_embeds_in_touse = indist_train_embeds_in_touse / np.linalg.norm(
            indist_train_embeds_in_touse,
            axis=1, keepdims=True)
        description = description + " unit norm,"

    # full train single fit
    mean = np.mean(indist_train_embeds_in_touse, axis=0)
    cov = np.cov((indist_train_embeds_in_touse - (mean.reshape([1, -1]))).T)

    eps = 1e-8
    cov_inv = np.linalg.inv(cov)

    # getting per class means and covariances
    class_means = []
    class_cov_invs = []
    class_covs = []
    for c in range(indist_classes):
        mean_now = np.mean(indist_train_embeds_in_touse[indist_train_labels_in == c], axis=0)

        cov_now = np.cov(
            (indist_train_embeds_in_touse[indist_train_labels_in == c] - (mean_now.reshape([1, -1]))).T)
        class_covs.append(cov_now)
        # print(c)

        eps = 1e-8
        cov_inv_now = np.linalg.inv(cov_now)

        class_cov_invs.append(cov_inv_now)
        class_means.append(mean_now)

    # the average covariance for class specific
    class_cov_invs = [np.linalg.inv(np.mean(np.stack(class_covs, axis=0), axis=0))] * len(class_covs)

    maha_intermediate_dict["class_cov_invs"] = class_cov_invs
    maha_intermediate_dict["class_means"] = class_means
    maha_intermediate_dict["cov_inv"] = cov_inv
    maha_intermediate_dict["mean"] = mean

    return maha_intermediate_dict

def get_relative_maha_distance(train_logits, cov_invs, class_cov_invs, means, class_means, targets,
                               norm_name="L2"):
    maha_0 = np.array([maha_distance(train_logits[i], cov_invs, means, norm_name) for i in range(len(targets))])
    maha_k = np.array([maha_distance(train_logits[i].reshape([1, -1]), class_cov_invs[targets[i]],
                                     class_means[targets[i]], norm_name) for i in range(len(targets))])
    # print(maha_0.shape,maha_k.shape)
    scores = maha_k - maha_0

    return scores



def maha_distance(xs, cov_inv_in, mean_in, norm_type=None):
    diffs = xs - mean_in.reshape([1, -1])
    #   print(cov_inv_in.shape,mean_in.shape,diffs.shape)

    second_powers = np.matmul(diffs, cov_inv_in) * diffs
    #   print(second_powers.shape)

    if norm_type in [None, "L2"]:
        return np.sum(second_powers, axis=1)
    elif norm_type in ["L1"]:
        return np.sum(np.sqrt(np.abs(second_powers)), axis=1)
    elif norm_type in ["Linfty"]:
        return np.max(second_powers, axis=1)
if __name__ == '__main__':
    per_class_dict = get_perclass_num(render='render')
    with open("per_model_num.json", "w") as json_file:
        json.dump(per_class_dict, json_file, indent=4)
    T = TrainCrossNet()
    # T.Train()
    # T.test_train()
    T.test()
    # T.PR_curve()
    # rate_results()
    # T.test_with_pr_curve()