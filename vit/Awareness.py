import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


class EdgeDevice:
    def __init__(self, id, args):
        self.device_id = id
        self.args = args
        self.dataset = None   
        self.selected_dataset = None   
        self.val = None
        self.model = None
        self.O = 0
        self.P = 0
        self.F = None

        self.sampler_train = None
        self.sampler_val = None

    def set_resource(self, Random=True ,O=0.8, P=0.8):
        if Random:
            self.O = np.random.uniform(0.5, 1.0)
            self.P = np.random.uniform(0.5, 1.0)
        else:
            self.O = O
            self.P = P
    
    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def Calculate_data_distribution(self):
        # 计算SIFT特征描述符
        print("Calculating SIFT descriptors\n")
        descriptors = compute_sift_descriptors_from_dataloader(self.dataset)

        # 选择最佳的混合高斯分布
        print("Selecting best GMM\n")
        self.F, bic_scores = select_best_gmm(descriptors)

    def print_resource(self):
        print(f'Edge device {self.device_id} has {self.O} GFLOPs and {self.P} parameters')

    def print_data_distribution(self):
        print(f'Edge device {self.device_id} has data distribution:\n')
        # 输出最佳混合高斯分布参数
        print_gmm_parameters(self.F)

    def get_resource(self):
        return self.O, self.P
    
    def get_data_distribution(self):
        return self.F
    
    def set_model(self, model):
        self.model = model

    def set_selected_dataset(self, public_dataset):
        gmm = self.F
        new_dataset = select_images_gmm(public_dataset, gmm)

        print("A new dataset has been created containing the number of images:", len(new_dataset))
        # 将新数据集转换为DataLoader
        new_data_loader = DataLoader(new_dataset, 
                                     sampler=self.sampler_train, 
                                     batch_size=self.args.batch_size, 
                                     num_workers=self.args.num_workers, 
                                     pin_memory=self.args.pin_mem, 
                                     drop_last=True)
        
        self.selected_dataset = new_data_loader

    def set_sampler(self, sampler_train, sampler_val):
        self.sampler_train = sampler_train
        self.sampler_val = sampler_val



# 计算SIFT特征描述符
def compute_sift_descriptors_from_dataloader(dataloader):
    sift = cv2.SIFT_create()
    descriptors_list = []

    for images, _ in dataloader:
        # 将Tensor移到CPU并转换为NumPy数组
        images = images.cpu().numpy()
        
        for image in images:
            if image is None:
                print("Image is None\n")
                continue
            
            # 将CHW格式转换为HWC格式，并缩放为uint8类型
            image_np = (image * 255).astype(np.uint8).transpose(1, 2, 0)
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            keypoints, descriptors = sift.detectAndCompute(image_gray, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)

    if descriptors_list:
        all_descriptors = np.vstack(descriptors_list)
        print("all_descriptors:", all_descriptors.shape)
        pca = PCA(n_components=16) # 使用PCA降维128->16
        reduced_descriptors = pca.fit_transform(all_descriptors)
        return reduced_descriptors
    else:
        return np.array([]) # 如果没有找到描述符，则返回两个空数组


# 使用BIC选择最佳高斯分布数量的函数 80
def select_best_gmm(descriptors, max_components=80):
    best_gmm = None
    lowest_bic = np.inf
    bic_scores = []
    num = 0

    for n in range(80, max_components + 1):
        print("GMM with", n, "components\n")
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(descriptors)
        bic = gmm.bic(descriptors)
        bic_scores.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
            num = n
    print("Best GMM has", num, "components\n")
    return best_gmm, bic_scores


# 输出混合高斯分布参数的函数
def print_gmm_parameters(gmm):
    print("Weights:\n", gmm.weights_)
    print("Means:\n", gmm.means_)
    print("Covariances:\n", gmm.covariances_)


def select_images_gmm(images, gmm, num_samples=200):
    sift = cv2.SIFT_create()
    pca = PCA(n_components=16)
    selected_images = []  
    print("Selecting images\n")

    for images, _ in images:
        # 将Tensor移到CPU并转换为NumPy数组
        images = images.cpu().numpy()    
        for image in images:
            if len(selected_images) >= num_samples:
                break
            if image is None:
                print("Image is None\n")
                continue          
            # 将CHW格式转换为HWC格式，并缩放为uint8类型
            image_np = (image * 255).astype(np.uint8).transpose(1, 2, 0)
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            keypoints, descriptors = sift.detectAndCompute(image_gray, None)
            if descriptors is not None:
                if (len(descriptors) < 16):
                    continue
                pca.fit(descriptors)
                descriptors = pca.transform(descriptors)
                image_log_prob = 0
                for desc in descriptors:                    
                    # 计算描述符的对数概率
                    log_prob = gmm.score_samples(desc.reshape(1, -1))
                    image_log_prob += log_prob
                image_log_prob /= len(descriptors) 
                    
                # data=open("../select_descriptors.txt",'a')
                # data.write(f"Log Probability: {image_log_prob} and threshold value: {np.log(0.1)}\n")  # 调试输出
                # data.close()

                if image_log_prob > np.log(0.1)-90:  # 设定一个阈值来选择描述符
                    selected_images.append(image)
      
    return selected_images

