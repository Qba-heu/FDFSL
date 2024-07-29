import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
# import h5py
# import imageio
# import umap
from sklearn.manifold import TSNE
import os


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

from operator import truediv
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



import torch.utils.data as data


class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()
        #        if medicinal==4 and d==2:
        #            self.train_data=self.imdb['data'][self.x1,:]
        #            self.train_labels=self.imdb['Labels'][self.x1]
        #            self.test_data=self.imdb['data'][self.x2,:]
        #            self.test_labels=self.imdb['Labels'][self.x2]

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  ##(17, 17, 200, 10249)
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 100:
            all_good[class_] = all_set[class_][:100]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    # image_data = imageio.imread(image_file)
    label_data = sio.loadmat(label_file)



    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]

    if data_key =="PaviaU":
    #   # dic-> narray , KSC:ndarray(512,217,204)
        data_all = image_data['paviaU']
        GroundTruth = label_data['paviaU_gt']
    elif data_key == 'Indian_pines_corrected':
        data_all = image_data['indian_pines_corrected']
        GroundTruth = label_data['indian_pines_gt']
    elif data_key == 'Salinas_corrected':
        data_all = image_data['salinas_corrected']
        GroundTruth = label_data['salinas_gt']
    elif data_key == 'Pavia':
        data_all = image_data['pavia']
        GroundTruth = label_data['pavia_gt']
    elif data_key == 'data_hsi':
        data_all = image_data['data']
        label_data2 = sio.loadmat('datasets/YRE/mask_train.mat')

        GroundTruth1 = label_data['mask_test']
        GroundTruth2 = label_data2['mask_train']
        GroundTruth = GroundTruth1+GroundTruth2
        w_sd,h_sd = GroundTruth1.shape
        GroundTruth[(GroundTruth == 20)[:w_sd, :h_sd]] = 0
        GroundTruth[(GroundTruth == 19)[:w_sd, :h_sd]] = 0
        for i in range(np.max(GroundTruth) - 1):
            count_class = np.copy(GroundTruth)
            # test_count = np.copy(test_gt)
            # sparse_class=np.copy(sparse_ground_truth)

            count_class[(GroundTruth != i + 1)] = 0
            # sparse_class[(sparse_ground_truth != i + 1)[:H_SD, :W_SD]] = 0
            class_num = np.count_nonzero(count_class)

            # test_count[test_gt != i + 1] = 0

            print([i + 1], ':', class_num)
            # print(i + 1, ':', class_num, np.count_nonzero(test_count))

        print('train_gt size is', np.count_nonzero(GroundTruth))
        # GroundTruth = label_data['mask_test']
        # GroundTruth = label_data['mask_test']
    else:
        raise ValueError("Load error, the {} dataset is unknow.".format(data_key))


    # data_all = image_data#['DataCube2']
    # H, W, B = data_all1.shape
    # data_all = data_all1[:(H // 2), :(W // 2), :]
    # data_all = image_data['salinas_corrected']
    # GroundTruth= label_data['indian_pines_gt']
    # GroundTruth= label_data['map']
    # GroundTruth = label_data['paviaU_gt']
    # GroundTruth = gt1[:(H // 2), :(W // 2)]
    # GroundTruth = label_data['salinas_gt']
    # w, h = gt1.shape
    # GroundTruth = np.zeros_like(gt1)
    # GroundTruth[(gt1 == 2)[:w, :h]] = 1
    # GroundTruth[(gt1 == 5)[:w, :h]] = 2
    # GroundTruth[(gt1 == 7)[:w, :h]] = 3
    # GroundTruth[(gt1 == 4)[:w, :h]] = 4
    # GroundTruth[(gt1 == 6)[:w, :h]] = 5
    # GroundTruth[(gt1 == 3)[:w, :h]] = 6
    # GroundTruth[(gt1 == 9)[:w, :h]] = 7

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)
    # print('tar_class_num:',np.max(GroundTruth)+1,'tar_samples_num:',np.count_nonzero(GroundTruth))
    # quit()

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data

class Task(object):

    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))

        class_list = random.sample(class_folders, self.num_classes)

        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]
            # print(self.support_labels)
            # print(self.query_labels)

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

# Sampler
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

# dataloader
def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和querya
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task,split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader



def color_SNE(dataset_src, label_tar):

    # label_dis = np.copy(label_tar)
    data_sne_src = np.copy(dataset_src)
    label_data_Src = np.copy(label_tar)
    X_tsne_src = umap.UMAP(n_components=2, random_state=100).fit_transform(data_sne_src)
    colors = [
        '#FF0000','#FF4500','#D2691E','#DEB887','#FFA500','#FFD700',
        '#808080','#9656DB','#7CFC00','#008000','#00FF7F','#00FFFF',
        '#1E90FF','#0000FF','#7B68EE','#FF1493'
    ]
    plt.figure(figsize=(14, 14))

    for cla_label in range(0, np.max(label_data_Src) + 1):
        plt.scatter(X_tsne_src[label_data_Src == cla_label][:, 0], X_tsne_src[label_data_Src == cla_label][:, 1],
                    c=colors[cla_label - 1], marker='o', s=1.5)
        print('class',cla_label,np.count_nonzero(X_tsne_src[label_data_Src == cla_label][:,0]),'\n')

    # scatter1 = ax.scatter(X_tsne_src[:, 0], X_tsne_src[:, 1], c=label_data_Src, marker='o', s=1.5)
    plt.legend()
    plt.savefig('feat' + str(label_tar.shape[0]) + '.png', dpi=300)
    # plt.show()

    # data_sne_src = np.copy(dataset_src)
    # # img_src = np.load('houston2013_ssvit.npy')
    # # data_sne_src = img_src[(gtSD != 0 ), :]
    # # data_sne_src = np.load('fea_tsne_UPSalinas.npy')
    # # label_data_Src = gt_src[(gtSD!=0)]
    # # label_data_tar = prediction[(gtSD!=0)]
    # # label_data_Src = gtSD[(gt_tar != 0)]
    #
    # # data_sne_tar = img_tar[(gtSD != 0), :]
    # # label_data_tar = gt_tar[(label_tar != 0)]
    # # fea_sne_src = fea_src[(gt_src != 0), :]
    # # fea_sne_tar = fea_tar[(gt_tar != 0), :]
    # # data_sne_src = img_src[(gtSD !=0), :]
    # # data_sne_tar = img_tar[(gt_tar !=0), :]
    # # fea_sne_src = fea_src[(gtSD !=0), :]
    # # fea_sne_tar = fea_tar[(gt_tar == class_show), :]
    # # standard_data_src = preprocessing.StandardScaler().fit_transform(data_sne_src)
    # # standard_data_tar = preprocessing.StandardScaler().fit_transform(data_sne_tar)
    # #
    # # standard_fea_src = preprocessing.RobustScaler().fit_transform(fea_sne_src)
    # # standard_fea_tar = preprocessing.RobustScaler().fit_transform(fea_sne_tar)
    #
    # # inpu_Data_src = standard_data_src[(label_data_Src==class_show),:]
    # # inpu_Data_tar = standard_data_tar[(label_data_tar == class_show), :]
    #
    #
    #
    #
    # # vis_data = dataset
    # # digits = vis_data
    # X_tsne_src = umap.UMAP(n_components=2, random_state=100).fit_transform(data_sne_src)
    # # X_tsne_tar = umap.UMAP(n_components=2, random_state=100).fit_transform(data_sne_tar)
    #
    # # Fea_tsne_src = umap.UMAP(n_components=2, random_state=100).fit_transform(data_sne_tar)
    # # Fea_tsne_tar = umap.UMAP(n_components=2, random_state=100).fit_transform(fea_sne_tar)
    #
    # # X_tsne_src = TSNE(n_components=2, random_state=100).fit_transform(data_sne_src)
    # # X_tsne_tar = TSNE(n_components=2, random_state=100).fit_transform(data_sne_tar)
    #
    # # preb_src = KMeans(n_clusters=1,random_state=50).fit(X_tsne_src).cluster_centers_
    # # preb_tar = KMeans(n_clusters=1, random_state=50).fit(X_tsne_tar).cluster_centers_
    #
    # ckpt_dir = "TSE_images"
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    #
    #     # ax1 = plt.axes()
    # plt.figure(figsize=(28,28))
    # ax = plt.subplot(121, aspect='equal')
    # # ax2 = plt.subplot(122, aspect='equal')
    # # ax.set_facecolor('lightgray')
    # # ax.set_title('Before UDA')
    # # ax2.set_facecolor('lightgray')
    # # ax2.set_title('After UDA')
    # # plt.axis(`'off')
    #
    # # ax.scatter(X_tsne_src[:, 0], X_tsne_src[:, 1], c=label_data_Src, cmap=plt.cm.get_cmap('jet',10), marker='o', s=1.5)
    # # ax.scatter(X_tsne_tar[:, 0], X_tsne_tar[:, 1], c=label_data_tar, cmap=plt.cm.get_cmap('jet',10),marker='o', s=1.5)
    #
    # # ax.scatter(Fea_tsne_src[:, 0], Fea_tsne_src[:, 1], c=label_data_Src, cmap=plt.cm.get_cmap('jet',10), marker='o', s=1.5)
    # # ax.scatter(Fea_tsne_tar[:, 0], Fea_tsne_tar[:, 1], c=label_data_tar, cmap=plt.cm.get_cmap('jet',10), marker='o', s=1.5)
    #
    # # ax.scatter(X_tsne_src[:, 0], X_tsne_src[:, 1], c='aqua', label="Src-C" + str_class, marker='o', s=1.5)
    # # plt.scatter(preb_src[:, 0], preb_src[:, 1], cmap='b', label="center_src", marker='o', s=25)
    # # ax.scatter(X_tsne_tar[:, 0], X_tsne_tar[:, 1], c='red', label="Tar-C" + str_class, marker='o', s=1.5)
    # # plt.scatter(preb_tar[:, 0], preb_tar[:, 1], c='r', label="center_tar", marker='o', s=25)
    #
    # # scatter1 = ax2.scatter(Fea_tsne_src[:, 0], Fea_tsne_src[:, 1], c=label_data_tar, cmap='tab20', marker='o', s=1.5)
    # scatter1 = ax.scatter(X_tsne_src[:, 0], X_tsne_src[:, 1], c=label_tar, cmap='tab20', marker='o', s=1.5)
    # legend1 = ax.legend(*scatter1.legend_elements(),title = 'classes')
    # # ax2.add_artist(legend1)
    # # ax2.axes.xaxis.set_visible([])
    # # ax2.axes.yaxis.set_visible([])
    # ax.axes.xaxis.set_visible([])
    # ax.axes.yaxis.set_visible([])
    # ax.axis('equal')
    # # ax.set_xlim(-20, 20)
    # # ax.set_ylim(-20, 20)
    #
    # # legend2 = ax.legend(*scatter2.legend_elements(), title='classes')
    # ax.add_artist(legend1)
    #
    #
    # # a,b = scatter1.legend_elements()
    #
    #
    #
    # # plt.legend()
    # plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    #
    # plt.grid()
    # plt.savefig('feat'+str(label_tar.shape[0])+'.png',dpi=300)
    #
    # plt.show()


def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0


def CORAL(source, target):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss

class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss