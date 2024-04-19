import torch
from torchvision import transforms

from torchmeta.datasets import MiniImagenet, TieredImagenet, CUB, CIFARFS
from torchmeta.transforms import ClassSplitter, Categorical

DATA_PATH = '/data2/jjlee_datasets/torchmeta_datasets/'


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.
    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'


def resize_transform(resize_size, normalize, status):
    if status == 'train':
        transform = transforms.Compose([
            transforms.Resize([resize_size, resize_size]),
            transforms.ToTensor(),
            # normalize,
            
            # transforms.Resize([int(resize_size * 1.5), int(resize_size * 1.5)]),
            # transforms.RandomCrop(resize_size, padding=8),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([resize_size, resize_size]),
            transforms.ToTensor(),
            # normalize
        ])
    return transform


def get_meta_dataset(args, dataset, only_test=False, method='maml'):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shots,
                                      num_test_per_class=args.num_shots_test)
    dataset_transform_test = ClassSplitter(shuffle=True,
                                           num_train_per_class=args.num_shots,
                                           num_test_per_class=args.num_shots_test)

    if method == 'proto':
        train_num_ways = args.num_ways_proto
    else:
        train_num_ways = args.num_ways

    if dataset == 'miniimagenet':
        
        mean, std = [0.4721, 0.4533, 0.4099], [0.2771, 0.2677, 0.2844]
        image_size = 84
        normalize = transforms.Normalize(mean=mean,  std=std)
        
        transform = resize_transform(image_size, normalize, 'train')
        test_transform = resize_transform(image_size, normalize, 'test')
        
        meta_train_dataset = MiniImagenet(DATA_PATH,
                                          transform=transform,
                                          target_transform=Categorical(train_num_ways),
                                          num_classes_per_task=train_num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(DATA_PATH,
                                        transform=test_transform,
                                        target_transform=Categorical(args.num_ways),
                                        num_classes_per_task=args.num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform_test)
        meta_test_dataset = MiniImagenet(DATA_PATH,
                                         transform=test_transform,
                                         target_transform=Categorical(args.num_ways),
                                         num_classes_per_task=args.num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform_test)

    elif dataset == 'tieredimagenet':
        
        mean, std = [0.4721, 0.4533, 0.4099], [0.2771, 0.2677, 0.2844]
        image_size = 84
        normalize = transforms.Normalize(mean=mean,  std=std)
        
        transform = resize_transform(image_size, normalize, 'train')
        test_transform = resize_transform(image_size, normalize, 'test')

        meta_train_dataset = TieredImagenet(DATA_PATH,
                                            transform=transform,
                                            target_transform=Categorical(train_num_ways),
                                            num_classes_per_task=train_num_ways,
                                            meta_train=True,
                                            dataset_transform=dataset_transform,
                                            download=True)
        meta_val_dataset = TieredImagenet(DATA_PATH,
                                          transform = test_transform,
                                          target_transform=Categorical(args.num_ways),
                                          num_classes_per_task=args.num_ways,
                                          meta_val=True,
                                          dataset_transform=dataset_transform_test)
        meta_test_dataset = TieredImagenet(DATA_PATH,
                                           transform = test_transform,
                                           target_transform=Categorical(args.num_ways),
                                           num_classes_per_task=args.num_ways,
                                           meta_test=True,
                                           dataset_transform=dataset_transform_test)
    
    elif dataset == 'CIFAR_FS':
        mean, std= [0.5074, 0.4867, 0.4411], [0.2675, 0.2566, 0.2763]

        normalize = transforms.Normalize(mean=mean, std=std)
        
        image_size = 32
        
        transform = resize_transform(image_size, normalize, 'train')
        test_transform = resize_transform(image_size, normalize, 'test')

        meta_train_dataset = CIFARFS(DATA_PATH,
                                    transform= transform ,
                                    target_transform=Categorical(train_num_ways),
                                    num_classes_per_task=train_num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)
        meta_val_dataset = CIFARFS(DATA_PATH,
                                    transform=test_transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_val=True,
                                    dataset_transform=dataset_transform_test)
        meta_test_dataset = CIFARFS(DATA_PATH,
                                    transform=test_transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_test=True,
                                    dataset_transform=dataset_transform_test)


    elif dataset == 'CUB':
        
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        image_size = 84
        
        transform = resize_transform(image_size, normalize, 'train')
        test_transform = resize_transform(image_size, normalize, 'test')

        meta_train_dataset = CUB(DATA_PATH,
                                    transform=transform,
                                    target_transform=Categorical(train_num_ways),
                                    num_classes_per_task=train_num_ways,
                                    meta_train=True,
                                    dataset_transform=dataset_transform,
                                    download=True)
        meta_val_dataset = CUB(DATA_PATH,
                                    transform=test_transform,
                                    target_transform=Categorical(args.num_ways),
                                    num_classes_per_task=args.num_ways,
                                    meta_val=True,
                                    dataset_transform=dataset_transform_test)
        
        meta_test_dataset = CUB(DATA_PATH,
                                transform=test_transform,
                                target_transform=Categorical(args.num_ways),
                                num_classes_per_task=args.num_ways,
                                meta_test=True,
                                dataset_transform=dataset_transform_test)

   
    else:
        raise NotImplementedError()

    if only_test:
        return meta_test_dataset

    return meta_train_dataset, meta_val_dataset
