import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, data_dir,
                 transform=transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.ToTensor()])):

        self.data_dir = data_dir
        self.class_set = sorted(os.listdir(data_dir))
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.class_encode = {cls: i for i, cls in enumerate(self.class_set)}

        for cls in self.class_set:
            dir = os.path.join(data_dir, cls)
            for img in os.listdir(dir):
                self.image_paths.append(os.path.join(dir, img))
                self.labels.append(self.class_encode[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, img_label
