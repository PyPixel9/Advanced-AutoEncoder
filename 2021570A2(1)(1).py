import torch
import torchaudio
import torchvision
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader , Subset
from torchvision.transforms import transforms
import numpy as np 
import math
import os
import os
import torch
import torchaudio
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import transforms
import numpy as np
import math
import random
from torchvision.transforms import ToTensor


def check_and_download_cifar10(root='./data', train=True):
    dataset_path = os.path.join(root, 'cifar-10-batches-py')
    if not os.path.exists(dataset_path):
        torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transforms.ToTensor())
        
check_and_download_cifar10(root='./data', train=True)
check_and_download_cifar10(root='./data', train=False)

def check_and_download_speechcommands(root='./data'):
    dataset_path = os.path.join(root, 'SpeechCommands', 'speech_commands_v0.02')
    if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
        torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True)

check_and_download_speechcommands(root='./data')

# ---------------------------------------------------------------------------------------------------------------------------------------
image_dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
image_dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

num_test = len(image_dataset_test)
test_indices = list(range(num_test))
test_split = int(np.floor(0.5 * num_test))

np.random.shuffle(test_indices)
testDataIndexes, validationDataIndexes = test_indices[test_split:], test_indices[:test_split]

image_test_sampler = Subset(image_dataset_test, testDataIndexes)
image_val_sampler = Subset(image_dataset_test, validationDataIndexes)

audio_dataset_train = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data',
    download=False,
    subset='training'
)

audio_dataset_val_test = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data',
    download=False,
    subset='testing'
)
# ---------------------------------------------------------------------------------------------------------------------------------------

class ImageDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        super().__init__()
        self.datasplit = split
        if split not in ["train", "test", "val"]:
            raise ValueError("Data split must be in ['train', 'test', 'val']")
        self.transform = transforms.ToTensor()
        if split == "train":
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=self.transform)

            # full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=self.transform)
            # subset_size = len(full_dataset) //300
            # subset_indices = random.sample(range(len(full_dataset)), subset_size)  # Randomly select indices
            # self.dataset = Subset(full_dataset, subset_indices)

        elif split == "val" :
            self.dataset = image_val_sampler
        else:
            self.dataset = image_test_sampler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target



class AudioDataset(Dataset):
    def __init__(self, root='./data', split='train'):
        super().__init__()
        self.datasplit = split
        if split not in ["train", "val", "test"]:
            raise ValueError("Data split must be in ['train', 'val', 'test']")
        self.dataset_path = os.path.join(root, 'SpeechCommands', 'speech_commands_v0.02')
        self.transform = ToTensor()

        self.classes = [
            "down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes",
            "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
            "tree", "wow", "zero", "one", "two", "three", "four", "five", "six",
            "seven", "eight", "nine"
        ]
        self.label_to_index = self.mapLabelToIndex()
        self._load_dataset()

    def _load_dataset(self):
        self.dataset = []
        subset_file = 'validation_list.txt' if self.datasplit == 'val' else 'testing_list.txt' if self.datasplit == 'test' else None
        if subset_file:
            with open(os.path.join(self.dataset_path, subset_file), 'r') as f:
                subset_filenames = [line.strip() for line in f.readlines()]
            for filename in subset_filenames:
                label = filename.split('/')[0]
                filepath = os.path.join(self.dataset_path, filename)
                if label in self.classes:
                    self.dataset.append(filepath)
        else: 
            all_files = []
            for label in self.classes:
                label_path = os.path.join(self.dataset_path, label)
                for filename in os.listdir(label_path):
                    filepath = os.path.join(label_path, filename)
                    if filepath not in self.dataset:
                        all_files.append(filepath)
            exclude_files = self.filenameFromList('validation_list.txt') + self.filenameFromList('testing_list.txt')
            self.dataset = [file for file in all_files if file not in exclude_files]
            
# 1-----
            # subset_size = len(self.dataset) // 600
            # subset_indices = random.sample(range(len(self.dataset)), subset_size)  # Randomly select indices
            # self.dataset = [self.dataset[i] for i in subset_indices]
# 1-----

    def filenameFromList(self, subset_file):
        with open(os.path.join(self.dataset_path, subset_file), 'r') as f:
            return [os.path.join(self.dataset_path, line.strip()) for line in f.readlines()]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        filepath = self.dataset[index]
        waveform, _ = torchaudio.load(filepath)
        label = filepath.split('/')[-2]
        label_index = self.label_to_index[label]
        transposed_waveform = torch.transpose(waveform, 0, 1).contiguous()
        current_length = transposed_waveform.size(0)
        if current_length < 16000:
            padding_size = 16000 - current_length
            padded_waveform = F.pad(transposed_waveform, (0, 0, 0, padding_size), "constant", 0)
        else:
            padded_waveform = transposed_waveform
        return padded_waveform, label_index

    def mapLabelToIndex(self):
        label_to_index = {label: index for index, label in enumerate(self.classes)}
        return label_to_index

# ---------------------------------------------------------------------------------------------------------------------
class ResnetBlock1D(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1):
        super(ResnetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels!=output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(output_channels)
            )

    def forward(self, x):
        if True: 
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.relu1(y)
            y = self.conv2(y)
            y = self.bn2(y)
            shortcut = self.shortcut(x)
            y += shortcut
            y = self.relu2(y)
        return y

class ResnetBlock2D(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResnetBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels,output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        if True: 
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            y += self.shortcut(x)
            y = F.relu(y)
        return y

class Resnet_Q1(nn.Module):
    def __init__(self):
        super(Resnet_Q1, self).__init__()
        self.input_channels_image = 64
        self.input_channels_audio = 8
        self.conv1_image = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_image = nn.BatchNorm2d(64)
        self.relu_image = nn.ReLU(inplace=True)
        self.maxpool_image = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1_audio = nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_audio = nn.BatchNorm1d(8)
        self.relu_audio = nn.ReLU(inplace=True)
        self.maxpool_audio = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1_image = self.FormLayerImage(ResnetBlock2D, 64, 2, stride=1)
        self.layer2_image = self.FormLayerImage(ResnetBlock2D, 128, 2, stride=2)
        self.layer3_image = self.FormLayerImage(ResnetBlock2D, 256, 2, stride=2)
        self.layer4_image = self.FormLayerImage(ResnetBlock2D, 512, 2, stride=2)
        self.avgpool_image = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_image = nn.Linear(512, 10)
        self.layer1_audio = self.FormLayerAudio(ResnetBlock1D, 8, num_blocks=4, stride=1)
        self.layer2_audio = self.FormLayerAudio(ResnetBlock1D, 16, num_blocks=4, stride=2)
        self.layer3_audio = self.FormLayerAudio(ResnetBlock1D, 32, num_blocks=5, stride=2)
        self.layer4_audio = self.FormLayerAudio(ResnetBlock1D, 64, num_blocks=5, stride=2)
        self.avgpool_audio = nn.AdaptiveAvgPool1d(1)
        self.fc_audio = nn.Linear(64, 35)

    def FormLayerImage(self, resnet2D, output_channels, num_blocks, stride, is_audio=False):
        layers = []
        input_channels = self.input_channels_audio if is_audio else self.input_channels_image
        for _ in range(num_blocks):
            layers.append(resnet2D(input_channels, output_channels, stride))
            input_channels = output_channels
        if is_audio:
            self.input_channels_audio = input_channels
        else:
            self.input_channels_image = input_channels
        return nn.Sequential(*layers)

    def FormLayerAudio(self, resnet1D, output_channels, num_blocks, stride):
        layers = []
        input_channels = self.input_channels_audio
        for _ in range(num_blocks):
            layers.append(resnet1D(input_channels, output_channels, stride))
            input_channels = output_channels
        self.input_channels_audio = input_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.conv1_image(x)
            x = self.bn1_image(x)
            x = self.relu_image(x)
            x = self.maxpool_image(x)
            x = self.layer1_image(x)
            x = self.layer2_image(x)
            x = self.layer3_image(x)
            x = self.layer4_image(x)
            x = self.avgpool_image(x)
            x = torch.flatten(x, 1)
            x = self.fc_image(x)
        else:
            x = x.transpose(1, 2)
            x = self.conv1_audio(x)
            x = self.bn1_audio(x)
            x = self.relu_audio(x)
            x = self.maxpool_audio(x)
            x = self.layer1_audio(x)
            x = self.layer2_audio(x)
            x = self.layer3_audio(x)
            x = self.layer4_audio(x)
            x = self.avgpool_audio(x)
            x = torch.flatten(x, 1)
            x = self.fc_audio(x)
        return x


class VGGBlock:
    def CombinedBlock(input_channels, output_channels, kernel_size, n_convolutions , type):
        if type == "1D": 
            layers = [
                nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True)
            ]
            for i in range(1, n_convolutions):
                layers.extend([
                    nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(inplace=True)])
            return nn.Sequential(*layers)
        else:
            layers = [
            nn.Conv2d(input_channels, 
                      output_channels, 
                      kernel_size=kernel_size, 
                      padding=kernel_size//2),
                  nn.BatchNorm2d(output_channels),
                  nn.ReLU(inplace=True)]
        
            for i in range(1, n_convolutions):
                layers.extend(
                    [
                        nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(inplace=True)
                    ]
                )
            return nn.Sequential(*layers)

    

    


class VGG_Q2(nn.Module):
    def __init__(self):
        super(VGG_Q2, self).__init__()

        self.features_AUDIO = nn.Sequential(
            VGGBlock.CombinedBlock(1, 64, 3, 2 , "1D"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            VGGBlock.CombinedBlock(64, 42, 4, 2, "1D"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            VGGBlock.CombinedBlock(42,28,5, 3, "1D"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            VGGBlock.CombinedBlock(28,18,6, 3, "1D"),
            nn.MaxPool1d(kernel_size=2, stride=2),
            VGGBlock.CombinedBlock(18,12,8, 3, "1D"),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_IMAGE = nn.Sequential(
            VGGBlock.CombinedBlock(3,64, 3, 2,"2D"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock.CombinedBlock(64,42, 4, 2,"2D"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock.CombinedBlock(42,28, 5, 3,"2D"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock.CombinedBlock(28,18, 6, 3,"2D"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock.CombinedBlock(18,12,8, 3,"2D"),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier_IMAGE = None

        self.classifier_AUDIO = self.classifier_AUDIO = nn.Sequential(
                        nn.Linear(6024, 256),
                        nn.ReLU(True),
                        nn.Linear(256, 128),
                        nn.ReLU(True),
                        nn.Linear(128, 35), 
                    )


    def forward(self, x):
        if x.shape[1] == 3:
            x = self.features_IMAGE(x)
            if self.classifier_IMAGE is None:
                with torch.no_grad():
                    self.num_flat_features = x.shape[1]*x.shape[2]*x.shape[3]
                    self.classifier_IMAGE = nn.Sequential(
                        nn.Linear(self.num_flat_features, 4096),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(4096, 4096),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(4096, 10),
                    ).to(x.device)
            x = torch.flatten(x, 1)
            x = self.classifier_IMAGE(x)
        else:
            x = x.transpose(1, 2)
            x = self.features_AUDIO(x)
            x = torch.flatten(x, 1)
            x = self.classifier_AUDIO(x)
        return x







    
class InceptionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dimension):
        super(InceptionBlock, self).__init__()
        if dimension == "2D":
            self.branch1 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            self.branch3 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        else:
            self.branch1 = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1),
                nn.BatchNorm1d(output_channels),
                nn.ReLU()
            )
            self.branch2 = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=3 , padding = 3//2),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(),
                nn.Conv1d(output_channels, output_channels, kernel_size=5, padding=5//2),
                nn.BatchNorm1d(output_channels),
                nn.ReLU()
            )
            self.branch3 = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=3 , padding=3//2),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(),
                nn.Conv1d(output_channels, output_channels, kernel_size=5, padding=5//2),
                nn.BatchNorm1d(output_channels),
                nn.ReLU()
            )
            self.branch4 = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=3//2)
            )

    def forward(self, x):
        if x.shape[1] == 3:
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            branch3 = self.branch3(x)
            branch4 = self.branch4(x)
            return torch.cat([branch1, branch2, branch3, branch4], 1)
        else:
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            branch3 = self.branch3(x)
            branch4 = self.branch4(x)
            return torch.cat([branch1, branch2, branch3, branch4], 1)

class Inception_Q3(nn.Module):
    def __init__(self):
        super(Inception_Q3, self).__init__()
        self.inception_blocks_image = nn.Sequential(
            InceptionBlock(3, 8 , "2D"),
            InceptionBlock(27, 12, "2D"),
            InceptionBlock(63, 24, "2D"),
            InceptionBlock(135, 87, "2D")
         )
        self.InceptionBlocks_A = nn.Sequential(
            InceptionBlock(1,2 ,"1D"),
            InceptionBlock(7,3,"1D"),
            InceptionBlock(16,4,"1D"),
            InceptionBlock(28,2,"1D"),
        )
        self.fc_image = nn.Linear(405504, 10)
        self.fc_audio = nn.Linear(544000, 35)
    
    def forward(self, x):
        if x.shape[1] == 3:
            x = self.inception_blocks_image(x)
            x = x.view(x.size(0), -1)
            x = self.fc_image(x)
        else:
            x = x.permute(0, 2, 1)            
            x = self.InceptionBlocks_A(x)
            x = x.view(x.size(0), -1)
            x = self.fc_audio(x)
        return x



class CustomNetwork_Q4(nn.Module):
    def __init__(self):
        super(CustomNetwork_Q4, self).__init__()
        self.RGB = 3
        self.res1_img = ResnetBlock2D(self.RGB, 3)
        self.res2_img = ResnetBlock2D(3, 3)
        self.inc1_img = InceptionBlock(3, 5, '2D')
        self.inc2_img = InceptionBlock(18, 10, '2D')
        self.res3_img = ResnetBlock2D(48, 48)
        self.inc3_img = InceptionBlock(48, 24, '2D')
        self.res4_img = ResnetBlock2D(120, 78)
        self.inc4_img = InceptionBlock(78, 60, '2D')
        self.res5_img = ResnetBlock2D(258, 140)
        self.inc5_img = InceptionBlock(140, 60, '2D')
        self.fc_img = nn.Linear(327680, 10)

        self.Mono = 1
        self.res1_audio = ResnetBlock1D(self.Mono, 1)
        self.res2_audio = ResnetBlock1D(1, 1)
        self.inc1_audio = InceptionBlock(1, 4, '1D')
        self.inc2_audio = InceptionBlock(13, 4, '1D')
        self.res3_audio = ResnetBlock1D(25, 7)
        self.inc3_audio = InceptionBlock(7, 4, '1D')
        self.res4_audio = ResnetBlock1D(19, 10)
        self.inc4_audio = InceptionBlock(10, 4, '1D')
        self.res5_audio = ResnetBlock1D(22, 13)
        self.inc5_audio = InceptionBlock(13, 4, '1D')
        self.fc_audio = nn.Linear(400000, 35)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.res1_img(x)
            x = self.res2_img(x)
            x = self.inc1_img(x)
            x = self.inc2_img(x)
            x = self.res3_img(x)
            x = self.inc3_img(x)
            x = self.res4_img(x)
            x = self.inc4_img(x)
            x = self.res5_img(x)
            x = self.inc5_img(x)
            x = x.view(x.size(0), -1)
            x = self.fc_img(x)
        else:
            x = x.permute(0, 2, 1) 
            x = self.res1_audio(x)
            x = self.res2_audio(x)
            x = self.inc1_audio(x)
            x = self.inc2_audio(x)
            x = self.res3_audio(x)
            x = self.inc3_audio(x)
            x = self.res4_audio(x)
            x = self.inc4_audio(x)
            x = self.res5_audio(x)
            x = self.inc5_audio(x)
            x = x.view(x.size(0), -1) 
            x = self.fc_audio(x)

        return x





# ------------------------------------------------------------------------------------------------------------------------------

def trainer(gpu="F", dataloader=None, network=None, criterion=None, optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    best_accuracy = 0
    for epoch in range(EPOCH):
        running_loss = float(0)
        correct_predictions = 0
        total_predictions = 0
        network.train()
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        print("Training Epoch: {}, [Loss: {:.4f}, Accuracy: {:.2f}]".format(epoch+1, epoch_loss, epoch_accuracy * 100))
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(network.state_dict(), 'best_checkpoint.pth')


def validator(gpu="F", dataloader=None, network=None, criterion=None, optimizer=None): 
    device = torch.device("cuda:0" if gpu == "T" else "cpu")
    network.to(device)
    network.load_state_dict(torch.load("best_checkpoint.pth"))

    for epoch in range(1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        n_epoch = epoch + 1
        loss_epoch = epoch_loss 
        t_accuracy = (correct / total) * 100
        print("Validation Epoch: {}, [Loss: {:.4f}, Accuracy: {:.2f}]".format(n_epoch, loss_epoch, t_accuracy))
    torch.save(network.state_dict(), "best_checkpoint.pth")


def evaluator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    
    device = torch.device("cuda:0" if gpu == "T" else "cpu")

    network = network.to(device)

    network.load_state_dict(torch.load("best_checkpoint.pth", map_location=device))

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(1):
        loss_epoch = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss_epoch += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        t_accuracy = (correct / total) * 100
        print("[Loss: {:.4f}, Accuracy: {:.2f}]".format(loss_epoch, t_accuracy))

        
        
        

