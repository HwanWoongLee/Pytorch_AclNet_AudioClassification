import os
import glob
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SoundDataset(Dataset):
    def __init__(self, data_path, transform=None, target_lengh=20480) -> None:
        super().__init__()
        self.data_path = data_path
        self.file_path = []
        file_list = glob.glob(data_path + "/**/*", recursive=True)
        for file in file_list:
            if os.path.isfile(file):
                self.file_path.append(file)
        
        self.transform = transform
        self.target_lengh = target_lengh
        
    def __len__(self):
        return self.file_path.__len__()
        
    def __getitem__(self, index):
        # librosa를 이용해 file_path의 오디오 파일을 읽고
        y, sr = librosa.load(self.file_path[index], sr=16000)
        
        y = torch.tensor(y, dtype=torch.float32)

        # 길이 조절절
        if len(y) > self.target_lengh:
            y = y[:self.target_lengh]
        else:
            pad_length = self.target_lengh - len(y)
            y = F.pad(y, (0, pad_length))

        # 차원 추가
        y = torch.reshape(y, (1, -1))

        # file_path에서 labeling ( './{path}/{XX_filename}.wav' -> int(XX) )
        label = int(self.file_path[index].replace(self.data_path, "", 1)[1:4]) - 1
        
        return y, label