# AclNet (Audio Classiciation)
Pytorch를 이용하여 오디오 분류 모델인 AclNet을 구현. <br/>
10가지 환경음 Dataset인 ESC10를 Classification.
- ESC10 Dataset : https://github.com/karolpiczak/ESC-50
- 논문 원본 : https://arxiv.org/abs/1811.06669

<br/>

## 개발 환경
- Python : 3.11.11
- Pytorch : 2.6.0+cu118
- librosa : 0.11.0
- Dataset : ESC10 400개

<br/>

## 결과
- Train Dataset 정확도 : 98.75%
- Test Dataset 정확도 : 83.75%
- 논문에서 제시하는 정확도 결과와 유사
- [jupyter notebook](https://github.com/HwanWoongLee/Pytorch_AclNet_AudioClassification/blob/main/AclNet_test.ipynb)

<br/>


### 결과 산점도
![image](https://github.com/user-attachments/assets/e5ce9c59-a168-47c9-8b40-6a7a52bed448)

<br/>

### 결과 Confusion Matrix
![image](https://github.com/user-attachments/assets/6e06e2cd-f3df-45a0-8383-31e24f709393)
