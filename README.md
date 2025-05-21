# Three-Dimensional Time Distributed Convolutional Neural Network-Long short-term memory (TD-CNNLSTM-LungNet)

This repository contains the official implementation of the **Three-Dimensional Time-Distributed Convolutional Neural Network–Long Short-Term Memory (TD-CNNLSTM-LungNet)** model, developed for the classification of pulmonary abnormalities using ultrasound video sequences. The proposed framework integrates a Time-Distributed Convolutional Neural Network (TD-CNN) with a Long Short-Term Memory (LSTM) architecture to effectively capture both spatial and temporal patterns. This integration enhances classification accuracy while improving the interpretability of the model's decision-making process.

## 🔬 Overview

TD-CNNLSTM-LungNet integrates Time-Distributed Convolutional Neural Networks with Long Short-Term Memory units to effectively capture spatial representations and temporal dynamics from ultrasound video sequences, enabling robust classification of pulmonary abnormalities.
<p align="center">
  <img src="https://github.com/mak-raiaan/3DTD-CNNLSTM-LungNet/blob/e2cd4827f9b581442a46f2d66e7c831d8a198237/Main%20Diagra%203DTD-CNNLSTM-LungNet.png" alt="Lung Model Architecture" width="1000"/>
</p>

## 📁 Repository Structure

- `asppst.py`: Main model architecture and implementation.  
  🔗 [View Code](https://github.com/mak-raiaan/3DTD-CNNLSTM-LungNet/blob/e2cd4827f9b581442a46f2d66e7c831d8a198237/3DTD-CNNLSTM-LungNet.py)

## 📊 Dataset

The model is trained and evaluated on the **COVID-US** dataset, a comprehensive dataset of endoscopic images and videos from the GI tract.

🔗 [Access COVID-US Dataset](https://github.com/nrc-cnrc/COVID-US)


## Citation

If you find this work useful for your research, please cite our paper:
```bibtex
@article{abian2024automated,
  title={Automated diagnosis of respiratory diseases from lung ultrasound videos ensuring XAI: an innovative hybrid model approach},
  author={Abian, Arefin Ittesafun and Khan Raiaan, Mohaimenul Azam and Karim, Asif and Azam, Sami and Fahad, Nur Mohammad and Shafiabady, Niusha and Yeo, Kheng Cher and De Boer, Friso},
  journal={Frontiers in Computer Science},
  volume={6},
  pages={1438126},
  year={2024},
  publisher={Frontiers Media SA}
}
