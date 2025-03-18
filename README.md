# Evidential Domain Adaptation for Remaining Useful Life Prediction with Incomplete Degradation [[Paper](https://ieeexplore.ieee.org/document/10930593)]
#### *by: Yubo Hou, Mohamed Ragab, Yucheng Wang, Min Wu, Chee-Keong Kwoh, Xiaoli Li, Zhenghua Chen*
#### IEEE Transactions on Instrumentation and Measurement (TIM-25).

This is a PyTorch implementation of this domain adaptation method for remaining useful Life prediction on time series data.

## Abstract
Accurate Remaining Useful Life (RUL) prediction without labeled target domain data is a critical challenge, and domain adaptation (DA) has been widely adopted to address it by transferring knowledge from a labeled source domain to an unlabeled target domain. Despite its success, existing DA methods struggle significantly when faced with incomplete degradation trajectories in the target domain, particularly due to the absence of late degradation stages. This missing data introduces a key extrapolation challenge. When applied to such incomplete RUL prediction tasks, current DA methods encounter two primary limitations. First, most DA approaches primarily focus on global alignment, which can misaligns late degradation stage in the source domain with early degradation stage in the target domain. Second, due to varying operating conditions in RUL prediction, degradation patterns may differ even within the same degradation stage, resulting in different learned features. As a result, even if degradation stages are partially aligned, simple feature matching cannot fully align two domains. To overcome these limitations, we propose a novel evidential adaptation approach called EviAdapt, which leverages evidential learning to enhance domain adaptation. The method first segments the source and target domain data into distinct degradation stages based on degradation rate, enabling stage-wise alignment that ensures samples from corresponding stages are accurately matched. To address the second limitation, we introduce an evidential uncertainty alignment technique that estimates uncertainty using evidential learning and aligns the uncertainty across matched stages. The effectiveness of EviAdapt is validated through extensive experiments on the C-MAPSS, N-CMAPSS and PHM2010 datasets. Results show that our approach significantly outperforms state-of-the-art methods, demonstrating its potential for tackling incomplete degradation scenarios in RUL prediction.

## Requirmenets:
- Python3.x
- Pytorch==1.12.1
- Numpy
- Sklearn
- Pandas
  
## Citation
If you found this work useful for you, please consider citing it.
```
@ARTICLE{10930593,
  author={Hou, Yubo and Ragab, Mohamed and Wang, Yucheng and Wu, Min and Alseiari, Abdulla and Kwoh, Chee-Keong and Li, Xiaoli and Chen, Zhenghua},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Evidential Domain Adaptation for Remaining Useful Life Prediction with Incomplete Degradation}, 
  year={2025},
}
```
