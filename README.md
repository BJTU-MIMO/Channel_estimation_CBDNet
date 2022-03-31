# Channel_estimation_CBDNet

This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] Y. Jin, J. Zhang, B. Ai, and X. Zhang, “Channel estimation for mmWave massive MIMO with convolutional blind denoising network,” IEEE Commun. Lett., vol. 24, no. 1, pp. 95–98, Jan. 2019.

*********************************************************************************************************************************
If you use this simulation code package in any way, please cite the original paper [1] above. 

Reference: We highly respect reproducible research, so we try to provide the simulation codes for our published papers. 
*********************************************************************************************************************************

## Abstract of the paper: 

Channel estimation is one of the foremost challenges for realizing practical millimeter-wave (mmWave) massive multiple-input multiple-output (MIMO) systems. To circumvent this problem, deep convolutional neural networks (CNNs) have been recently employed to achieve impressive success. However, current deep CNNs based channel estimators are only suitable to a small range of signal-to-noise ratios (SNRs). Unlike the existing works, the modified convolutional blind denoising net- work (CBDNet) is proposed to improve the robustness for noisy channel by adopting noise level estimation subnetwork, non-blind denosing subnetwork, and asymmetric joint loss functions for blind channel estimation. Furthermore, the CBDNet can adjust the estimated noise level map to interactively reduce the noise in the channel matrix. Numerical results demonstrate that the proposed CBDNet-based channel estimator can outperform the traditional channel estimators, traditional compressive sensing techniques and deep CNNs in terms of the normalized mean squared error. In addition, the CBDNet can be used over a large range of SNRs, which hugely reduce the cost of offline training.

## Content of Code Package

The package generates the simulation SE results which are used in Figure 6, Figure 7, Figure 8, and Figure 9. To be specific:

- `CBDNet.py`: Main function;
- `models.py`: Generate the CBDNet Network;
- `mmWave_channel_generate.m`: Generate the mmWave channel for training, verifying and testing;
- `functional.py`: the used function in the project;

See each file for further documentation.

## License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

Enjoy the reproducible research!




