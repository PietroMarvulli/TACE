# Predict TACE response in HCC patient from CT images using ML and DL mondels
This project aims to develop a model that predicts the response to TACE treatment from CT images in patients with Hepatocellular Carcinoma.
Through a literature study we have seen all the methods used and the data on which the predictive models were based so far implemented.
To analyse which is the best all methods have been tested on a public dataset available on the [TCIA](https://www.cancerimagingarchive.net/) called [HCC-TACE-SEG](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/)
## State Of The Art
| Authors                           | Image Type  |  N  | Best Model                   |   Clinical   |   Radiomics   |   Deep   |
|-----------------------------------|-------------|-----|--------------------------------|-------------|--------------|-----------|
| [Fateema et al.](https://doi.org/10.1109/ICEEICT62016.2024.10534501) | CT          |  69 | Modified AlexNet              |               |               |    ✔    |
| [Zhang et al.](https://doi.org/10.1007/s11547-024-01785-z)           | AP & VP CT  | 367 | RF                             |     ✔       |     ✔        |         |
| [Zhang et al.](https://doi.org/10.3389/fphar.2024.1315732)           | AP & VP CT  | 108 | LASSO LR                       |              |     ✔        |         |
| [Wang et al.](https://doi.org/10.1016/j.acra.2023.08.001)            | AP & VP CT  | 243 | XGBoost                        |     ✔       |     ✔        |         |
| [Sun et al.](https://doi.org/10.2147/JHC.S443660)                    | AP CT       | 399 | ResNet18 + LR                  |     ✔       |     ✔        |    ✔    |
| [Fan et al.](https://doi.org/10.1007/s00261-023-03868-3)             | AP & VP CT  |  92 | LR                             |     ✔       |     ✔        |         |
| [Dai et al.](https://doi.org/10.1016/j.acra.2023.05.027)             | AP & VP CT  | 351 | LR                             |     ✔       |     ✔        |         |
| [Chang et al.](https://doi.org/10.1109/IEEECONF58974.2023.10405004)  | AP, VP & NC CT | 105 | DenseNet-121               |             |             |    ✔    |
| [Bernatz et al.](https://doi.org/10.1038/s41598-023-27714-0)         | AP & VP CT  |  61 | RF                             |     ✔       |     ✔        |         |
| [An et al.](https://doi.org/10.1186/s12885-023-10620-z)              | AP CT       | 289 | RF                             |     ✔       |     ✔        |         |
| [Zhang et al.](https://doi.org/10.1186/s40644-022-00457-3)           | DSA         | 605 | DSA-Net (U-Net + MLP)          |     ✔       |             |    ✔    |
| [Wang et al.](https://doi.org/10.1016/j.ejrad.2022.110527)           | VP CT       | 543 | EfficientNetV2                 |             |             |    ✔    |
| [Ren et al.](https://doi.org/10.3389/fbioe.2022.872044)              | AP CT       | 103 | KNN & ResNet50                 |             |     ✔        |    ✔    |
| [Peng et al.](https://doi.org/10.3389/fonc.2022.853254)              | AP CT       | 313 | LR                             |     ✔       |     ✔        |         |
| [Li et al.](https://doi.org/10.3390/jpm12020248)                     | AP & VP CT  | 248 | Multi-task Deep Learning Model |     ✔       |             |    ✔    |
| [Bai et al.](https://doi.org/10.1007/s00270-022-03221-z)             | AP & VP CT  | 111 | SVM & LASSO LR                 |     ✔       |     ✔        |         |
| [Pino et al.](https://doi.org/10.1109/EMBC46164.2021.9630913)        | Late AP & DP CT |  92 | TwinLiverNet                 |             |             |    ✔    |
| [Niu et al.](https://doi.org/10.3748/WJG.V27.I2.189)                 | AP CT       | 218 | LR                             |     ✔       |     ✔        |         |
| [Guo et al.](https://doi.org/10.2147/jhc.s316117)                    | NC CT       |  94 | LR                             |     ✔       |     ✔        |         |
| [Chen et al.](https://doi.org/10.1159/000512028)                     | AP and NC CT | 595 | LR                          |     ✔       |     ✔        |         |
| [Peng et al.](https://doi.org/10.1007/s00330-019-06318-1)            | AP CT       | 789 | ResNet50                       |             |             |    ✔    |
| [Morshid et al.](https://doi.org/10.1148/ryai.2019180021)            | VP CT       | 105 | RF                             |     ✔       |     ✔        |         |

*Table: Summary of studies on prediction models for TACE response*

## Clinical-Radiomics Approaches
For pipelines combining radiomic and clinical features, we have analysed all combinations of model and set inputs.
<p align="center">
  <img src="https://github.com/PietroMarvulli/TACE/blob/main/README_imgs/scheme1.png" alt="Clinical-Radiomics Approaches" width="350"/>
</p>
<p align="center"><b>Figura 1:</b> Clinical-Radiomics Approaches</p>

## Deep and Mixed Aprroaches
As for the methods that integrate feature extraction or deep learning model training, we chose to replicate the most representative works and which had more information on parameters and implementation.
In particular, the work of [Peng et al.](https://doi.org/10.1007/s00330-019-06318-1), [Ren et al.](https://doi.org/10.3389/fbioe.2022.872044), [Sun et al.](https://doi.org/10.2147/JHC.S443660) and [Wang et al.](https://doi.org/10.1016/j.ejrad.2022.110527)
