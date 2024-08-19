# pan-cancer-plasma-proteome

This repository includes the code used in pan-cancer-plasma-proteome cohort study.mic study.

**Plasma proteomic profiling of pan-cancer patients discovers biomarkers of cancers**

Lin Bai, Jiacheng Lyu, Jinwen Feng, Xiaoqiang Qiao, Yuanyuan Qu, Guojian Yang, Yuanxue Zhu, Lingxiao Liao, Hui Gao, Aimin Zang, Zeya Xu, Tao Ji, Wencong Ding, Hailiang Zhang, Lingli Zhu, Yan Wang, Liang Wang, Xiaofang Wang, Yumiao Li, Jinghua Li, Xiaoping Yin, Guofa Zhao, Dan Liu, Xiangpeng Gao, Sha Tian, Subei Tan, Yan Pu, Lingling Li, Yongshi Liao, Dingwei Ye, Youchao Jia, Chen Ding

## Code overview
> The below figure numbers were corresponded to the paper version.

### 1. 1-cohort_description-QC.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for cohort design, quality control and data distribution for the pan-cancer study

Output figures:  
* Figure 1b, 1c, S1b, S1c, S1e

### 2. 2-proteome_clustering.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the proteomic analysis of PC proteome clustering

Output figures:  
* Figure 2a, 2b, 2c, S2, S3c, S3d

### 3. 3-digestive_cluster_characterization.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the proteomic analysis of the characteristic of the Digestive-rich

Output figures:  
*  Figure 2d, S3e, S3f, S3g, S3i, S4a, S4b, S4c, S4d

### 4. 4-system_group_characterization.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the proteomic analysis of the characteristic of tumors of different physiological system groups

Output figures:  
* Figure 3a, 3b, 3c, 3e, 3f, 3g, S6

### 5. 5-LC_subtypes_characterization.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the proteomic analysis of the characteristic of different tumor types and the subtypes of the LC

Output figures:  
* Figure 4a, 4b, 4c, 4d, 4e, 4f, 4h, 4i, 4j, S7a

### 6. 6-core_tumor_marker.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the proteomic tumor biomarker candidates selection

Output figures:  
* Figure S8b, 5c, 5d, 5e

### 7. 7_ML_model.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for machine learning model evaluation

Output figures:  
* Figure S11a, S11b, 8b, 8c, 8d, 8e, 8f, 8g

## Environment requirement
The following package/library versions were used in this study:
* python (version 3.9.15)
* pandas (version 1.5.3)
* numpy (version 1.26.3)
* scipy (version 1.12.0)
* statsmodels (version 0.14.1)
* matplotlib (version 3.7.3)
* seaborn (version 0.11.2)
* scikit-learn (version 1.2.1)
* rpy2 (version 3.5.6)
* gprofiler (version 1.0.0)
* adjustText

## Folders Structure
The files are organised into four folders:
* *code*: contains the python code in the ipython notebook to reproduce all analyses and generate the the figures in this study.
* *document*: which contains all the proteomics and clinical patient informations required to perform the analyses described in the paper.
* *documents*: contains the related annotation files, the Source Data, and the Supplementary Table produced by the code.
* *figure*: contains the related plots produced by the code.
