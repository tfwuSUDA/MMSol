# MMSol

**Multimodal Protein Solubility Prediction with Noise-Resistant Learning**

> **Abstract**  
> Protein solubility plays a critical role in determining its biological function, such as enabling proper protein delivery and ensuring that proteins remain soluble during cellular processes or therapeutic applications. 
Accurate prediction of protein solubility with computational methods accelerates the development of therapeutically relevant proteins and industrial enzymes.
However, existing models do not fully account for the interaction of multimodal information and are limited by label noise in protein solubility experimental data.
To address this, we propose a new protein solubility prediction model MMSol that considers three modalities of information: sequence, structure, and function, which enrich the protein representation. Additionally, we incorporates an anti-noise algorithm during training to mitigate the impact of label noise.
In the empirical study, we evaluate our model on both noise-free and noisy datasets. The result demonstrates that due to our model's capability to integrate proteins' multi-modality, and the incorporation of the anti-noise algorithm, the model achieves superior performance in both noisy and noise-free scenarios.

---

## Installation

### Create the conda environment and activate it.

```bash
conda create -n mmsol python==3.10
conda activate mmsol
```

### Install basic packages
```bash
# install requirements
pip install -r requirements.txt

pip install biopython
```


## Dataset Preparation

Please download the required .pkl files (for node features, edge features, GO features, etc.) from the following address:https://zenodo.org/records/15117305

Download and extract the files under the ./data/ directory, following this structure:
  ./data/
    noise_free/
        eSOL_edge/
            noise_free_train_LPE_5_1.pkl
        eSOL_go/
            train_go.pkl
        ...
    noise/
        noise_edge/
        noise_go/
        ...


##  Training Tasks

###  Noise-Free Environment

- **Classification Task**

  To train a classification model in a noise-free setting:
```bash
  python train_noise_free.py
```
  To test:
```bash
  python test_noise_free.py
```
- **Regression Task**

  To train a regression model in a noise-free setting:
```bash
  python train_noise_free_reg.py
```
  To test:
```bash
  python test_noise_free_reg.py
```
###  Noise Environment

- **Classification Task**

  To train a classification model in a noise setting:
```bash
  python train_noise.py
```
  To test:
```bash
  python test_noise.py
```



