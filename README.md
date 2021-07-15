# Semi_Supervised_Learning
Implementation of Semi-Supervised and Semi-Weakly Supervised Teacher/Student Learning approaches.

If you find this code useful, consider citing the accompanying article:
Sebastian Otálora, Niccolò Marini, Manfredo Atzori, and Henning Müller, et al. "Semi-Weakly Supervised Learning for Prostate Cancer Image Classification with Teacher-Student Deep Convolutional Networks." The 5th MICCAI Workshop on Large-scale Annotation of Biomedical data and Expert Label Synthesis LABELS 2020.

## Requirements
Python==3.6.9, albumentations==0.1.8, numpy==1.17.3, opencv==4.2.0, pandas==0.25.2, pillow==6.1.0, torchvision==0.3.0, pytorch==1.1.0

## Best models
- The best models for the Teacher and the Student, trained with the Semi-Weakly Supervised approach, are available [here](https://drive.google.com/drive/folders/1HdCrMq5ojhi9fKF23e3viRt0hrVtRbvl?usp=sharing).
- The best models for the Teacher and the Student, trained with the Semi-Supervised approach, are available upon the paper publication.

## Datasets
Two datasets are used for the experiments:
- [The Tissue Micro Array Zurich (TMAZ)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP)
- [The Cancer Genome Atlas-PRostate ADenocarcinoma (TCGA-PRAD)](https://portal.gdc.cancer.gov/projects/TCGA-PRAD) 

The lists of images used (split in training, validation, testing partitions) are in: 
- [TMAZ_images](https://github.com/ilmaro8/Semi_Supervised_Learning/tree/master/csv_files/List_Strongly_Annotated_Images/)
- [TCGA_images](https://github.com/ilmaro8/Semi_Supervised_Learning/tree/master/csv_files/List_Weakly_Annotated_Images/)

## Folder organization

### csv_files
It inclused the csv files. The csv files include a list tuples: filename, label.

```
├── csv_files
│   ├── List_Strongly_Annotated_Images
│   │   ├── test_images.csv
│   │   ├── train_images.csv
│   │   └── valid_images.csv
│   ├── List_Weakly_Annotated_Images
│   │   ├── test_WSI.csv
│   │   ├── train_WSI.csv
│   │   └── valid_WSI.csv
│   ├── Pseudo_Labeled_Data
│   │   ├── Semi_Supervised
│   │   │   ├── test
│   │   │   │   ├── csv_densely_semi_annotated_subset_0.csv
│   │   │   │   └── ...
│   │   │   ├── train
│   │   │   │   ├── csv_densely_semi_annotated_subset_0.csv
│   │   │   │   └── ...
│   │   │   └── valid
│   │   │       ├── csv_densely_semi_annotated_subset_0.csv
│   │   │       └── ...
│   │   └── Semi_Weakly_Supervised
│   │       ├── test
│   │       │   ├── csv_densely_semi_annotated_subset_0.csv
│   │       │   └── ...
│   │       ├── train
│   │       │   ├── csv_densely_semi_annotated_subset_0.csv
│   │       │   └── ...
│   │       └── valid
│   │           ├── csv_densely_semi_annotated_subset_0.csv
│   │           └── ...
│   ├── Strongly_Annotated_data
│   │   ├── test_patches.csv
│   │   ├── train_patches.csv
│   │   └── valid_patches.csv
│   └── Weakly_Annotated_Data
│       ├── test_patches.csv
│       ├── train_patches.csv
│       └── valid_patches.csv

```

### Images_masks
It includes the tissue masks.

```
├── Images_masks
│   ├── TMA
│   │   ├── test
│   │   │   ├── masks_0.jpg
│   │   │   └── ...
│   │   ├── train
│   │   │   ├── masks_0.jpg
│   │   │   └── ...
│   │   └── valid
│   │       ├── masks_0.jpg
│   │       └── ...
│   └── WSI
│       ├── test
│       │   ├── masks_0.jpg
│       │   └── ...
│       ├── train
│       │   ├── masks_0.jpg
│       │   └── ...
│       └── valid
│           ├── masks_0.jpg
│           └── ...
```

### Inference_scripts
It includes the scripts to test both the models at patch level and at whole slide image level.

### model_weights
It includes the weights of the models.

```
models_weights
│   ├── Semi_Supervised
│   │   ├── Student_model
│   │   │   ├── fully_supervised
│   │   │   │   └── N_EXP_0
│   │   │   │       ├── checkpoints
│   │   │   │       └── student_model.pt
│   │   │   ├── student_variant_I
│   │   │   │   ├── perc_18
│   │   │   │   │   ├── N_EXP_0
│   │   │   │   │   │   ├── checkpoints
│   │   │   │   │   │   └── student_model.pt
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   ├── student_variant_II
│   │   │   │   ├── perc_18
│   │   │   │   │   ├── N_EXP_0
│   │   │   │   │   │   ├── checkpoints
│   │   │   │   │   │   └── student_model.pt
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── student_variant_III
│   │   │       ├── perc_18
│   │   │       │   ├── N_EXP_0
│   │   │       │   │   ├── checkpoints
│   │   │       │   │   └── student_model.pt
│   │   │       │   └── ...
│   │   │       └── ...
│   │   └── Teacher_model
│   │       └── strong_labels_training
│   │           └── N_EXP_0
│   │               ├── checkpoints
│   │               └── teacher_model.pt
│   └── Semi_Weakly_Supervised
│       ├── Student_model
│       │   ├── fully_supervised
│       │   │   └── N_EXP_0
│       │   │       ├── checkpoints
│       │   │       └── student_model.pt
│       │   ├── student_variant_I
│       │   │   ├── perc_18
│       │   │   │   ├── N_EXP_0
│       │   │   │   │   ├── checkpoints
│       │   │   │   │   └── student_model.pt
│       │   │   │   └── ...
│       │   │   └── ...
│       │   ├── student_variant_II
│       │   │   ├── perc_18
│       │   │   │   ├── N_EXP_0
│       │   │   │   │   ├── checkpoints
│       │   │   │   │   └── student_model.pt
│       │   │   │   └── ...
│       │   │   └── ...
│       │   └── student_variant_III
│       │       ├── perc_18
│       │       │   ├── N_EXP_0
│       │       │   │   ├── checkpoints
│       │       │   │   └── student_model.pt
│       │       │   └── ...
│       │       └── ...
│       └── Teacher_model
│           ├── strong_labels_training
│           │   └── N_EXP_0
│           │       ├── checkpoints
│           │       └── teacher_model.pt
│           └── weak_labels_training
│               └── N_EXP_0
│                   ├── checkpoints
│                   └── teacher_model.pt
```

### Strongly_Annotated_patches
It includes the patches extracted from the strongly-annotated data

### Training_scripts
It includes the scripts to train both the models.

### utils
It includes scripts used to define methods.

### Weakly_Annotated_patches
It includes the patches extracted from the weakly-annotated data

## Scripts organization
The repository includes the scripts for training the models (training_scripts), for testing the models (inference scripts) and for the preprocessing/generation of the csv files (utils).

### Training_scripts
- Teacher_training.py -d -a -w -s -e -b. The script is used to train the teacher model.
	* -d: dataset to use to train the model (options: weak/strong).
	* -a: approach to use (ssl: semi-supervised learning, swsl: semi-weakly supervised learning).
	* -w: number of the pre-trained model to fine-tune with strongly-annotated data (only in semi-weakly supervised learning).
	* -s: number of experiment for the training.
	* -e: number of epochs to train the model.
	* -b: batch size.

- Student_training.py -v -a -s -n -p -e -b. The script is used to train the student model.
	* -v: student variant approach to use. 
		* -I: training only with pseudo-labels
		* -II: pre-training with pseudo-labels and fine-tuning with strongly-annotated data
		* -III: training with both pseudo-labels and strongly-annotated data
		* -baseline: training only with strongly-annotated data
	* -a: approach to use (ssl: semi-supervised learning, swsl: semi-weakly supervised learning).
	* -s: subset of pseudo labels to use. If the scripts within the repository are used to generate the pseudo-labels data, the -s varies between 19 (1000 pseudo labels per class) and 1 (20000 psuedo labels per class).
	* -n: number of experiment for the training.
	* -p: number of the pre-trained model to fine-tune with strongly-annotated data (only variant II).
	* -e: number of epochs to train the model
	* -b: batch size

### Inference_scripts:
- Teacher_inference_patches.py -d -a -n -b -t. The script is used to test the teacher model at patch level.
	* -d: dataset to use to test the model (options: weak/strong).
	* -a: approach to use (ssl: semi-supervised learning, swsl: semi-weakly supervised learning).
	* -n: number of experiment to test.
	* -b: batch size.

- Teacher_inference_WSI.py -d -a -n -b -t. The script is used to test the teacher model at whole slide image level.
	* -d: after with training to test the model (options: weak/strong).
	* -a: approach to use (ssl: semi-supervised learning, swsl: semi-weakly supervised learning).
	* -n: number of experiment to test.
	* -b: batch size.
	* -t: amount of the patches per WSI to use. The patches are ranked by the Blue-Ratio.

- Student_inference_patches.py -d -v -a -s -n -b. The script is used to test the student model at patch level.
	* -d: dataset to use to test the model (options: weak/strong).
	* -v: student training variant approach to test.
	* -a: approach to use (ssl: semi-supervised learning, swsl: semi-weakly supervised learning).
	* -s: subset of pseudo labels used to train the model.
	* -n: number of experiment to test.
	* -b: batch size

- Student_inference_WSI.py -v -a -s -n -b -t. The script is used to test the student model at whole slide image level.
	* -v: student training variant approach to test.
	* -a: approach to use (ssl: semi-supervised learning, swsl: semi-weakly supervised learning).
	* -s: subset of pseudo labels used to train the model.
	* -n: number of experiment to test.
	* -b: batch size.
	* -t: amount of the patches per WSI to use. The patches are ranked by the Blue-Ratio.

### utils
- TMA_Patch_Extractor.py -d -s -n -t -p. Script to extract the patches from the TMAZ dataset (using pixel-wise annotated masks).
	* -d: dataset where extract patches (train/valid/test).
	* -s: size of the tiles to extract (before the resize to 224x224).
	* -n: number of patches to extract.
	* -t: number of threads.
	* -p: minimum percentage of tissue in a tile to be accepted.

- WSI_Patch_Extractor.py -d -s -t -p. Script to extract the patches from the TCGA-PRAD dataset (using masks generated by [HistoQC](https://github.com/choosehappy/HistoQC)).
	* -d: dataset where extract patches (train/valid/test).
	* -s: size of the tiles to extract (before the resize to 224x224).
	* -t: number of threads.
	* -p: minimum percentage of tissue in a tile to be accepted.

- Annotator.py -d -n -a. Script to annotate the data with pseudo labels.
	* -d: dataset to annotate (train/valid/test).
	* -n: weights used to annotate (number of experiment).
	* -a: approach to use (ssl: semi-supervised learning, swsl: semi-weakly supervised learning).

- Create_Densely_Weak_Labels.py -d -t. Script to create weakly-annotated labels.
	* -d: dataset to extract patches (train, valid, test)
	* -t: amount of patches to select

- Create_Pseudo_labels.py -a -d. Script to select the top ranked pseudo labels.
	* -a: approach to use (ssl: semi-supervised learning, swsl: semi-weakly supervised learning).
	* -d: dataset to annotate (train/valid/test).

- Models.py. Definition of the teacher and the student models.

- DataGenerator.py. The generators for training and testing datasets, the data augmentation pipeline.

- ImbalancedDatasetSampler.py. The sampler used for the class-wise data augmentation. It is taken from [ufoym repository](https://github.com/ufoym/imbalanced-dataset-sampler).

## Reference
If you find this repository useful in your research, please cite:

[1] Marini, N., Otálora, S., Müller, H., & Atzori, M. (2021). Semi-supervised training of deep convolutional neural networks with heterogeneous data and few local annotations: an experiment on prostate histopathology image classification, Medical Image Analysis

Paper link: https://www.sciencedirect.com/science/article/pii/S1361841521002115

## Acknoledgements
This project has received funding from the EuropeanUnion’s Horizon 2020 research and innovation programme under grant agree-ment No. 825292 [ExaMode](http://www.examode.eu). Infrastructure fromthe SURFsara HPC center was used to train the CNN models in parallel. Otálora thanks Minciencias through the call 756 for PhD studies.
