# TailDiff
This is the PyTorch implementation of the paper "TailDiff: Multimodal Semantic-guided Sequence Diffusion for Next POI Recommendation of Long-tail Users"

Please cite our paper if you use the code or datasets.
---

## Datasets
We evaluate TailDiff on four real-world POI check-in datasets:

- **Foursquare-NYC**: POI check-in data from New York City 
- **Foursquare-TKY**: POI check-in data from Tokyo  
- **Yelp-New Orleans**: POI check-in data from New Orleans 
- **Yelp-Philadelphia**: POI check-in data from Philadelphia  

Each dataset consists of chronological user check-in sequences, where each POI is associated with user interaction histories. All datasets are preprocessed with unified indexing (user/POI IDs start from 1).
### Dataset Statistics

| Dataset          | #Users | #POIs |    #Check-in   |
|------------------|--------|-------|----------------|
| Foursquare-NYC   |  1083  | 4638  |     142968     |
| Foursquare-TKY   |  2292  | 5930  |     407974     |
| Yelp-New Orleans |  5510  | 1515  |     119526     |
| Yelp-Philadelphia|  10105 | 2966  |     285074     |

## Code Structure
Main files and directories:

- `main.py`
- `models.py`
- `modules.py`
- `datasets.py`
- `utils.py`
- `trainers.py`
- `data_augmentation.py`
- `data/`  # Preprocessed datasets

## Model Overview

TailDiff consists of three main components:

1. Multimodal Semantic-aware User Modeling
Integrates POI images, review text, and metadata to construct semantic-aware user representations from limited interaction histories.

2. Conditional Diffusion-based Sequence Augmentation
Uses a conditional diffusion model to generate continuous augmented sequence representations for long-tail users and maps them to candidate POI sequences. 

3. Ranking-consistency-based Augmented Sequence Selection
Selects reliable augmented sequences based on their ranking consistency with the user's original interaction sequence for joint fine-tuning.

Key file roles:

- `main.py`: Program entry point; configures datasets, training stages, and evaluation.  
- `datasets.py`: Loads and constructs user–POI interaction sequences and multimodal POI representations.  
- `data_augmentation.py`: Implements diffusion-based augmented sequence generation, semantic POI mapping, and augmented sequence selection.  
- `models.py`: Defines the sequential recommendation model and multimodal POI representation components.  
- `modules.py`: Implements the conditional diffusion model and related neural network modules.  
- `trainers.py`: Handles model pretraining, diffusion training, joint fine-tuning, validation, and testing. 
- `utils.py`: Provides utility functions, including random seed initialization, data processing, evaluation metrics, and other helper functions.

Simplified execution overview:
 `main.py → datasets.py / data_augmentation.py → models.py → trainers.py / modules.py → utils.py`.



## Model Training
To train our model  on the four POI  datasets(with default hyper-parameters):

```bash
# Train on Foursquare-NYC dataset
python main.py --dataset NYC

# Train on Foursquare-TKY dataset
python main.py --dataset TKY

# Train on Yelp-New Orleans dataset
python main.py --dataset new_orleans

# Train on Yelp-Philadelphia dataset
python main.py --dataset philadelphia