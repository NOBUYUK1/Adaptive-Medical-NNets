# MedNNets: Adaptive Medical Neural Networks

This is the Pytorch implementation for our project focused on the analysis of lossy and domain-shifted datasets of medical images.

**TLDR;** We've used adversarial learning to develop a high-performing network trained on unannotated medical images. This allows the networks to perform impressively across different data distributions.

## Getting Started

### System Requirements
- Linux (Ubuntu 18.04.05)
- NVIDIA GPU (Nvidia GeForce GTX 1080 Ti x 4 on local workstations, and Nvidia V100 GPUs on Cloud)
- Python (v3.6)

### Python Requirements
Various Python libraries are required, including Pytorch 1.4.0, PyYAML 5.3.1, scikit-image 0.14.0, scikit-learn 0.20.0, SciPy 1.1.0, opencv-python 4.2.0.34, Matplotlib 3.0.0, NumPy 1.15.2 etc.

## Dataset Preparation
.txt files are lists for source and target domains. Detailed instructions and examples of naming and organizing your data are provided.

## Training
Different methods to train the MD-nets are provided, along with the commands and parameters needed.

### Automated execution
To run all the experiments reported in the project, use the command: `./experiments/scripts/run_DATA_SET_NAME_experiments.sh`.
The experiment log file and the saved models will be stored at `./experiments/logs/EXPERIMENT_NAME/` and `./experiments/models/EXPERIMENT_NAME`

## Testing
You can test the datasets on reported models as per the instructions provided.

## Citing and Contact
Please cite our work if you find it useful, follow the instructions to format the citation correctly.

The project is maintained by NOBUYUK1, for a more detailed description of the project consult the complete ReadMe content. Contact us via hshafiee[at]bwh.harvard.edu for any questions or queries.