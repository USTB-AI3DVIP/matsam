# MatSAM: Efficient Extraction of Microstructures of Materials via Visual Large Model

This repository includes the source code and test data for the paper https://arxiv.org/abs/2401.05638
> We are continuously improving the code and the documentation. 
> If you have any questions or suggestions, please feel free to contact us.

## Abstract 


## To run the code

### Environment requirements

We suggest using conda to create a virtual environment listed in /requirements.txt.

### Statement of the code

For ease of reproducibility, we have provided the source code in the form of a jupyter notebook. 
The notebook is divided into sections, each of which corresponds to a step in the proposed method.
Sets of hyperparameters are provided in the notebook, 
and the user can adjust them according to the specific requirements of the dataset.


- /notebook/matsam_example.ipynb gives an example of how to use the code to extract grains from a polycrystalline microscopy image.

- /notebook/matsam_public_datasets.ipynb gives the evaluation results of the proposed method on the public datasets:
  - NBS-2 (Ni-superalloys-Super2 in https://doi.org/10.1038/s41524-022-00878-5)
  - NBS-3 (Ni-superalloys-Super4 in https://doi.org/10.1038/s41524-022-00878-5)
  - AZA (XCT-Al-Zn-alloy in https://doi.org/10.1016/j.matchar.2020.110119)
  - UHCS (Ultrahigh-carbon-steel in https://doi.org/10.1016/j.actamat.2023.119086)

> For in-house datasets, we will provide partial test data and the corresponding ground truth in the future updates.