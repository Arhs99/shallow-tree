# shallow-tree
Synthetic accessibility scoring is an invaluable tool for generative chemistry and more generally for filtering or scoring molecular designs that come either from AI or human designers. It's not hard to understand that synthetic scoring approaches that do not consider how the target molecule can be synthesized are of quite limited use in practice. This work aims to produce a tool that can predict synthetic routes but in a reasonable time frame for many real-world applications.
## Description
The idea of the tool is to restrict the depth of search in order to improve computation speed but in the same time having access to a large amount of synthetically accessible chemical space of enough complexity to be useful in drug discovery applications. A number of obvious optimizations like caching, parallel execution etc can significantly reduce times while other approaches could also be tested in the future. 

## Implementation
The tool is based on aizynthfinder provided by the MolecularAI group in AstraZeneca https://github.com/MolecularAI/aizynthfinder and specifically I use here:
- The expansion policy model and templates provided by the group
- A filter policy model provided by the group
Please refer to the detailed documentation in https://molecularai.github.io/aizynthfinder/ for instructions on how to train policy models, construct stocks etc 

## New features
- A **full tree search** is implemented (DFS) instead of MCTS for increased accuracy of predictions
- **Caching** tree branches works really well to increase speed by 2x-3x
- Vectorization of hot loops resulted in 3x speed gains
- A maximum search **depth can be set**
- A **context search mode** is available for the analysis of collections of molecules with a common scaffold (e.g. parallel libraries) where generally there is only one disconnection of interest in the first step of the retrosynthesis. See also the Jupyter notebook example.
- A large amount of code has been removed or refactored for simplification and speed optimisation.
- **Customized collections of templates** The current policy model and templates have been generated from the USPTO dataset and cover chemical synthesis knowledge before 2019 and thus uderrepresenting or not including at all important modern synthetic methods such as sp<sup>2</sup>-sp<sup>3</sup> cross coupling reactions, late-stage functionalization reactions and so on. This feature gives the option to add new reaction templates and enhance or otherwise modify the synthetic knowledge of the tool. In this repo you can find ```shallowtree/rules/direct.csv``` an example collection of templates that cover standard cross coupling reactions.

## Installation

- Clone the GitHub repository e.g. ```git clone https://github.com/Arhs99/shallow-tree.git```
- Then execute:
```commandline
cd shallow-tree
conda env create -f env.yml
conda activate shallow-tree
```
For use with a GPU install tensorflow from conda as follows, it will pick the correct CUDA libraries compatible with your set-up
```commandline
conda install -y tensorflow-gpu=2.8.0 -c conda-forge
```

## Usage
### In python code
One needs to import the ```Expander``` class and use any of the two available methods:
- ```search_tree``` provides scoring and predicted routes and starting materials for a set of query molecules
- ```context_search``` a SMILES string parameter is required for the desired scaffold which is also indicating the attachment point.

See the ```synth_score_NOTEBOOK.ipynb``` jupyter notebook example in this repository

### Command line
```search_cli``` is the command line tool. These are 2 examples for each of the available search modes:
```commandline
echo 'Clc1ccccc1COC5CC(Nc3n[nH]c4cc(c2ccccc2)ccc34)C5' | searchcli --config config.yml --scaffold '[*]c1n[nH]c2cc(-c3ccccc3)ccc12' --depth 2 --routes
```
and
```commandline
searchcli --config config.yml --depth 2 --routes < smiles.txt > routes.csv
```

## Parallelization
A significant gain in speed and extended ability to scale-up can be achieved by serving the models using ```tensorflow serving``` and parallelization by batching the SMILES inputs. See the [parallel folder README](shallowtree/parallel/README.md) for more details.

## References
1. Genheden S, Thakkar A, Chadimova V, et al (2020) AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.12465371.v1
