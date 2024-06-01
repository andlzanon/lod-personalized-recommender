# Balancing the trade-off between accuracy and diversity in recommender systems with personalized explanations based on Linked Open Data

## Description 
This is the source code used for the experiments of two papers:

The first paper is you can find the link to our paper on ScienceDirect [here](https://www.sciencedirect.com/science/article/pii/S0950705122006682). It proposes a reordering algorithm that aims to improve or maintain a collaborative filtering recommendation engine's accuracy, while also providing more diversity, coverage and fairness with the ability to generate personalized explanations to the user with the Wikidata Linked Open Data.
 

>Zanon, André Levi, Leonardo Chaves Dutra da Rocha, and Marcelo Garcia Manzato. "Balancing the trade-off between accuracy and diversity in recommender systems with personalized explanations based on Linked Open Data." Knowledge-Based Systems 252 (2022): 109333.


The second paper proposes a approach to explaining recommendation based on graph embeddings, that are trained on the Wikidata Linked Open Data. A cosine similarity is between a user embedding and a path embedding. The user embedding is the sum pooling of the user's interacted items embeddings and the path's embedding are the sum pooling of item and edges that connect an interacted item with a recommended. The path with most similarity to the user is chosen to be displayed. 

>Zanon, André Levi, Leonardo Chaves Dutra da Rocha, and Marcelo Garcia Manzato. "Model-Agnostic Knowledge Graph Embedding Explanations for Recommender Systems". The 2nd World Conference on eXplainable Artificial Intelligence (2023)" 


## Citation 
If this repository could be usefull to you, please cite us:
    
    @article{zanon2022balancing,
        title       = {Balancing the trade-off between accuracy and diversity in
                     recommender systems with personalized explanations based on 
                     Linked Open Data},
        author      = {Zanon, Andr{\'e} Levi and 
                      da Rocha, Leonardo Chaves Dutra and 
                      Manzato, Marcelo Garcia},
        journal     = {Knowledge-Based Systems},
        volume      = {252},
        pages       = {109333},
        year        = {2022},
        publisher   = {Elsevier}
    }


## Project Organization

:file_folder: datasets: file with MovieLens 100k and LastFM datasets, folds of cross validation and experiments outputs and results for all folds

:file_folder: generated_files: files of metadata generated from the Wikidata for items on both datasets

:file_folder: preprocessing: source code for extracting Wikidata metadata and cross validation folder creation

:file_folder: recommenders: implementation of recommender engines and proposed reordering approach. Each file represents one recommendation engine, except the Neural Collaborative Filtering algorithm that has two classes with the NCF prefix. A [base class](https://github.com/andlzanon/lod-personalized-recommender/blob/main/recommenders/base_recommender.py) for all recommenders was also implemented.

:page_facing_up: main.py: main source code to run command line arguments experiments

:page_facing_up: evaluation_utils.py: evaluation of recommender engines source code

:page_facing_up: requirements.txt: list of library requirements to run the code

## Wikidata extracted metatdata 
The files [props_wikidata_movilens_small.csv](https://github.com/andlzanon/lod-personalized-recommender/blob/main/generated_files/wikidata/props_wikidata_movielens_small.csv) and [props_artists_id.csv](https://github.com/andlzanon/lod-personalized-recommender/blob/main/generated_files/wikidata/last-fm/props_artists_id.csv) contains the [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page). metadata extracted using [SPARQLWrapper 1.8.5](https://github.com/RDFLib/sparqlwrapper) library for the [MovieLens 100k dataset](https://github.com/andlzanon/lod-personalized-recommender/tree/main/datasets/ml-latest-small) and the and the [LastFM artist dataset](https://github.com/andlzanon/lod-personalized-recommender/tree/main/datasets/hetrec2011-lastfm-2k). For the MovieLens we extracted metadata from 97% of the movies available and for the LastFM we extracted 66% of the artists available.

## Reproduction 

All the generated files and results are available in this repository for the [MovieLens 100k database](https://github.com/andlzanon/lod-personalized-recommender/tree/main/datasets/ml-latest-small/folds) and the [LastFM database](https://github.com/andlzanon/lod-personalized-recommender/tree/main/datasets/hetrec2011-lastfm-2k/folds). Bellow are the libraries and command line arguments to reproduce the results of those two folders.

### Libraries

* [numpy 1.21.1](https://numpy.org/)
* [pandas 1.0.4](https://pandas.pydata.org/)
* [scipy 1.5.2](https://www.scipy.org/)
* [networkx 2.5.1](https://github.com/networkx/networkx) 
* [pygini 1.0.1](https://github.com/mckib2/pygini)
* [sklearn 0.20.3](https://github.com/scikit-learn/scikit-learn)
* [openpyxl 3.0.7](https://github.com/theorchard/openpyxl)
* [gensim 4.2.0](https://github.com/piskvorky/gensim)
* [node2vec 0.4.3](https://github.com/eliorc/node2vec)
* [requests 2.25.1](https://github.com/psf/requests)
* [pytorch 1.13.1](https://pytorch.org/)
* [sparqlwrapper 1.8.5](https://github.com/RDFLib/sparqlwrapper)
* [caserecommender 1.1.0](https://github.com/caserec/CaseRecommender)
* [pykeen 1.5.0](https://github.com/pykeen/pykeen)

### Enviroment 
To install the libraries used in this project, use the command: 
    
    pip install requirements

Or create a conda enviroment with the following command:

    conda env create --f requirements.yml
    
After this step it is necessary to install the CaseRecommender library with the command:
    
    pip install -U git+git://github.com/caserec/CaseRecommender.git

We used [Anaconda](https://www.anaconda.com/) to run the experiments. The version of Python used was the [3.7.3](https://www.python.org/downloads/release/python-373/).

## Documentation

### Command-Line Documentation Arguments to Run Experiments 
You can run experiments with command line arguments. 

The documentation of each arguments follows bellow along with examples that was the commands used in the experiments:

* `mode`:  Set 'run' to run accuracy experiments, 'validate' to run statistical accuracy relevance tests, 'explanation' to run 
explanation experiments, 'validate_expl' to run statistical accuracy tests or 'maut' to run the multi-attribute utility theory
for an explanation algorithm;

* `dataset`: Either 'ml' for the small movielens dataset or 'lastfm' for the lastfm dataset;

* `begin`: Fold to start the experiment;

* `end`: Fold to end the experiment;

* `alg`: Algoritms to run separated by space. E.g.: "MostPop BPRMF UserKNN PageRank NCF EASE. Only works on the 'run' and 'maut' modes;

* `reord`: Algoritms to reorder separated by space. E.g.: "MostPop BPRMF UserKNN PageRank NCF EASE." Only works on the 'run' mode;

* `nreorder`: Number of recommendations to reorder. Only works on the 'run' mode;

* `pitems`: Set of items to build user semantic profile. Only works on the 'run' mode;

* `policy`: Policy to extract set of items to build semantic profile. 'all' to get all items, 'last' for the last interacted, 'first' for the first interacted, 'random' for random items. Only works on the 'run' mode;

* `baseline`: Name of the file without extension of the baseline to validate results. E.g.: 'bprmf'. Only works on the 'validation' mode;

* `sufix`: Reorder sufix on result file after the string of the baseline. E.g.: 'bprmf'. Only works on the 'validation' mode;

* `metrics`: Reorder sufix on result file after the string of the baseline. E.g.: path[policy=last_items=01_reorder=10_hybrid]. Only works on the 'validation' mode;

* `method`: Statistical relevance test. Either 'ttest', 'wilcoxon' or 'both'. Only works on the 'validation' mode.

* `save`: Boolean argument to save or not result in file. Only works on the 'validation' mode.

* `fold`: Fold to consider when generating explanations. Only works on 'explanation' mode.

* `min`: Minimum number of user interacted items to explain. Works on the 'explanation' mode.

* `max`: Maximum number of user interacted items to explain. Works on the 'explanation' mode.

* `max_users`: Maximum number of users to generate explanations to. Works on the 'explanation' mode.

* `reordered_recs`: Explain baseline or reordered algorithm. Works on the 'explanation' mode.

* `expl_alg`: Algorithm to explain recommendations. Either explod, explod_v2, pem, diverse or rotate. Works only on 'explanation' mode.

* `n_explain`: Number of recommendations to explain. Min 1, max: 10. Works only on 'explanation' mode.

* `expl_algs`: List of explanation algorithms to get explanations from ouputed explanations. Options of explanation algorithms are explod, explod_v2, pem, diverse or rotate. 
Works only on 'maut' mode.


Therefore there are three main commands: the 'run' that is responsable of running an experiment, the 'validate' to run a statistical relevance test comparison of a baseline with a proposed metric and 
the 'explanation' mode that generates to a fold recommendations just like the run, but it prints on the console the items names, the semantic profile and the explanation paths.  

### Examples

To run the MovieLens experiments use the following command line:

    python main.py --mode=run --dataset=ml --begin=0 --end=9 --alg="MostPop BPRMF UserKNN PageRank NCF EASE" --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last

To run the Lastfm experiments use the following command line:

    python main.py --mode=run --dataset=lastfm --begin=0 --end=9 --alg="MostPop BPRMF UserKNN PageRank NCF EASE" --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last

To run a statistical relevance test for ranking metrics use the following command in which the bprmf baseline is compared to the proposed reordering with policy of last items, percentage of historic items to build user profile of 0.1 and reordering of the top 10 of the baseline:

    python main.py --mode=validate --dataset=lastfm --sufix=path[policy=last_items=01_reorder=10_hybrid] --baseline=bprmf --method="both" --save=1 --metrics="MAP AGG_DIV NDCG GINI ENTROPY COVERAGE"

To run an explanation experiments for the movielens dataset for the explod_v2 explanations algorithm run the following command. To compare results with the ExpLOD algorithm change the parameter to explod on expl_alg parameter or pem to the PEM algorithm. To run with the reordered explanation of the KBS paper change the reordered_recs param to 1:
    
    python main.py --mode=explanation --dataset=ml --begin=0 --end=9 --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last --min=0 --max=0 --max_users=0 --expl_alg=explod_v2 --reordered_recs=0 --n_explain=5

To run an explanation samples for the pem explanations algorithm on the movielens dataset for the PageRank algorithm for a maximum of 2 users that have at least 0 interactions and a max of 20 interactions. To compare results with the ExpLOD algorithm change the parameter to explod on expl_alg parameter. Change the reordered_recs parameter to explain the reordered recommendations or the baseline algorithm:

    python main.py --mode=explanation --dataset=ml --begin=0 --end=1 --reord="PageRank" --nreorder=10 --pitems=0.1 --policy=last --min=0 --max=20 --max_users=2 --expl_alg=pem --reordered_recs=0
    
To run a statistical relevance test for explanation metrics for the PageRank algorithm and movielens dataset use the command:
    
    python main.py --mode=validate_expl --baseline=wikidata_page_rank8020 --dataset=ml --reordered_recs=0
    
To run the Multi-Attribute Utility Theory between explanation algorithms for a recommendation algorithm and explanation metrics run the following command: 
    
    python main.py --mode=maut --dataset=ml --expl_algs="explod explod_v2 pem rotate" --alg=ease --expl_metrics="LIR SEP ETD" --n_explain=5

### Papers Command-Line Experiments

To run the experiments from the paper: "Balancing the trade-off between accuracy and diversity in recommender systems with personalized explanations based on Linked Open Data", 
run the following commands:
 
To run the ranking of the base algorithms and reordering with the following commands:
    
    python main.py --mode=run --dataset=ml --begin=0 --end=9 --alg="MostPop BPRMF UserKNN PageRank NCF EASE" --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last
    python main.py --mode=run --dataset=lastfm --begin=0 --end=9 --alg="MostPop BPRMF UserKNN PageRank NCF EASE" --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last

Then compare the algorithms with the following command to perform statistical validation. Change the dataset and baseline accordingly to the documentation. 
    
    python main.py --mode=validate --dataset=lastfm --sufix=path[policy=last_items=01_reorder=10_hybrid] --baseline=bprmf --method="both" --save=1 --metrics="MAP AGG_DIV NDCG GINI ENTROPY COVERAGE"

To run the experiments from the paper: "Model-Agnostic Knowledge Graph Embedding Explanations for Recommender Systems", 
run the following commands:

    python main.py --mode=explanation --dataset=ml --begin=0 --end=0 --reord="EASE" --nreorder=5 --pitems=0.1 --policy=last --min=0 --max=0 --max_users=0 --expl_alg=rotate --reordered_recs=0 --n_explain=5

In the paper we ran for both datasets and all recommendation algorithms (`MostPop BPRMF UserKNN PageRank NCF EASE`) 
on the `reord` param. The `n_explain` was 1 and 5. The `expl_alg` was `explod`, `explod_v2`, `pem` and `rotate`.
Only the fold 0 (`begin` and `end` should be 0 for all experiments), was used in the paper's experiments, therefore a 90/10 
split for training and testing.

Then to run the method MAUT run the command:
    
    python main.py --mode=maut --dataset=ml --expl_algs="explod explod_v2 pem rotate" --alg=ease --expl_metrics="LIR SEP ETD" --n_explain=5

In this command we run maut for the ml dataset, comparing the `expl_alg` for the `ease` recommender using the metrics `LIR SEP ETD`
as attributes. In the paper we ran for both datasets, all recommendation algorithms (`MostPop BPRMF UserKNN PageRank NCF EASE`) and `n_explain` 1 and 5. 

## Papers outputs and metrics

All results for all the papers are in this repository. To find them use the `datasets` folder and then choose the MovieLens
or LastFM datasets folder. In the folds folder there are the 10 folds used. 

- For each folds folder there is the output folder  

    - In the root, there are ranking items for every user for a recommender or a recommender and a reordering algorithm. 
    The file name represents if there were reoderings with the addition of the word `lod_reorder_path` 
    in the beginning of file name, along with the params for the reodering algorithm.
    
    - In the explanations folder there are the paths extracted for every user of the dataset for an explanation
    algorithm. The file name represents the params used, therefore, the explanation algorithm used,
    if the recommendation was reordered, the quantity of recommendations to explain and
    and the recommender that generated the recommendations are on the file name.

- For each folds folder there is the results folder 
    
    - In the root, there are ranking metrics for a recommender or a recommender and a reordering algorithm. 
    The file name represents if there were reoderings with the addition of the word `lod_reorder_path` 
    in the beginning of file name, along with the params for the reodering algorithm.
    
    - In the explanations folder there are the paths metrics for the explanation algorithm. The file name represents the params used,
    therefore, the explanation algorithm used, if the recommendation was reordered, the quantity of recommendations to explain and
    and the recommender that generated the recommendations are on the file name.
