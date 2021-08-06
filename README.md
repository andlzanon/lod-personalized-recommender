# Improving Recommendation Throught Personalized Explanations and Linked Open Data

## Description
This algorithm aims to improve recommendation engine's accuracy and generate explanations personalized to the user throught Linked Open Data with [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page).

## Reproduction
### Libraries
To install the libraries used in this project use one of the commands: 
    
    pip install <lib>==<version>
    pip install requirements
    
To install the CaseRecommender library use the command (this is a necessary step when using the requirements file to install dependencies):
    
    pip install -U git+git://github.com/caserec/CaseRecommender.git

* [numpy 1.21.1](https://numpy.org/)
* [pandas 1.0.4](https://pandas.pydata.org/)
* [scipy 1.5.2](https://www.scipy.org/)
* [networkx 2.5.1](https://github.com/networkx/networkx) 
* [pygini 1.0.1](https://github.com/mckib2/pygini)
* [requests 2.25.1](https://github.com/psf/requests)
* [SPARQLWrapper 1.8.5](https://github.com/RDFLib/sparqlwrapper)
* [CaseRecommender 1.1.0](https://github.com/caserec/CaseRecommender)

We used [Anaconda](https://www.anaconda.com/) to run the experiments. The version of Python used was the [3.7.3](https://www.python.org/downloads/release/python-373/).

### Command-Line Arguments to Run Experiments
The command line arguments to run experiments are:

* mode:  Set 'run' to run experiments and 'validate' to run statistical relevance tests;

* dataset: Either 'ml' for the small movielens dataset or 'lastfm' for the lastfm dataset;

* begin: Fold to start the experiment;

* end: Fold to end the experiment;

* alg: Algoritms to run separated by space. E.g.: "MostPop BPRMF UserKNN PageRank NCF EASE. Only works on the 'run' mode;

* reord: Algoritms to reorder separated by space. E.g.: "MostPop BPRMF UserKNN PageRank NCF EASE." Only works on the 'run' mode;

* nreorder: Number of recommendations to reorder. Only works on the 'run' mode;

* pitems: Set of items to build user semantic profile. Only works on the 'run' mode;

* policy: Policy to extract set of items to build semantic profile. 'all' to get all items, 'last' for the last interacted, 'first' for the first interacted, 'random' for random items. Only works on the 'run' mode;

* baseline: Name of the file without extension of the baseline to validate results. E.g.: 'bprmf'. Only works on the 'validation' mode;

* sufix: Reorder sufix on result file after the string of the baseline. E.g.: 'bprmf'. Only works on the 'validation' mode;

* metrics: Reorder sufix on result file after the string of the baseline. E.g.: path[policy=last_items=01_reorder=10_hybrid]. Only works on the 'validation' mode;

* method: Statistical relevance test. Either 'ttest', 'wilcoxon' or 'both'. Only works on the 'validation' mode.

* save: Boolean argument to save or not result in file

Therefore there are two main commands: the 'run' that is responsable of running an experiment and the 'validate' to run a statistical relevance test comparison of a baseline with a proposed metric.

### Examples

To run the MovieLens experiments use the following command line:

    python main.py --mode=run --dataset=ml --begin=0 --end=9 --alg="MostPop BPRMF UserKNN PageRank NCF EASE" --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last

To run the Lastfm experiments use the following command line:

    python main.py --mode=run --dataset=lastfm --begin=0 --end=9 --alg="MostPop BPRMF UserKNN PageRank NCF EASE" --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last

To run a statistical relevance test use the following command in which the bprmf baseline is compared to the proposed reordering with policy of last items, percentage of historic items to build user profile of 0.1 and reordering of the top 10 of the baseline:

    python main.py --mode=validate --dataset=lastfm --sufix=path[policy=last_items=01_reorder=10_hybrid] --baseline=bprmf --method="both" --save=False --metrics="MAP NDCG GINI ENTROPY COVERAGE"



