# Improving Recommendation Throught Personalized Explanations and Linked Open Data

## Description
This algorithm aims to improve recommendation engine's accuracy and generate explanations personalized to the user throught Linked Open Data with [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page).

## Reproduction
### Libraries

* [numpy 1.21.1](https://numpy.org/)
* [pandas 1.0.4](https://pandas.pydata.org/)
* [scipy 1.5.2](https://www.scipy.org/)
* [networkx 2.5.1](https://github.com/networkx/networkx) 
* [pygini 1.0.1](https://github.com/mckib2/pygini)
* [requests 2.25.1](https://github.com/psf/requests)
* [PyTorch 1.1.0](https://pytorch.org/)
* [SPARQLWrapper 1.8.5](https://github.com/RDFLib/sparqlwrapper)
* [CaseRecommender 1.1.0](https://github.com/caserec/CaseRecommender)

### Enviroment
To install the libraries used in this project, use the command: 
    
    pip install requirements

Or create a conda enviroment with the following command:

    conda create --name <env> --file requirements.txt
    
After this step it is necessary to install the CaseRecommender library with the command:
    
    pip install -U git+git://github.com/caserec/CaseRecommender.git

We used [Anaconda](https://www.anaconda.com/) to run the experiments. The version of Python used was the [3.7.3](https://www.python.org/downloads/release/python-373/).

### Command-Line Documentation Arguments to Run Experiments
You can run experiments with command line arguments. 

The documentation of each arguments follows bellow along with examples that was the commands used in the experiments:

* mode:  Set 'run' to run accuracy experiments, 'validate' to run statistical accuracy relevance tests, 'explanation' to run 
explanation experiments and 'validate_expl' to run statistical accuracy tests;

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

* save: Boolean argument to save or not result in file. Only works on the 'validation' mode.

* fold: Fold to consider when generating explanations. Only works on 'explanation' mode.

* min: Minimum number of user interacted items to explain. Works on the 'explanation' mode.

* max: Maximum number of user interacted items to explain. Works on the 'explanation' mode.

* max_users: Maximum number of users to generate explanations to. Works on the 'explanation' mode.

* reordered_recs: Explain baseline or reordered algorithm. Works on the 'explanation' mode.

* expl_alg: Algorithm to explain recommendations. Either max, diverse or explod. Works only on 'explanation' mode.

Therefore there are three main commands: the 'run' that is responsable of running an experiment, the 'validate' to run a statistical relevance test comparison of a baseline with a proposed metric and 
the 'explanation' mode that generates to a fold recommendations just like the run, but it prints on the console the items names, the semantic profile and the explanation paths.  

### Examples

To run the MovieLens experiments use the following command line:

    python main.py --mode=run --dataset=ml --begin=0 --end=9 --alg="MostPop BPRMF UserKNN PageRank NCF EASE" --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last

To run the Lastfm experiments use the following command line:

    python main.py --mode=run --dataset=lastfm --begin=0 --end=9 --alg="MostPop BPRMF UserKNN PageRank NCF EASE" --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last

To run a statistical relevance test use the following command in which the bprmf baseline is compared to the proposed reordering with policy of last items, percentage of historic items to build user profile of 0.1 and reordering of the top 10 of the baseline:

    python main.py --mode=validate --dataset=lastfm --sufix=path[policy=last_items=01_reorder=10_hybrid] --baseline=bprmf --method="both" --save=1 --metrics="MAP AGG_DIV NDCG GINI ENTROPY COVERAGE"

To run an explanation experiments for the movielens dataset for the diverse explanations algorithm run the following command. To compare results with the ExpLOD algorithm change the parameter to explod on expl_alg parameter. Change the reordered_recs parameter to explain the reordered recommendations or the baseline algorithm:
    
    python main.py --mode=explanation --dataset=ml --begin=0 --end=9 --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last --min=0 --max=0 --max_users=0 --expl_alg=diverse --reordered_recs=0

To run an explanation experiments for the lastfm dataset for the diverse explanations algorithm run the following command. To compare results with the ExpLOD algorithm change the parameter to explod on expl_alg parameter. Change the reordered_recs parameter to explain the reordered recommendations or the baseline algorithm:
    
    python main.py --mode=explanation --dataset=lastfm --begin=0 --end=9 --reord="MostPop BPRMF UserKNN PageRank NCF EASE" --nreorder=10 --pitems=0.1 --policy=last --min=0 --max=0 --max_users=0 --expl_alg=diverse --reordered_recs=0

To run an explanation samples for the diverse explanations algorithm on the movielens dataset for the PageRank algorithm for a maximum of 2 users that have at least 0 interactions and a max of 20 interactions. To compare results with the ExpLOD algorithm change the parameter to explod on expl_alg parameter. Change the reordered_recs parameter to explain the reordered recommendations or the baseline algorithm:

    python main.py --mode=explanation --dataset=ml --begin=0 --end=1 --reord="PageRank" --nreorder=10 --pitems=0.1 --policy=last --min=0 --max=20 --max_users=2 --expl_alg=diverse --reordered_recs=0
    
To evaluate the explanations diversity offline for the PageRank algorithm and movielens dataset use the command:
    
    python main.py --mode=validate_expl --baseline=wikidata_page_rank8020 --dataset=ml --reordered_recs=0

