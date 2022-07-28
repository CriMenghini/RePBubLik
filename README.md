# Reducing Polarization and Improving Diverse Navigability

In this repository we collect the code to reproduce the experiments of the paper:

* [RePBubLik: Reducing the Polarized Bubble Radius with Link Insertions](https://arxiv.org/pdf/2101.04751.pdf)

and the ShuffLik algorithm presented in the journal version of the article.

<hr>

## RePBubLik: Reducing the Polarized Bubble Radius with Link Insertions

_S. Haddadan, C. Menghini, M. Riondato, E. Upfal_

The topology of the hyperlink graph among pages expressing different opinions may influence the exposure of readers to diverse content. Structural bias may trap a reader in a “polarized” bubble with no access to other opinions. We model readers’ behavior as random walks. A node is in a “polarized” bubble if the expected length of a random walk from it to a page of different opinion is large. The structural bias of a graph is the sum of the radii of highly-polarized bubbles. We study the problem of decreasing the structural bias through edge insertions. “Healing” all nodes with high polarized bubble radius is hard to approximate within a logarithmic factor, so we focus on finding the best 𝑘 edges to insert to maximally reduce the structural bias. We present RePBubLik, an algorithm that leverages a variant of the random walk closeness centrality to select the edges to insert. RePBubLik obtains, under mild conditions, a constant-factor approximation. It reduces the structural bias faster than existing edge-recommendation methods, including some designed to reduce the polarization of a graph.


### Reproduce experiments

After downloading the repository do the following:

- In the root folder create the directory `data/` that you can download [here](https://drive.google.com/drive/folders/18XEFWgdx50lSlRY5EtnOzYToQt0j9f4W?usp=sharing)
- Install requirements: `pip install -r requirements.txt` (be sure to also [install GCC](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/))
- Run the following: `python setup.py`
- Compute the bubble radius and centralities of nodes in the graph: `python main.py -topic <TOPIC> -t 10 -proc radius -b 2 -topk 10 -unweighted false`
- Get possible edges to add for ROV: `python controversy_for_edges.py -topic <TOPIC>`
- Compute embedding for node2vec `python node2vec.py -topic <TOPIC> -r 20 -t 10`
- Get possible edges with node2vec `python n2v_candidates.py -topic <TOPIC> -t 10`
- Run `python main.py -topic <TOPIC> -t 10 -proc addition -b 2 -topk 10 -maxedges 400 -iter 1 -unweighted false`

### Run RePBubLik+ and baseline on your graph

In the folder `data/` add the following files:
- `<name>_blue_pre_labeled_nodes.tsv` composed of three columns `node_name \t node_id \t node_color`
- `<name>_red_pre_labeled_nodes.tsv` composed of three columns `node_name \t node_id \t node_color`
- `clickstream_weighted_edges.tsv` composed of three columns `node_id \t node_id \t weight`

Then repeat the procedure indicated above for reproduce the experiments. Note the the code is parallelized to work on multiple core. Some optimization for larger graph can be done to speed up the centrality computation.

<hr>

## ShuffLik: Increasing Diverse Navigability on Graphs by Swapping Edge Weights

### Reproduce experiments

1) Download the 25M movielens dataset
2) Build graphs on top of Movielens: `python3 movielens_datasets.py` and `python3 div_movielens_datasets.py`
3) Compute BR and centralities for a single graph: `python3 main.py -topic <TOPIC> -dataset movielens -t <LEN_WALKS> -proc diameter  -maxedges <NUM_SWAPS> -unweighted false -edges <GRAPH> -centralities <CENTRALITY>`
4) Run ShuffLik: `python3 swap_edges.py -topic <TOPIC> -dataset movielens -t <LEN_WALKS> -maxswap 500 -centralities <CENTRALITY> -edges <GRAPH>`
    - Choose the topic and the length of the rws (this par needs to be the same as the previous computation)
    - <GRAPH>: if vanilla applies the algorithm to the vanilla-RecNet, otherwise if diversity run it on dic-RecNet 
    - <CENTRALITY>: if true it runs ShuffLik, if false it runs WeightDifference. On diversity graph, we only run CENTRALITY=false


<hr>

## Cite us

If you either find our work interesting and related to yours or you use this codebase or anything in it, please cite this work!

```
@inproceedings{10.1145/3437963.3441825,
author = {Haddadan, Shahrzad and Menghini, Cristina and Riondato, Matteo and Upfal, Eli},
title = {RePBubLik: Reducing Polarized Bubble Radius with Link Insertions},
year = {2021},
isbn = {9781450382977},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3437963.3441825},
doi = {10.1145/3437963.3441825},
booktitle = {Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
pages = {139–147},
numpages = {9},
keywords = {fairness, polarization, bias},
location = {Virtual Event, Israel},
series = {WSDM '21}
}
```
