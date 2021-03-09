# RePBubLik: Reducing the Polarized Bubble Radius with Link Insertions

_S. Haddadan, C. Menghini, M. Riondato, E. Upfal_

The topology of the hyperlink graph among pages expressing different opinions may influence the exposure of readers to diverse content. Structural bias may trap a reader in a “polarized” bubble with no access to other opinions. We model readers’ behavior as random walks. A node is in a “polarized” bubble if the expected length of a random walk from it to a page of different opinion is large. The structural bias of a graph is the sum of the radii of highly-polarized bubbles. We study the problem of decreasing the structural bias through edge insertions. “Healing” all nodes with high polarized bubble radius is hard to approximate within a logarithmic factor, so we focus on finding the best 𝑘 edges to insert to maximally reduce the structural bias. We present RePBubLik, an algorithm that leverages a variant of the random walk closeness centrality to select the edges to insert. RePBubLik obtains, under mild conditions, a constant-factor approximation. It reduces the structural bias faster than existing edge-recommendation methods, including some designed to reduce the polarization of a graph.

Full paper: [https://arxiv.org/pdf/2101.04751.pdf](https://arxiv.org/pdf/2101.04751.pdf)

<hr>


## Reproduce experiments

After downloading the repository do the following:

- In the root folder create the directory `data/` that you can download [here](googledrivelink)
- Install requirements: `pip install -r requirements.txt` (be sure to also [install GCC](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/))
- Run the following: `python setup.py`
- Compute the bubble radius and centralities of nodes in the graph: `python main.py -topic <TOPIC> -t 10 -proc radius -b 2 -topk 10 -unweighted false`
- Get possible edges to add for ROV: `python controversy_for_edges.py -topic <TOPIC>`
- Compute embedding for node2vec `python node2vec.py -topic <TOPIC> -r 20 -t 10`
- Get possible edges with node2vec `python n2v_candidates.py -topic <TOPIC> -t 10`
- Run `python main.py -topic <TOPIC> -t 10 -proc addition -b 2 -topk 10 -maxedges 400 -iter 1 -unweighted false`

## Run RePBubLik+ and baseline on your graph

In the folder `data/` add the following files:
- `<name>_blue_pre_labeled_nodes.tsv` composed of three columns `node_name \t node_id \t node_color`
- `<name>_red_pre_labeled_nodes.tsv` composed of three columns `node_name \t node_id \t node_color`
- `clickstream_weighted_edges.tsv` composed of three columns `node_id \t node_id \t weight`

Then repeat the procedure indicated above for reproduce the experiments. Note the the code is parallelized to work on multiple core. Some optimization for larger graph can be done to speed up the centrality computation.