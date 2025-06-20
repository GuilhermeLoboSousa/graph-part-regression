# GraphPart

[![PyPI version](https://badge.fury.io/py/graph-part.svg)](https://badge.fury.io/py/graph-part)
![PyPI - Downloads](https://img.shields.io/pypi/dm/graph-part)
[![PyPIDownloadsTotal](https://pepy.tech/badge/graph-part)](https://pepy.tech/project/graph-part)
[![Stars](https://img.shields.io/github/stars/graph-part/graph-part?logo=GitHub&color=yellow)](https://github.com/graph-part/graph-part/stargazers)



**Biological sequence dataset partitioning method**


Graph-Part is a Python package for generating partitions (i.e. train-test splits, or splits for cross-validation) of biological sequence datasets. It ensures minimal homology between different partitions, while balancing partitions for labels or other desired criteria.

Preprint: https://www.biorxiv.org/content/10.1101/2023.04.14.536886v1

## 🔧 Modifications in this fork made by me

This fork includes several modifications to the original code developed by the authors of the preprint above, with the goal of improving the quality and usability of the train/test splits, **now specifically adapted for regression problems** (continuous targets), not just binary or multiclass classification:

- **Better balance of target value distributions across sets:**  
  Partitions are constructed to ensure that the distribution of the continuous regression target (e.g., stability, activity, etc.) is as similar as possible between sets, using quantile-based binning and sum balancing.
- **Enforced sequence identity threshold** between datasets (e.g., between training and test sets), ensuring minimal homology and reducing data leakage.
- **Increased intra-set diversity**, which may enhance model generalization and learning performance.
- **Support for regression-specific partitioning:**  
  The partitioning algorithm now considers both the distribution of discretized target bins and the total sum of target values per partition, aiming for fair and representative splits for regression tasks.

These changes aim to generate more realistic and challenging sets for machine learning tasks involving biological sequences with continuous labels.

## 📊 Upcoming comparison with original results

In the next 7 days, I plan to run a comparative evaluation between this modified version (with regression-aware partitioning) and the original method described in the preprint. The results and main differences will be summarized and shared in a Jupyter Notebook within this repository.

## Installation

GraphPart relies on [needleall](https://www.bioinformatics.nl/cgi-bin/emboss/help/needleall) from the [EMBOSS](http://emboss.sourceforge.net/) package for Needleman-Wunsch alignments of sequences. Please refer to the official EMBOSS documentation for installation methods.
Additionally, GraphPart supports [MMseqs2](https://github.com/soedinglab/MMseqs2) for alignments. To use other algorithms that compute pairwise similarity measures, please refer to the `precomputed` mode.

We recommend to install GraphPart in a conda environment, and install EMBOSS from [bioconda](https://anaconda.org/bioconda/emboss). The same goes for [MMseqs2](https://anaconda.org/bioconda/mmseqs2).
```
conda install -c bioconda emboss

# If you also want to use the MMseqs2 mode
conda install -c conda-forge -c bioconda mmseqs2 
```

Alternatively, on Ubuntu, EMBOSS is available directly via `sudo apt-get install emboss` .

To install GraphPart, run
```
pip install graph-part
```
The command `graphpart` will now be available on your command line.


Alternatively, you can install GraphPart from source:
```
conda install -c bioconda emboss
git clone https://github.com/graph-part/graph-part.git
cd graph-part
pip install .
```

## Instructions

### Command line interface for FASTA data
As an example, this is a basic command for partitioning a dataset at a maximum pairwise cross-partition identity of 30% into 5 folds. The resulting partitions are balanced for equal frequencies of the class labels specified in `label=` in the [FASTA headers](#Input-format). `--threads` can be adapted according to your system and has no effect on the partitioning itself, it only affects the resource usage and processing speed.
```
graphpart needle --fasta-file netgpi_dataset.fasta --threshold 0.3 --out-file graphpart_assignments.csv --labels-name label --partitions 5 --threads 12
```

Alternatively, a train-validation-test split of the data can be made instead of folds:
```
graphpart needle --fasta-file netgpi_dataset.fasta --threshold 0.3 --out-file graphpart_assignments.csv --labels-name label --test-ratio 0.1 --val-ratio 0.05 --threads 12
```
### Python API
A tutorial notebook showcasing how to use GraphPart from within Python is included at [tutorial.ipynb](tutorial.ipynb). The tutorial also covers partitioning of small molecule data.


## Input format
GraphPart works on FASTA files with a custom header format, e.g.
```
>P42098|label=CLASSA|priority=0
MAPSWRFFVCFLLWGGTELCSPQPVWQDEGQRLRPSKPPTVMVECQEAQLVVIVSKDLFGTGKLIRPADL
>P0CL66|label=CLASSB|priority=1
MKKYLLGIGLILALIACKQNVSSLDEKNSVSVDLPGEMKVLVSKEKNKDGKYDLIATVDKLELKGTSDKN
```
Alternatively , ":" and "&nbsp;-&nbsp;" (note there are spaces on either side of the `-`) can be used as separators instead of "|". It should be taken care that the sequence identifiers themselves contain no separator symbols. The keywords `label` and `priority` can be customized by specifying the `--labels-name` and `--priority-name` arguments. Both elements of the header are optional, GraphPart can also just partition based on sequences alone, without any class balancing.   
You can find a script to convert `.csv` datasets into the custom `.fasta` format at [csv_to_fasta.py](csv_to_fasta.py)

## Output format

GraphPart produces a `.csv` file that contains the cluster assignment for each sequence. Column `cluster` contains the partition number. Removed sequences are not contained in the output file.
```
AC,priority,label-val,between_connectivity,cluster
P42098,False,0,0,0.0
Q6LEM5,False,0,0,0.0
Q9JI81,False,0,0,1.0
```


## API

Minimal command:  
```
graphpart ALIGNMENT_MODE -ff FASTAFILE.fasta -th 0.3
```

### Supported aligners

Alignment mode  | Description
----------------|----------------------
[`needle`](#needle)        | Use EMBOSS needleall to compute exact pairwise global Needleman-Wunsch (NW) identities for all sequences.
[`mmseqs2`](#mmseqs2)       | Use MMseqs2 to compute fast identities from local alignments. Use with caution for nucleotides, as there it cannot be guaranteed that MMseqs2 computes all pairwise alignments.
[`precomputed`](#precomputed)   | Use a list of precomputed identities or other similarity/distance metrics.
[`mmseqs2needle`](#mmseqs2needle) | Uses MMseqs2 to compute fast identities, followed by recomputation of NW identities of all identities between the separation threshold and an user-defined lower threshold.

### Arguments

#### Shared

Long                    | Short | Description
------------------------|-------|------------
`--fasta-file`          |`-ff`  | Path to the input fasta file, formatted according to [the input format](#input-format).
`--out-file`            |`-of`  | Path at which to save the partition assignments as `.csv`. Defaults to `graphpart_result.csv`.
`--threshold`           |`-th`  | The desired partitioning threshold, should be within the bounds defined by the metric.
`--partitions`          |`-pa`  | Number of partitions to generate. Defaults to 5.
`--transformation`      |`-tf`  | Transformation to apply to the similarity/distance metric. GraphPart operates on distances, therefore similarity metrics need to be transformed. Can be any of `one-minus`, `inverse`, `square`, `log`, `None`. See the [source](graph_part/transformations.py) for definitions. As an example, when operating with sequence identities ranging from 0 to 1, the transformation `one-minus` yields corresponding distances. Defaults to `one-minus`.
`--priority-name`       |`-pn`  | The name of the retention priority in the fasta headers. If specified, the algorithm first tries to reach the treshold by removing/moving low-priority (`0`) samples before proceeding to `1` samples. Defaults to `None`. See the [the input format section](#input-format) for an example.
`--labels-name`         |`-ln`  | The name of the label in the fasta headers. Used for balancing partitions. Defaults to `None`.
`--initialization-mode` |`-im`  | Use either slow or fast restricted nearest neighbor linkage or no initialization. Can be any of `slow-nn`, `fast-nn`, `simple`. Defaults to `slow-nn`.
`--no-moving`           |`-nm`  | By default, the removing procedure tries to relocate sequences to another partition if it finds more within-threshold neighbours in any. This flag disallows moving. In high-redundancy datasets, moving can lead to imbalanced partitions and should be disabled.
`--save-checkpoint-path`|`-sc`  | Optional path to save the computed identities above the chosen threshold as an edge list. Can be used to quickstart runs in the `precomputed` mode. Defaults to `None` with no file saved.
`--test-ratio`          | `-te` | Make a train-val-test split instead of partitions for cross-validation. Overrides `--partitions` when specified. Defaults to 0. Needs to be a multiple of 0.05.
`--val-ratio`           | `-va` |Make a train-val-test split instead of partitions for cross-validation. Overrides `--partitions` when specified. Defaults to 0. Needs to be a multiple of 0.05.

#### needle

Long                    | Short | Description
------------------------|-------|------------
`--denominator`         |`-dn`  | Denominator to use for percent sequence identity computation. The number of perfect matching positions is divided by the result of this operation. Can be any of `shortest`, `longest`, `mean`, `full`, `no_gaps`. The first three options are computed from the original lengths of the aligned sequences. `full` refers to the full length of the alignment, including gaps, and is the default. `no_gaps` subtracts gaps from the full alignment length.
`--threads`             |`-nt`  | The number of threads to run in parallel. If `-1`, will use all available resources. Defaults to 1.
`--chunks`              |`-nc`  | The number of chunks into which to split the fasta file for multithreaded alignment. Defaults to 10.
`--parallel-mode`       |`-pm`  | The Python parallelization strategy to use. `multithread` or `multiprocess`. Multiprocessing is potentially faster (especially for short sequences), but increases memory usage. Defaults to `multithread`
`--nucleotide`          |`-nu`  | Use this flag if the input contains nucleotide sequences. By default, assumes proteins.
`--triangular`          |`-tr`  | Only compute triangular of the full distance matrix. Twice as fast, but can yield slightly different results if an alignment has two different solutions with the same score, but different identities. In some cases, a pairwise identity can be slightly above the threshold in one solution, and slightly below in another (e.g A:B = 29.8%, B:A = 30.4%).
`--gapopen`             |`-gapopen`     | [10.0 for any sequence] The gap open penalty is the score taken away when a gap is created. The best value depends on the choice of comparison matrix. The default value assumes you are using the EBLOSUM62 matrix. (Floating point number from 1.0 to 100.0)
`--gapextend`           |`-gapextend`   | [0.5 for any sequence] The gap extension penalty is added to the standard gap penalty for each base or residue in the gap. This is how long gaps are penalized. Usually you will expect a few long gaps rather than many short gaps, so the gap extension penalty should be lower than the gap penalty. An exception is where one or both sequences are single reads with possible sequencing errors in which case you would expect many single base gaps. You can get this result by setting the gap open penalty to a very low value and using the gap extension penalty to control gap scoring. (Floating point number from 0.0 to 10.0)
`--endextend`           |`-endextend`   | [0.5 for any sequence] The end gap extension, penalty is added to the end gap penalty for each base or residue in the end gap. This is how long end gaps are penalized. (Floating point number from 0.0 to 10.0)
`--endweight`           |`-endweight`   | Flag. Apply end gap penalties. By default, no end gap penalties are applied.
`--endopen`             |`-endopen`     | [10.0 for any sequence] The end gap open penalty is the score taken away when an end gap is created. The best value depends on the choice of comparison matrix. The default value assumes you are using the EBLOSUM62 matrix for protein sequences. (Floating point number from 1.0 to 100.0)
`--matrix`              |`-datafile`    | This is the scoring matrix file used when comparing sequences. By default it is the file 'EBLOSUM62'. These files are found in the 'data' directory of the EMBOSS installation. If `--nucleotide`, the default is 'EDNAFULL'.


#### mmseqs2  
  
  
Long                    | Short | Description
------------------------|-------|------------
`--denominator`         |`-dn`  | Denominator to use for percent sequence identity computation. The number of perfect matching positions is divided by the result of this operation. Can be any of `shortest`, `longest`, `n_aligned`. `n_aligned` is the length of the alignment. Use this with caution, as GraphPart doesn't use coverage controls in the mmseqs2 mode. Defaults to `shortest`.
`--nucleotide`          |`-nu`  | Use this flag if the input contains nucleotide sequences. By default, assumes proteins. Use with caution! Not guaranteed to compute all pairwise alignments.
`--prefilter`           |`-pr`  | Use MMseqs2 prefiltering at the highest sensitivity instead of forcing computation of all-vs-all alignments.

#### precomputed  
  

Long                    | Short | Description
------------------------|-------|------------
`--edge-file`           |`-ef`  | Path to a comma separated file containing precomputed pairwise metrics, the first two columns should contain sequence identifiers specified in the  `--fasta-file`. This is can be used to run GraphPart with an alignment tool different from the default `needleall` and `mmseqs`.
`--metric-column`       |`-mc`  | Specifies in which column the metric is found. Indexing starts at 0, defaults to 2 when left unspecified.

#### mmseqs2needle

This mode combines `needle` and `mmseqs2`. After a first run of `mmseqs2`, all pairwise alignments with an indentity below `recompute-threshold` are computed using `needle`. Only then the partitioning is performed. Note that if `recompute-threshold` is very low, it can be faster to just run in `needle` mode. All arguments for the two modes are reused here, with the following exceptions:

Long                    | Short | Description
------------------------|-------|------------
`--recompute-threshold` | `-re` | The threshold for MMseqs2 above which alignments should be recomputed using needleall. Has to be a number lower than `--threshold`.
`--denominator-mmseqs`  | `-dnm`| Replaces `--denominator`. Applies to the mmseqs2 alignment step.
`--denominator-needle`  | `-dnn`| Replaces `--denominator`. Applies to the needle alignment step.
## Citation

    GraphPart: Homology partitioning for biological sequence analysis
    Felix Teufel, Magnús Halldór Gíslason, José Juan Almagro Armenteros, Alexander Rosenberg Johansen, Ole Winther, Henrik Nielsen
    bioRxiv 2023.04.14.536886; doi: https://doi.org/10.1101/2023.04.14.536886

## FAQ

- **How should I pick `chunks` ?**  
`chunks` should be picked so that all `threads` are utilized. Each chunk is aligned to each other chunk, so `threads` <= `chunks`*`chunks` results in full utilization.

- **I want to test multiple thresholds and partitioning parameters - How can I do this efficiently ?**  
When constructing the graph, we only retain identities that are larger than the selected `threshold`, as only those form relevant edges for partitioning the data. All other similarities are discarded as they are computed. To test multiple thresholds, the most efficient way is to first try the lowest threshold to be considered and save the edge list by specifying `--save-checkpoint-path EDGELIST.csv`. In the next run, use `graphpart precomputed -ef EDGELIST.csv` to start directly from the previous alignment result.

- **GraphPart starts with nicely balanced partitions, but after homology removal the sizes are very imbalanced.**  
By default, GraphPart tries to retain as many sequences as possible. In cases where the initialization clustering is far away from a valid solution (this happens when there are a lot of classes, with potentially small counts, and when there is high overall sequence similarity in the data), moving sequences between partitions will cause some partitions to grow large at the expense of others. You can try `--no-moving` to prevent this behaviour. 


- **I want to use a different similarity metric than sequence identity.**
GraphPart can be used with any similarity or distance metric. To do so, you need to provide a list of precomputed pairwise similarities in the `precomputed` mode. The first two columns of the file should contain the sequence identifiers specified in the `--fasta-file`. The third column should contain the similarity metric. The `--metric-column` argument can be used to specify the column index. If you want to use a similarity metric, you need to specify a transformation using `--transformation`. See the [source](graph_part/transformations.py) for definitions. As an example, when operating with sequence identities ranging from 0 to 1, the transformation `one-minus` yields corresponding distances. If your metric is a distance, you can use `--transformation None` to skip the transformation step.

- **I want to use a different alignment tool than EMBOSS needleall or MMseqs.**
This is supported by the [precomputed](#precomputed) mode. Please refer to the answer above.

- **How does the moving step decide to which partition to move a sequence to?**
After initialization of the partitions, GraphPart iteratively moves sequences between partitions and removes sequences from the data to achieve homology separation. For each sequence, we compute how many connections it has to sequences in each other partition. If there are partitions with more connections than the current partition, the sequence is moved to the partition with the maximum number of connections. If there is a tie in the number of connections, the sequence is moved to the partition that appeared first when iterating the underlying graph data structure. We do not explicitly control this order.
