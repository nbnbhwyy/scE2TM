# $scE^2TM$: Toward Interpretable Single-Cell Embedding via Topic Modeling

The full description of $scE^2TM$ and its application on published single cell RNA-seq datasets are available.


The repository includes detailed installation instructions and requirements, scripts and demos.


## 1 Schematic overview of $scE^2TM$.

![](Flow.jpg)

**(a)** To better collaborate the information of different modalities, clusters and topic heads are trained based on mutually refined neighborhood information by encouraging consistent clustering assignments of mutual nearest neighbors of the corresponding cells of different modalities in the embedding space.. 
**(b)** ECR clusters gene embeddings $\mathbf{g}_{j}$ ($\textcolor{blue}{\bullet}$) as samples and topic embeddings $\mathbf{t}_{k}$ ($\textcolor{red}{\star}$) as centers with soft-assignment $\pi_{\epsilon, j k}^{*}$. 
Here ECR pushes $\mathbf{g}_{1}$ and $\mathbf{g}_{2}$ close to $\mathbf{t}_{1}$ and away from $\mathbf{t}_{3}$ and $\mathbf{t}_{5}$.
**(c)** Sparse linear decoders learn topic embeddings and gene embeddings as well as sparse topic-gene dependencies during reconstruction, thus ensuring model interpretability.
## 2 Installation
Create a new python environment.
```bash
conda create --name  scE2TM_env python=3.8.8
conda activate scE2TM
```

Install the dependencies from the provided requirements.txt file.
```bash
pip install -r requirements.txt
```

## 3 Usage

### Data format

$scE^2TM$ requires the input of cell-by-cell gene matrices, external embedding of cells, and true cell type information in .CSV object format.

The true cell type information is only used for prediction accuracy assessment.

We provide default data (Wang) for users to understand and debug the $scE^2TM$ code.


### Training

```bash
python run.py
```
### Tutorial

We provide three tutorials in the `tutorial` directory that introduce the usage of $scE^2TM$ and reproduce the main quantitative results of our paper:

- [Clustering and Interpretable Evaluation](tutorial/Clustering%20and%20Interpretable%20Evaluation)
- [Pathway Enrichment](tutorial/Pathway%20Enrichment)
- [Topic gene embedding](tutorial/Topic%20gene%20embedding)

## Reference

If you use $scE^2TM$ in your work, please cite

## License

This project is licensed under the MIT License.
