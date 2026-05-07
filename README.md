:warning: This repository is under construction

# Joint Identification and Induction of Semantic Frames

This repository contains the code and resources to reproduce the experiments from the paper
"Joint Identification and Induction of Semantic Frames with Scalable Semi-Supervised Graph Clustering" (LREC 2026).


## Overview

A scalable semi-supervised clustering (SSC) pipeline that jointly performs:
- Frame Identification (FId): assign known FrameNet labels to frame-evoking elements (FEEs), a.k.a classification.
- Frame Induction (FIn): find clusters of instances that evoke the same (possibly unknown) frame, a.k.a clustering.

Key components:
- Local–global two-step clustering (per-lemma local clustering → global clustering of local clusters).
- Semi-supervised adaptation of Chinese Whispers (CW) to enforce hard must-link / cannot-link constraints from labeled training instances.
- Contextualized target representations: weighted average of masked and unmasked BERT token embeddings.
- Support for English (BFN v1.7) and German (SALSA v2.0) experiments presented in the paper, other languages are straight forward.

## Repository structure

- data/
  - instructions to download and prepare BFN (1.7) and SALSA (2.0) splits used in experiments
  - scripts to convert dataset formats to the pipeline input
- src/
  - cli.py - main entry point for running commands
  - database.py - 
  - embeddings.py — code for extracting masked/unmasked BERT embeddings and weighted averaging
  - ...
- experiments/
  - scripts to run hyperparameter tuning on dev sets and to run test evaluations
  - example configurations in JSON format for the reported best setups
- pkg/
  - third party packages for word piece re-tokenization and chinese whispers clustering
- sql/
  - sql scripts for initializing the database with helper functions and utilities
- requirements.txt — Python package requirements
- .env - configuration, i.e. database location, data location, etc.
- docker-compose.yml - compose file to start database
- dvc.yml - [DVC](https://dvc.org/) configuration to ensure data integrity and reproducibility
- Makefile - run `make <target>` for ease of access, and see example commands
- LICENSE
- CITATION
- README.md (this file)

## Getting started

<u>Prerequisites</u>
- Python 3.12+
- [Docker](https://www.docker.com/) (with docker compose enabled), or a [ParadeDB](https://www.paradedb.com/) instance
- Recommended: CUDA-enabled GPU for faster embedding extraction
- Install dependencies:
  ```
  make uv-init
  make uv-install
  ```

<u>Prepare data</u>

- Download BFN v1.7 and SALSA v2.0 as described in 
  - `data/fn1.7/README.md.`, and 
  - `data/salsa/README.md.`

- Prepare database:
  ```
  make start-db-stack
  ```
  or change the database connection string in `.env` **if you use your own ParadeDB instance**

- Run DVC pipeline to ensure data integrity and repropducibility:
  ```
  make prepare-data
  ```
  This process inserts the tokenized datasets into the database, extracts embeddings and inserts respective embeddings according to our experiments.

- ...

## Notes & limitations

- Current implementation restricts FEEs to verbal LUs only (paper limitation).
- The semi-supervised CW enforces hard constraints from labeled instances; new frame clusters remain unnamed (manual or LLM-based labeling required).
- Preliminary experiments showed larger LLM embeddings may encode excessive sentence info and hurt clustering — see paper footnote and experiments for details.

## Cite

If you use this code or follow the method, please cite the [LREC 2026 paper](https://lrec.elra.info/lrec2026-main-786):


Barteld, F., Remus, S., Anwar, S., Stawecki, J., Ziem, A., & Biemann, C. (2026). **Joint Identification and Induction of Semantic Frames with Scalable Semi-Supervised Graph Clustering**. In *Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)* (https://doi.org/10.63317/5q7o3fgim7pb).


```
@inproceedings{barteld-etal-2026-joint,
  title = {Joint Identification and Induction of Semantic Frames with Scalable Semi-Supervised Graph Clustering},
  author = {Barteld, Fabian and Remus, Steffen and Anwar, Saba and Stawecki, Julian and Ziem, Alexander and Biemann, Chris},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  year = {2026},
  pages = {10020--10030},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  doi = {10.63317/5q7o3fgim7pb}
}
```


## Contact / Contributing
- Issues and pull requests are welcome.
- For questions about reproducing experiments, open an issue with a reproducibility tag and include the dataset and config used.
- For all questions and contributions, please see the [code of conduct](CODE_OF_CONDUCT.MD).