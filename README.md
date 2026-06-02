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
- examples/ - example configuration files to run a clustering
- experiments/
  - configurations to run hyperparameter tuning on dev sets and to run test evaluations
  - example configurations in JSON format for the reported best setups
  - once computed, results will be stored here
- pkg/
  - third party packages for word piece re-tokenization and chinese whispers clustering
- sql/
  - sql scripts for initializing the database with helper functions and utilities
- pyproject.toml - Python package configuration
- uv.lock — Python package configuration locked for the current version
- .python-version — pin the Python version 
- requirements-dev.txt — Python package requirements (for development purposes)
- requirements-gen.txt — Python package requirements locked for reference purposes
- .env - configuration, i.e. database location, data location, etc.
- Dockerfile - to build the ssc4frames image
- docker-compose.yml - compose file to start the container stack (database, database admin interface, ssc4frames app container)
- dvc.yml - [DVC](https://dvc.org/) pipeline configuration
- dvc.lock - created by the [DVC](https://dvc.org/) pipeline to ensure data integrity and reproducibility
- Makefile - run `make <target>` for ease of access, and see example commands
- LICENSE - this software is provided under the Apache v2 License
- CHANGELOG.md - keep track of changes across versions
- CITATION.md - how to cite this repository and paper 
- CODE_OF_CONDUCT.MD - please follow the code of conduct for issues and contributions
- README.md (this file)


## Getting started

<u>Prerequisites</u>
- Python 3.12+
- [Docker](https://www.docker.com/) (with docker compose enabled), or a [ParadeDB](https://www.paradedb.com/) instance
- Recommended: CUDA-enabled GPU for faster embedding extraction
- The [Makefile](./Makefile) shows example commands and typical use case targets and is split into dev and user experiences (with the prefix `-user` and `-dev`). The `user` experience expects no installation, the `dev` experience expects [dev containers](https://code.visualstudio.com/docs/devcontainers/containers) or a python dev environment.
- Prepare database:
  ```
  make user-start-db-stack
  make user-init-db

  or as dev

  make dev-init-db # (assumes that the docker db container has been started)
  ```
  or change the database connection string in `.env` **if you use your own ParadeDB instance**

<u>Prepare data</u>

- Download BFN v1.7 and SALSA v2.0 as described in 
  - `data/fn1.7/README.md.`, and 
  - `data/salsa/README.md.`

- Prepare dataset. Run DVC pipeline to ensure data integrity and repropducibility:
  ```
  make user-attach-dockerapp
  # in the docker container run 
  make user-prepare-data

  or as dev

  make user-prepare-data:
  ```
  This process inserts the tokenized datasets into the database, extracts embeddings and inserts respective embeddings according to our experiments.  

<u>Run clusterings and get results</u>
- see [default configuration](./examples/default-with-comments.jsonc)
- see [sample configurations](./examples)
- Commands
  ```
  # run default configuration
  ssc4frames clustering run

  # run specific configuration in json
  ssc4frames clustering run examples/fid_cw_bfn.json

  # run default configuration with ovverrides
  ssc4frames clustering run '{"data":{"dataset":"fn1.7-sample"}}'
  # run default configuration with ovverrides, don't wait for key confirmation
  ssc4frames clustering run --no-wait '{"data":{"dataset":"fn1.7-sample"}}'
  # run default configuration with ovverrides, don't wait for key confirmation, skip the test set
  ssc4frames clustering run --no-wait '{"data":{"dataset":"fn1.7-sample", "splits":["train", "dev"], "testsplits":["dev"]}}'

  # get frame instances joined with embeddings
  ssc4frames data instances -b <batchsize> -e <embedding-model> -e <embedding-model> <datasetsplitname>
  ssc4frames data instances -b 10 -e bert-base-uncased -e bert-base-uncased-masked fn1.7-default 

  # get infos about one or many clusterings using their clustering ids
  ssc4frames clustering info <clustering-id> ... <clustering-id>
  ssc4frames clustering info 1 3 4 

  # list the clusters of a clustering
  ssc4frames clustering clusters -b <batchsize> <clustering-id>
  ssc4frames clustering clusters -b 10 4
  # list the clusters of a clustering with the clusterembeddings
  ssc4frames clustering clusters -e -b <batchsize> <clustering-id>
  ssc4frames clustering clusters -e -b 10 4

  # get the clustered instance assignments
  ssc4frames clustering instances <clustering-id>
  # get results for the entire dataset including unassigned instances (e.g. for the test set if omitted during clustering)
  ssc4frames clustering instances -a <clustering-id>

  ```

<u>Run experiments</u>
- Run the following commands inside the docker app container (`make user-attach-dockerapp`) or within the virtual environment (`make dev-uv-init dev-uv-activate-venv`)
- Update files from your local dvcstore (./dvcstore)
```
dvc pull
```
- Reproduce paper result tables:
```
dvc repro -R .
```
- This runs the DVC pipelines defined in `./dvc.yaml` (i.e. data preparation) and `./experiments/dvc.yaml` (hyperparameter tunixng + final test runs)
- Push changes to your local dvc store (./dvcstore)
```
dvc commit 
dvc push
```
- This avoids recomputing results if they were already computed


## Notes & limitations

- Current implementation restricts FEEs (frame evoking elements) to verbal LUs only (paper limitation).
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