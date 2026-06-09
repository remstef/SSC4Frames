# CHANGES to the SSC4Frames project

### Update version - 06/09/2026 - 1
  - restructure cli commands - outsource subcommands into their own files

---

### Update version - 06/03/2026
  - update transformers version and remove obsolete wp_retok

---

### Update version - 06/03/2026
  - updated ptgcl version that fixed python syntax warnings for latex code
  - ignore subprocess warnings (git revision hash)
  - minor cli fixes

---

### Update version - 06/03/2026
  - added several sub commands, which allow to view the dataset instances with embeddings and retrieve clusters and their cluster embeddings, and retreive instances with their aggregated embeddings:
    - data instances 
    - clustering list
    - clustering info
    - clustering clusters
    - clustering instances

--

### Update version - 06/02/2026
  - added embeddings support for 'clustering get' command

--

### Update version - 05/21/2026
  - fix potential bugs in cli

--

### Update version - 05/19/2026 - 3
  - fix bug in cli.get_experiment_from_ctxobj

--

### Update version - 05/19/2026 - 2
  - minor cli bugfix

--

### Update version - 05/19/2026 - 1
  - simplified interacting with experiments on cli
  - updated dvc outputs in dvc.lock

--

### Update version - 05/19/2026
  - added version command

--

### Update version - 05/13/2026 - 4
  - added script exection instead of python module exectuion
  - added script for hyperparameter test generation
  - added hyperparameter tuning on dev sets configurations
  - added experiments dvc.yaml pipeline

--

### Update version - 05/13/2026 - 3
  - bump version to test gh actions docker build workflow 
  - bugfix typo in gh actions

--

### Update version - 05/13/2026 - 2
  - bump version to test gh actions docker build workflow 
  - updated uv.lock

---

### Update version - 05/13/2026 - 1
  - bump version to test gh actions docker build workflow 
  - updated github actions

---

### Update version - 05/13/2026
  - add docker app container
  - added more documentation
  - add github actions
  - add cli support for clusterings

---

### Initialized project - 05/11/2026

---
