# Specify the shell to use for all commands
SHELL := /bin/bash

.ONESHELL: $(MAKECMDGOALS)

# all targets are phony, i.e. they always run, even if the "target" file exists
.PHONY: $(MAKECMDGOALS)

# all targets should be silent
.SILENT: $(MAKECMDGOALS)

#
## Sample targets for users of the app
#
user-start-db-stack:
	docker compose up -d

user-init-db:
	@echo "create database ssc4frames" | docker compose exec -T db psql postgresql://root:root@localhost/ssc4frames
	docker compose exec app ssc4frames data init-db-tables
	cat ./sql/0_clean.sql ./sql/1_views.sql ./sql/2_functions.sql | docker compose exec -T db psql postgresql://root:root@localhost/ssc4frames

user-attach-dockerapp:
	docker compose exec app bash

user-dvc-pull:
	@echo -n "Run this target within the app container or with an installed version of SSC4Frames. Enter 'y' if you want to proceed: " \
		&& read ans \
		&& [ $${ans:-'N'} = 'y' ] \
		&& dvc pull

user-prepare-data:
	@echo -n "Run this target within the app container or with an installed version of SSC4Frames. Please see instructions in ./data/fn1.7 and ./data/salsa to prepare the data. Enter 'y' if you want to proceed: " \
		&& read ans \
		&& [ $${ans:-'N'} = 'y' ] \
		&& dvc repro

#
## Sample targets for development purposes
#
dev-uv-init:
	python -m pip install -U uv
	uv venv

dev-uv-install:
	uv pip install --editable .

dev-uv-install-requirements:
	uv pip install -U -r requirements-dev.txt

dev-init-db:
	@echo "create database ssc4frames" | psql postgresql://root:root@db/ssc4frames
	ssc4frames data init-db-tables
	cat ./sql/0_clean.sql ./sql/1_views.sql ./sql/2_functions.sql | psql postgresql://root:root@db/ssc4frames

dev-uv-activate-venv:
	@echo "please run manually 'source .venv/bin/activate'"

dev-uv-deactivate-venv:
	@echo "please run manually 'deactivate'"

dev-uv-run:
	@echo "please run any command manually 'uv run -m ssc4frames'"

dev-uv-build:
	uv pip install -U build
	uv run -m build

dev-requirements-no-torch:
	uv export --quiet \
	  --format requirements-txt \
    --no-hashes \
    --no-dev \
    --no-emit-project \
    --no-editable \
    --prune torch \
    --prune torchvision \
    -o requirements-no-torch.txt

dev-deploy: dev-requirements-no-torch
	@echo -n "Have you commited all relevant changes, updated the version tag in pyproject.toml and added your changes in CHANGLOG.md? Enter 'y' if you want to proceed: " \
		&& read ans \
		&& [ $${ans:-'N'} = 'y' ] \
		&& uv lock \
		&& git status \
		&& git commit pyproject.toml requirements-no-torch.txt uv.lock CHANGELOG.md -m 'bump version' \
		&& echo -n "Push commmits? Enter 'y' if you want to proceed: " \
		&& read ans \
		&& [ $${ans:-'N'} = 'y' ] \
		&& git push

deploy: dev-deploy



