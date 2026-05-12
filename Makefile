
.ONESHELL:

SHELL:=/bin/bash

#
## Sample targets for users of the app
#

user-start-db-stack:
	docker compose up -d

user-init-db:
	@echo "create database ssc4frames" | docker compose exec -T db psql postgresql://root:root@localhost/ssc4frames
	docker compose exec app python -m ssc4frames data init-db-tables
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
	uv python pin 3.12
	uv venv --python 3.12

dev-uv-install:
	uv pip install --editable .

dev-uv-install-requirements:
	uv pip install -U -r requirements-dev.txt

dev-init-db:
	@echo "create database ssc4frames" | docker exec -i ssc4framesdb psql postgresql://root:root@localhost/ssc4frames
# 	uv run -m ssc4frames data init-db-tables
	docker exec ssc4framesdev uv run -m ssc4frames data init-db-tables
	cat ./sql/0_clean.sql ./sql/1_views.sql ./sql/2_functions.sql | docker exec -i ssc4framesdb psql postgresql://root:root@localhost/ssc4frames

dev-uv-activate-venv:
	@echo "please run manually 'source .venv/bin/activate'"

dev-uv-deactivate-venv:
	@echo "please run manually 'deactivate'"

dev-uv-run:
	@echo "please run any command manually 'uv run -m ssc4frames'"

dev-uv-build:
	uv pip install -U build
	uv run -m build

dev-uv-lock-requirements:
	uv pip compile requirements-dev.txt -o requirements-gen.txt

dev-uv-add-dev-requirements:
	uv add -r requirements-dev.txt

