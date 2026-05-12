
.ONESHELL:

SHELL:=/bin/bash

start-db-stack:
	docker compose up -d

uv-init:
	python -m pip install -U uv
	uv python pin 3.12
	uv venv --python 3.12

uv-install:
	uv pip install --editable .

uv-install-devrequirements:
	uv pip install -U -r requirements-dev.txt

init-db-dev:
	@echo "create database ssc4frames" | docker exec -i ssc4framesdb psql postgresql://root:root@localhost/ssc4frames
# 	uv run -m ssc4frames data init-db-tables
	docker exec ssc4framesdev uv run -m ssc4frames data init-db-tables
	cat ./sql/0_clean.sql ./sql/1_views.sql ./sql/2_functions.sql | docker exec -i ssc4framesdb psql postgresql://root:root@localhost/ssc4frames

init-db:
	@echo "create database ssc4frames" | docker compose exec -T db psql postgresql://root:root@localhost/ssc4frames
	docker compose exec app python -m ssc4frames data init-db-tables
	cat ./sql/0_clean.sql ./sql/1_views.sql ./sql/2_functions.sql | docker compose exec -T db psql postgresql://root:root@localhost/ssc4frames

attach-dockerapp:
	docker compose exec app bash

uv-activate-venv:
	@echo "please run manually 'source .venv/bin/activate'"

uv-deactivate-venv:
	@echo "please run manually 'deactivate'"

uv-run:
	@echo "please run any command manually 'uv run -m ssc4frames'"

uv-build:
	uv pip install -U build
	uv run -m build

uv-lock-requirements:
	uv pip compile requirements-dev.txt -o requirements-gen.txt

uv-add-dev-requirements:
	uv add -r requirements-dev.txt
	
prepare-data:
	@echo -n "Please see instructions in ./data/fn1.7 and ./data/salsa to prepare the data. Enter 'y' if you want to proceed: " \
		&& read ans \
		&& [ $${ans:-'N'} = 'y' ] \
		&& dvc repro
