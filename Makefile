.PHONY: delete_aim_run clean wandb data lint requirements 
.PHONY: run_cuboid run_stpp
.PHONY: upload_results download_results clean_results view_local_results

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = autoint
PYTHON_INTERPRETER = python3
RESULT_DIR = results/
BUCKET_NAME = autoint
DESTINATION_PATH = "s3://${BUCKET_NAME}/results/"

export PYTHONPATH = src

run_cuboid:
	python src/experiment/run_cuboid.py -c configs/prodnet_cuboid_sine.yaml

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Update kubectl config
update_kubeconfig:
	kubectl delete configmap autoint-src-tune --ignore-not-found=true
	kubectl create configmap autoint-src-tune --from-file=src/tune/
	kubectl delete configmap autoint-configs --ignore-not-found=true
	kubectl create configmap autoint-configs --from-file=configs/

## Toggle wandb
wandb:
	if grep -q "wandb.init(mode=\"disabled\")" test/conftest.py; then \
		sed -i 's/wandb.init(mode="disabled")/wandb.init(project=pytest.fn, entity="point-process", config=wandb_config)/' test/conftest.py; \
		echo "wandb enabled"; \
	else \
		sed -i 's/wandb.init(project=pytest.fn, entity="point-process", config=wandb_config)/wandb.init(mode="disabled")/' test/conftest.py; \
		echo "wandb disabled"; \
	fi

## Make Dataset
data: test_environment
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 --max-line-length=120 --ignore=E402,E731,F541,W291,E122,E127,F401,E266,E241,C901,E741,W293,F811,W504 src

#################################################################################
# Aim-stack related                                                             #
#################################################################################

## Upload Results to S3
upload_results:
	@mkdir -p ${RESULT_DIR}
	@python src/zip_results.py
	@rm -rf ${RESULT_DIR}.aim
	@s3cmd put --skip-existing ${RESULT_DIR}* ${DESTINATION_PATH}

## Download Results from S3
download_results:
	@mkdir -p ${RESULT_DIR}
	@s3cmd sync ${DESTINATION_PATH} ${RESULT_DIR}

## View Latest Downloaded Results
view_results: download_results
	@rm -rf ${RESULT_DIR}.aim
	@unzip `ls -t ${RESULT_DIR}aim* | head -1` -d ${RESULT_DIR}.aim
	@aim up --port 1551 --host 0.0.0.0 --repo ${RESULT_DIR}.aim/

## View Latest Local Results
local_results: 
	@aim up --port 1551 --host 0.0.0.0 --repo .aim/

## Delete all local and remote results; Run with caution
clean_results:
	@printf "This target will delete all remote and local archive, please type 'yes' to proceed: "
		@read ans; \
		if [ "$$ans" != "yes" ]; then \
			echo "Not deleted"; \
			exit 1; \
	fi
	@rm -rf ${RESULT_DIR}
	@s3cmd rm ${DESTINATION_PATH} --recursive

## Set up python interpreter environment
create_environment:
	conda create --name autoint --file conda-linux-64.lock
	conda activate autoint
	poetry install

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py
	poetry check
	
#################################################################################
# Autoint Tests                                                                 #
#################################################################################

test_1d_sine:
	poetry run python -m pytest -s test/test_autoint_1d_sine.py

test_speed_benchmark:
	poetry run python -m pytest -s test/test_autoint_speed_benchmark.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
