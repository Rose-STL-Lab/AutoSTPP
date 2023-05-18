.PHONY: delete_aim_run clean data lint requirements yaml test help
.PHONY: run_cuboid run_stpp
.PHONY: upload_results download_results clean_results view_local_results

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL = /bin/bash
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = autoint
PYTHON_INTERPRETER = python3
RESULT_DIR = results/
BUCKET_NAME = autoint
S3_PATH = s3://${BUCKET_NAME}/
RESULT_PATH = "s3://${BUCKET_NAME}/results/"
MODEL_PATH = "s3://${BUCKET_NAME}/models/"
CONFIG_PREFIX = autoint-configs

export PYTHONPATH = src

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

#################################################################################
# Cuboid related                                                                #
#################################################################################

run_cuboid:
	python src/experiment/run_cuboid.py -c configs/prodnet_cuboid_sine.yaml

run_cuboid_normal:
	python src/experiment/run_cuboid.py -c configs/prodnet_cuboid_normal.yaml

#################################################################################
# STPP related                                                                  #
#################################################################################

# List all the config files that end with 'stpp.yaml'
CONFIG_FILES := $(wildcard configs/*stpp.yaml)

config ?= autoint_stpp
run_stpp:
	python src/experiment/run_stpp.py -c configs/$(config).yaml

#################################################################################
# Kubernetes related                                                            #
#################################################################################

postfix ?=
## Update kubectl config, infuse config to template, and delete dangling pods older than 24 hours
update_kubeconfig:
	@kubectl get configmaps -o json \
		| jq '.items[] | select((.metadata.creationTimestamp | fromdateiso8601) < (now - 86400) and (.metadata.ownerReferences == null)) | .metadata.name' \
		| grep $(CONFIG_PREFIX) | xargs -I {} kubectl delete configmap {} --ignore-not-found=true
	@kubectl delete configmap $(CONFIG_PREFIX)$(postfix) --ignore-not-found=true
	@kubectl create configmap $(CONFIG_PREFIX)$(postfix) --from-file=configs/
	@yq -i '(.volumes[] | select(.name == "$(CONFIG_PREFIX)")).configMap.name |= "$(CONFIG_PREFIX)$(postfix)"' kube/template.yaml
# @kubectl create configmap autoint-s3cfg --from-file=/home/ubuntu/.s3cfg

ITEMS = item1 item2 item3
test:
	$(foreach item,$(ITEMS), \
		[ $(item) == "item2" ] && \
			echo "Skipping $(item)" \
		|| \
			echo "Processing $(item)"; \
	)

yaml: 
	@yq '((.. | select(has("command"))).command |= load("kube/startup.yaml") + .)' kube/$(source).yaml > kube/$(dest).yaml
	@yq -i '(.. | select(has("command"))).command[-2] += " " + (.. | select(has("command"))).command[-1]' kube/$(dest).yaml
	@yq -i '(.. | select(has("command")).command)[-1] = ""' kube/$(dest).yaml
	@yq -i 'del(.. | select(length == 0))' kube/$(dest).yaml
	@yq -i '(.. | select(has("file"))) |= load(.file) *d . | del(.. | select(has("file")).file)' kube/$(dest).yaml

interactive: source = interactive
interactive: dest = build/run_interactive
interactive: update_kubeconfig yaml
	@kubectl delete -f kube/$(dest).yaml --ignore-not-found=true
	@kubectl create -f kube/$(dest).yaml
	$(eval POD_NAME=$(shell yq eval '.metadata.name' kube/$(dest).yaml))
	@echo "Waiting for pod $(POD_NAME) to be ready"
	@kubectl wait --for=condition=Ready pod/$(POD_NAME) --timeout=1h
	@kubectl port-forward pod/$(POD_NAME) 1550:1551 --address 0.0.0.0

job ?= tune_cuboid
source ?= $(job)
dest ?= build/run_$(job)
job: update_kubeconfig yaml
	$(eval JOB_NAME=$(shell echo "$(PROJECT_NAME)-$(job)$(postfix)" | tr '_' '-'))
	@yq -i eval '.metadata.name = "$(JOB_NAME)"' kube/$(dest).yaml
	@kubectl delete job $(JOB_NAME) --ignore-not-found=true
	@kubectl create -f kube/$(dest).yaml


BATCH_NAMES = sthp0 sthp1 sthp2 stscp0 stscp1 stscp2 earthquakes_jp covid_nj_cases
# Kube config filename
job_name ?= stpp
# Lightning config filename
config_fn ?= autoint_copula_stpp
# Seed list
SEEDS ?= 1551 1552 1553
## Launch jobs for all datasets
batch_job: 
	@sed -i 's,configs/[^[:space:]]*.yaml,configs/$(config_fn).yaml,g' kube/$(job_name).yaml	
	$(foreach name, $(BATCH_NAMES), \
		([ $(name) == "earthquakes_jp" ] && [ $(config_fn) == "deep_stpp" ]) && ( \
			yq -I4 -i '((.. | select(has("constrain_b"))).constrain_b |= "clamp")' configs/$(config_fn).yaml; \
			yq -I4 -i '((.. | select(has("s_min"))).s_min |= 1.0e-3)' configs/$(config_fn).yaml \
		) || ( \
			yq -I4 -i '((.. | select(has("constrain_b"))).constrain_b |= false)' configs/$(config_fn).yaml; \
			yq -I4 -i '((.. | select(has("s_min"))).s_min |= 1.0e-4)' configs/$(config_fn).yaml); \
		yq -I4 -i '((.. | select(has("name"))).name |= "$(name)")' configs/$(config_fn).yaml && \
		$(eval postf := $(subst _,-,$(name))) \
		$(eval pref := $(subst _,-,$(config_fn))) \
		$(foreach seed, $(SEEDS), \
			yq -I4 -i '(.seed_everything |= $(seed))' configs/$(config_fn).yaml; \
			if kubectl get job $(PROJECT_NAME)-$(job_name)-$(pref)-$(postf)-$(seed) > /dev/null 2>&1; then \
				echo "Job $(PROJECT_NAME)-$(job_name)-$(pref)-$(postf)-$(seed) already exists"; \
			else \
				$(MAKE) job job=$(job_name) postfix=-$(pref)-$(postf)-$(seed); \
			fi; \
		) \
	)

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

## Unlock aim run (for deletion) given prefix
unlock_aim_run: 
	$(eval LOCKS=$(shell find .aim -name '*.softlock'))
	@echo $(LOCKS)
	@printf "This target will delete the files above, please type 'yes' to proceed: "
		@read ans; \
		if [ "$$ans" != "yes" ]; then \
			echo "Not deleted"; \
			exit 1; \
	fi
	@$(RM) -rf $(LOCKS); \
	echo "Deleted $(LOCKS)"

## Deactivate all aim runs
toggle_aim:
	@if [ -d .aim ]; then \
		echo "Renamed to .aim.old"; \
		mv .aim .aim.old; \
		aim init; \
	else \
		if [ -d .aim.old ]; then \
			echo "Renamed to .aim"; \
			mv .aim.old .aim; \
		fi \
	fi

#################################################################################
# S3 related                                                                    #
#################################################################################

## Check file existence on S3
fd:
	@s3cmd ls ${S3_PATH} --recursive | grep $(file)

## Upload Results to S3
upload_results:
	@mkdir -p ${RESULT_DIR}
	@python src/zip_results.py
	@rm -rf ${RESULT_DIR}.aim
	@s3cmd put --skip-existing ${RESULT_DIR}* ${RESULT_PATH}

## Upload all model checkpoints to S3
upload_models:
	@find . -type f | grep ckpt | rev | cut -d'/' -f4- | rev | xargs -I {} s3cmd sync --recursive {} ${MODEL_PATH}

## Download all model checkpoints from S3
download_models:
	@s3cmd get --skip-existing --recursive ${MODEL_PATH} .aim

## Remove locally deleted results from S3
sync_results:
	@printf "This target will delete unseen remote archive, please type 'yes' to proceed: "
		@read ans; \
		if [ "$$ans" != "yes" ]; then \
			echo "Not deleted"; \
			exit 1; \
	fi
	@rm -rf ${RESULT_DIR}.aim
	@s3cmd sync --skip-existing --delete-removed ${RESULT_DIR} ${RESULT_PATH}

## Download Results from S3
download_results:
	@mkdir -p ${RESULT_DIR}
	@s3cmd get --skip-existing --recursive ${RESULT_PATH} ${RESULT_DIR}

## View Latest Downloaded Results
view_results: download_results
	@rm -rf ${RESULT_DIR}.aim
ifeq ($(file),)
	@unzip `ls -t ${RESULT_DIR}aim* | head -1` -d ${RESULT_DIR}.aim
else
	@unzip $(file) -d ${RESULT_DIR}.aim
endif
	@aim up --port 1551 --host 0.0.0.0 --repo ${RESULT_DIR}.aim/

repo ?= .aim/
## View Latest Local Results
local_results: 
	@aim up --port 1551 --host 0.0.0.0 --repo $(repo)

## Delete all remote results; Run with caution
clean_results:
	@printf "This target will delete all remote archive, please type 'yes' to proceed: "
		@read ans; \
		if [ "$$ans" != "yes" ]; then \
			echo "Not deleted"; \
			exit 1; \
	fi
	@rm -rf ${RESULT_DIR}/.aim
	@s3cmd rm ${RESULT_PATH} --recursive

## Set up python interpreter environment
create_environment:
	conda create --name autoint --file conda-lock.yml
	conda activate autoint
	poetry install

## Test python environment is setup correctly
test_environment:
	@$(PYTHON_INTERPRETER) test_environment.py
	@poetry check
	@echo ">>> Poetry is setup correctly!"
	
#################################################################################
# AutoInt Tests                                                                 #
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
