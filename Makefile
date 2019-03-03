ENV_NAME=computer-vision-examples

create-conda:
	conda env create -f environment.yml -n $(ENV_NAME)

update-conda:
	conda env update -f environment.yml -n $(ENV_NAME)

remove-conda:
	conda env remove -y -n $(ENV_NAME)
