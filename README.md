# Yubo test

Description courte et informative du projet.

## Description

In this test we will explore how build an api who is using wisely  async for the sake of parallel prediction 

## Fonctionnalités

Get tag classification pour every jpeg in the repo Python_Engineer/test_images/

## Dependencies 

- poetry
- docker
- docker-compose
## Installation

Running all service first

```bash
docker-compose up --build
```

Then create de virtual env with poetry to running the  test script

```bash
poetry install
```


## Run

from the general repository active the poetry venv

```bash
poetry shell
```

then run the testing script: 

```bash
python tests/testing_script.py
```