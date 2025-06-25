# Tweet Project

This project demonstrates a simple text classification pipeline built with scikit-learn. The dataset can be found in `data/tweets.csv` and the source code lives in `src/`.

## Installation

Install the dependencies then run the main script:

```bash
pip install -r requirements.txt
python main.py
```

To run the unit tests locally:

```bash
pytest -v
```

## Docker

A `Dockerfile` is provided to execute the project in an isolated environment.

Build the image from the project root:

```bash
docker build -t tweet_project .
```

Run the main script inside the container (default command):

```bash
docker run --rm tweet_project
```

Run the test suite with pytest:

```bash
docker run --rm tweet_project pytest -v
```

