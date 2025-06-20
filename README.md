# TimeStress: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

[**TimeStress**](https://huggingface.co/datasets/Orange/TimeStress) is a dataset designed to study the quality of the temporal representation of facts in large language models (LLMs) by analyzing their ability to distinguish between correct and incorrect factual statements contextualized with a date and formatted as a question. For example, a question might be: *"In 2011, who was the president of the USA? Barack Obama."*

The evaluation of language models is based on the principle that the probability assigned by the model to the correct answer, given the question and the date, should be higher when the date is correct compared to when it is incorrect. Mathematically, this is expressed as: 

```math
P(\text{answer} \mid \text{date}_{\text{correct}}, \text{question}) > P(\text{answer} \mid \text{date}_{\text{incorrect}}, \text{question})
```

TimeStress includes numerous correct and incorrect statements, with each date expressed in three different precisions. This allows for the evaluation of LLMs along two dimensions: by varying the date on the **timeline** and by adjusting the **precision**. An illustration of these two dimensions and a sample of TimeStress are provided below.

![image](https://github.com/user-attachments/assets/ab2aa232-feaa-4ac8-bc6c-74cf9294e618)

## Repository Content

The main goal of this repository is to reproduce the results of our paper:

[Khodja, H. A., et al. (2025). Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations. arXiv preprint arXiv:2502.01220.](https://arxiv.org/abs/2502.01220) (accepted for publication at **ACL Workshop - L2M2**)

We provide the source code and data to:

- Generate TimeStress from scratch using a Wikidata dump.
- Collect predictions from 18 studied LLMs on TimeStress.
- Analyze the behavior of LLMs to draw conclusions about the consistency of their temporal representation of facts. The figures and tables from our paper are generated in this step.

Each use case is detailed in the following two sections. Since the generation of TimeStress takes a lot of time, it war already generated and published in [Huggingface](https://huggingface.co/datasets/Orange/TimeStress). Consequently, the first section, consisting in the generation of TimeStress, can be skipped.

## Generate TimeStress from Scratch

**1. Install Python Packages**

Install the environment using the python package manager **uv** ([How to install uv?](https://docs.astral.sh/uv/getting-started/installation/)) by running the following command at the root of this repository:

```uv sync```

Wait until the packages are installed.

**2. Push Wikidata Dump to MongoDB**

**First**, install the Mongo Database and set the following environment variables:

- STORAGE_FOLDER: Set this variable to represent the path to the folder where to store intermediate files. 
- MONGO_URL: If MongoDB does not run locally with no authentification, specify its URL in this variable. Else, do nothing.

Plan **150GB** of disk storage in this folder and **150GB** for MongoDB.

**Second**, activate the `.venv` virtual environment initialized by `uv` and execute the two following script to download everything necessary for the generation of TimeStress.

```python
from wikidata_tools.wikidata import TempWikidata, WikidataPopularity, WikidataPrepStage

wd = TempWikidata("20210104", WikidataPrepStage.PREPROCESSED)
wd.build(confirm=False)
```

```python
from wikidata_tools.wikidata import WikidataPrepStage

wikipop = WikidataPopularity("20210104")
wikipop.build(confirm=False)
```

The first script downloads the Wikidata dump of January 4, 2021, preprocesses it, and pushes it to MongoDB. The second, downloads the necessary files to compute the popularity of facts and makes it also available in MongoDB.  

*These two scripts can be run in parallel to save time.*

**3. Build TimeStress**

Run the notebook situated in `experiments/temporality-context-paper/generate_dataset.ipynb` to generate **TimeStress**. This notebook contains comments to explain the build process. 

At the end of its execution, a file named `timestress.pkl` is generated next to the notebook. Open it using the following script to explore it:

```python
import pandas as pd
timestress = pd.read_pickle(PATH_TO_TIMESTRESS_PKL)
```

**NOTE**: TimeStress is available in [Huggingface](https://huggingface.co/datasets/Orange/TimeStress).

## Collection of LLMs' Predictions
If not done already, install the **necessary Python packages** by following the instructions in Section `Generate TimeStress from Scratch/Install Python Packages`.

To collect LLM's predictions on TimeStress, call the script `run.py` from the `experiments/temporality-context-paper` folder as follows:

```
uv run python run.py --model MODEL_NAME [--instruct] --experiment EXPERIMENT
```

Call this command for each `MODEL_NAME` in this list:
```
meta-llama/Llama-3.1-8B
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.1-70B
meta-llama/Llama-3.1-70B-Instruct
mistralai/Mistral-7B-v0.3
mistralai/Mistral-7B-Instruct-v0.3
mistralai/Mistral-Nemo-Base-2407
mistralai/Mistral-Nemo-Instruct-2407
google/gemma-2-2b
google/gemma-2-2b-it
google/gemma-2-9b-it
google/gemma-2-9b
google/gemma-2-27b
google/gemma-2-27b-it
apple/OpenELM-450M
apple/OpenELM-450M-Instruct
apple/OpenELM-3B
apple/OpenELM-3B-Instruct
```

and for each `EXPERIMENT` in the following list:
```
classic
explain_granularity_generalization
explain_date
```

**Important**: Specify the `--instruct` flag when the model is instruction tuned. Else, do not put the flag.

There are also other parameters that can be set such as the LLM's precision (16bit or 32bit), and the batch size; more details can be found in the script. **By default, the script uses the TimeStress dataset from Huggingface**. To use a custom one, specify its path using the `--timestress_path` argument.

**Note**: The `classic` experiment is enough to reproduce most results in the paper. The other experiments are exclusively related to *Explanation Prompts* results (see the [paper](https://arxiv.org/abs/2502.01220)).

## Analysis of LLMs' Predictions

Follow the instructions inside the notebook `experiments/temporality-context-paper/analyze_results.ipynb` to analyze the LLMs' predictions and generate most of the plots and tables in our paper.


## Having an issue with our code?

If you have a problem running our code, please let us know by opening an issue ;)

## How to cite our work?

```
@misc{khodja2025factualknowledgelanguagemodels,
      title={Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations}, 
      author={Hichem Ammar Khodja and Frédéric Béchet and Quentin Brabant and Alexis Nasr and Gwénolé Lecorvé},
      year={2025},
      eprint={2502.01220},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.01220}, 
}
```

## Licence

Look for the LICENCE.txt file at the root of this project
