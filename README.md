
## Project Report Template

> This repository serves as a template for your project reports as part of the Document Analysis lecture. To set up your project report as a webpage using GitHub Pages, simply follow the steps outlined in the next chapter.
>
>**Some Organizational Details:** Get creative with your project ideas! Just make sure they relate to Natural Language Processing and incorporate this specified dataset: [Link to data](https://huggingface.co/datasets/webis/tldr-17), [Link to paper](https://aclanthology.org/W17-4508.pdf). Submissions should be made in teams of 2-3 students. Each team is expected to create a blog-style project website, using GitHub Pages, to present their findings. Additionally, teams will deliver a lightning talk during the final lecture to discuss their project. Add all your code, such as Python scripts and Jupyter notebooks, to the `code` folder. Use markdown files for your project report. [Here](https://docs.gitlab.com/ee/user/markdown.html) you can read about how to format Markdown documents. 
>
>Have fun working on your project! 🥳

## Setup The Report Template

Follow this steps to set up your project report:

1. **Fork the Repository:** Begin by creating a copy of this repository for your own use. Click the `Fork` button at the top right corner of this page to do this.

2. **Configure GitHub Pages:** Navigate to `Settings` -> `Pages` in your newly forked repository. Under the `Branch` section, change from `None` to `master` and then click `Save`.

3. **Customize Configuration:** Modify the `_config.yml` file within your repository to personalize your site. Update the `title:` to reflect the title of your project and adjust the `description:` to provide a brief summary.

4. **Start Writing:** Start writing your report by modifying the `README.md`. You can also add new Markdown files for additional pages by modifying the `_config.yml` file. Use the standard [GitHub Markdown syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) for formatting. 

5. **Access Your Site:** Return to `Settings` -> `Pages` in your repository to find the URL to your live site. It typically takes a few minutes for GitHub Pages to build and publish your site after updates. The URL to access your live site follows this schema: `https://<<username>>.github.io/<<repository_name>>/`

***

# Beyond the Literal: Unmasking Sarcasm’s Signature on Reddit

_Group members: Emrecan Ulu, Flora Hirche

## Introduction

- set the stage
- relevant studies or work that have tackled similar issues
- main question or problem

Differentiating between the literal and the intended meaning of a text computationally remains a challenging task. 
...
The goal of this project was to automatically detect sarcasm in our Reddit dataset, identify typical characteristics of sarcasm and explore the topics that the sarcastic comments focus on. 

## Dataset

- short description of the datasets used in your project (highlight aspects that are particularly relevant to your work)

We used xxx as the training dataset for fine-tuning the sarcasm detector. It contains a balanced number of labeled sarcastic and non-sarcastic comments. ...
We used the xxx dataset to detect sarcastic comments in it and analyze them further. It contains ...

## Methods

We fine-tuned RoBERTa (Sequence Classifier) on sarcasm detection, using our labeled training dataset. In order to evaluate its performance in detecting sarcasm, we compared the fine-tuned model to a linear regression model that was trained on the same data and on the same task. We applied TF-IDF vectorization to find the most characteristic words that were used in the sarcastic comments compared to the non-sarcastic ones. In the linguistic analysis, we applied sentiment analysis to obtain the sentiment incongruity score for sarcastic and non-sarcastic comments and determined the punctuation density by xxx. For the topic modeling within the sarcastic comments, we used xxx to extract the sentence embeddings and determine their similarity, then we applied HDBSCAN clustering.

### Setup 


Outline the tools, software, and hardware environment, along with configurations used for conducting your experiments. Be sure to document the Python version and other dependencies clearly. Provide step-by-step instructions on how to recreate your environment, ensuring anyone can replicate your setup with ease:

```bash
conda create --name myenv python=<version>
conda activate myenv
```

Include a `requirements.txt` file in your project repository. This file should list all the Python libraries and their versions needed to run the project. Provide instructions on how to install these dependencies using pip, for example:

```bash
pip install -r requirements.txt
```

### Experiments

- how you conducted the experiments: detailed explanations of preprocessing steps and model training
- preprocessing: describe  data cleaning, normalization, or transformation steps you applied to prepare the dataset, along with the reasons for choosing these methods
- model training: explain the methodologies and algorithms you used, detail the parameter settings and training protocols, and describe any measures taken to ensure the validity of the models

#### Preprocessing 

##### xxx data (for fine-tuning)

We removed empty comments from the dataframe and tokenized the remaining comments. 

##### Webis-TLDR data

To prepare the data for detecting sarcasm, we transformed it into a dataframe and selected three subreddits out of the top 20 subreddits containing the most comments. To be able to analyze sarcasm in diverse contexts and styles, we chose one subreddit from each category:

1. **humor-oriented** - We chose the subreddit 'r/WTF', in which we expected a high rate of sarcasm, primarily in the context of personal stories. (The bigger subreddit 'r/funny' contained more posts than we were able to process.)
2. **political/debate-oriented** - We chose 'r/worldnews', in which we expected a high rate of sarcasm, to capture sarcasm on political and controversial topics.
3. **informational/explanatory** - We chose 'r/explainlikeimfive' as a subreddit in which we expected a diverse range of topics and a lower rate of sarcasm compared to the other selected subreddits.

We cleaned the pre-selected data by removing duplicate comments within the same subreddit, empty comments and columns that we wouldn't need in our analyses (body, normalized body, ...). 

After we labeled the data using the fine-tuned model for sarcasm detection, we did xxx to prepare the data for topic modeling.

## Results and Discussion

Present the findings from your experiments, supported by visual or statistical evidence. Discuss how these results address your main research question.

## Conclusion

Summarize the major outcomes of your project, reflect on the research findings, and clearly state the conclusions you've drawn from the study.

## Contributions

| Team Member  | Contributions                                             |
|--------------|-----------------------------------------------------------|
| Emrecan Ulu  | topic modeling, ...                                       |
| Flora Hirche | preprocessing, model fine-tuning, evaluation              |

## References

Include a list of academic and professional sources you cited in your report, using an appropriate citation format to ensure clarity and proper attribution.

