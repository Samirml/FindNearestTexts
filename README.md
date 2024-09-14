# Interface for searching for similar texts from the corpus


## Description

This system was developed for use as one of the modules of the question-answer system within the company. A system can find similar news excerpts of text from a corpus of documents. It can be trained on any corpus of documents and used to quickly search for relevant documents in corporate structures.
The project implements: a pipeline for learning a model based on bidirectional recurrent neural networks for the task of classifying named entities (NER) (based on a dataset https://www.kaggle.com/datasets/juliangarratt/conll2003-dataset), as well as a prototype interface for searching for similar texts based on prompt from the user. The learning pipeline is implemented based on LSTM written in Pytorch. The interactive interface is implemented on FASTAPI using Jinja2Templates. 
The proposed prototype of the interactive interface implements an error handling system in case of receiving incorrect input from the user. 


## Main technologies used
- **Python**
- **PyTorch**
- **Uvicorn** 
- **NumPy**
- **Tokenizers** 
- **Pydantic** 
- **FastAPI**
- **Pandas**
- **Jinja2**
- **Huggingface**
  
## Scheme of the program operation

The program is divided into  logical parts:

1. Data Preprocessing (including the markup of named entities in the text (correct preparation of the dataset))
2. Training of the neural network
3. Saving model weights
4. Applying model to create search system
5. APP creation

**The first** part includes the preprocessing of the dataset for deep learning tasks. Based on the BIO labeling (Beginning, Inside, Outside), each word is assigned a specific label. And each entity will have a set of labels corresponding to it if that entity consists of several words. 
To tokenize entities, a tokenizer from the BERT model based on the WordPiece tokenizer was used. Using the functions inside the module data.py the number of tokens is aligned with the number of labels for all texts in the batch.
**The second** part was devoted training the model for the task of multiclass classification using a bidirectional LSTM neural network.
In **The third** part we save the model for later use.
**The fifth** part was devoted to the creation of a search engine for similar texts. A number of modules have been implemented for this purpose (extract_keywords.py , find_nearest_texts.py ), in which named entities are highlighted from the incoming text. Based on the selected entities, using the binary measure of similarity of Jacquard (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html), we find texts in which the greatest coincidence with the current text is based on the selected words. 
**The sixth** part is dedicated to the creation of a web interface. Using FastAPI as well as an HTML markup template (Jinja2Templates), a prototype of the interface was created, where the system outputs 5 similar texts to the user's request. The input data format is partially regulated via Pydantic. Since the trained model is not trained to find similar texts for any user request, the system provides input error handling, namely:
1. If the user has entered a text with a length of less than 10 characters, the system will output: "There is too little information. Please, complete the text!". This must be done because small texts often do not reflect the meaning of the user's request.
2. If the user's text is not in the document corpus, the system will display the error: "Warning: The recommendation system might not be accurate for this text as it has low similarity to the training data.". Since the model was trained only on one thematic corpus of documents (on news), searching for similar texts for any search query will lead to poor quality of search results.

If you want to build a similar system, but to search for relevant documents in response to one request (document), then you should leave the architecture the same, but train the model on a different dataset.

In order to launch a project, you need to:
1. Clone repository
```bash
git clone https://github.com/Samirml/FindNearestTexts
cd find-similar-texts
```
2. Create a virtual environment and activate it
Linux:
```bash
python3 -m venv env
source env/bin/activate
```
Windows
```bash
python -m venv env
env\Scripts\activate
```
3. Install dependencies
```py
pip install -r requirements.txt
```
4.Project Launch (to train and generate images)
if you want to re-train the model but with different parameters aand implement a different architecture, then change the architecture in the file src/model_upgrade.py and then 
launch it:
```bash
python src/train.py
```
If you want to use trained model only, then clone the repository and write:
```bash
uvicorn use_pretrained_model.find_nearest_texts:app --reload
```
the application will be launched on your computer and you will see this interface:

![Снимок экрана от 2024-09-14 21-31-15](https://github.com/user-attachments/assets/c243d743-b854-4101-a8d4-6c482a17a1d7)

Then try to write some text and a system will output similar texts.
But due to the limitations of the model described above, here is a list of suggestions for which the system will find relevant similar texts:
1. Tallinna Pank, one of the largest banks in Estonia, made a 11 - month 1996 net profit of 46. 6 million kroons, the bank said on Friday.
2. Nader Jokhadar had given Syria the lead with a well - struck header in the seventh minute.
3. Italy recalled Marcello Cuttitta
4. Japan started the second half brightly but Bitar denied them an equaliser when he dived to his right to save Naoki Soma's low drive in the 53rd minute.
5. Wasim Akram b Harris 4
6. Susi Susanti ( Indonesia ) beat Han Jingna ( China ) 11 - 5 11 - 4.
7. But China saw their luck desert them in the second match of the group, crashing to a surprise 2 - 0 defeat to newcomers Uzbekistan.

## License
The idea of the task was taken from Karpov.Courses (https://karpov.courses/deep-learning?_gl=1*gvc6ll*_ga*NDI1MzY4NTU3LjE3MjM5NzU4OTE.*_ga_DZP7KEXCQQ*MTcyNTg3MzAyNi4xMTYuMC4xNzI1ODczMDI2LjYwLjAuMA..).

## Authors and contacts
To contact the author, write to the following email: samiralzgulfx@gmail.com
