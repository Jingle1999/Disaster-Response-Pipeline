# Disaster-Response-Pipeline

Disaster Response Pipeline - Identifying messages that matter out of social media content following a disaster (Udacity Nanodegree Data Science Program):

“Following a Disaster, typically you’ll get millions and millions of communications either direct or via social media right at the time when disaster response organisations have the least capacity to filter and then pull out the messages which are the most important. And often one of every thousand messages might be relevant to the data response professional. In order to respond appropriately we need to filter the message and need to find the messages that matter to the different organisations as they take care of different problems; one organisation will take care about water, another about medical aid, etc.” (Udacity Datascience Nanodegree Program – introduction to the Data Response Pipeline-project).

The task basically is split up into 4 parts with the Machine Learning Pipeline as central analytical part.
> Extract, Transform, Load the clean data (ETL) into a SQL-lite database
> Natural Language Processing (NLP), especially by performing starting-verb-extraction
> Machine Learning Pipeline (MLP) for analysing the data
> Visualisation of results in a WebApp that performs the classification

In order to understand how the different files work together (ETL, NLP, MLP, Web App) I’d like to show the file structure on my local computer: 
•	Working_files
o	app
 - run.py
 - templates
  •	go.html
  •	master.html
o	data
 -	categories.csv
 -	messages.csv
 -	ETL_Pipeline.db
 -	process_data.py
o	models
 -	classifier.pkl
 -	train_classifier.py
1st) For achieving good results make sure, that you have a 64bit version of Python installed as well as running 
2nd) Implement ETL pipeline by performing process_data.py in the data-folder
3rd) Train and improve the data on the ML-pipeline by starting train_classifier.py > used ada boost for final classification. 
Good: sequential learning technique in order to gradually improve classification. 
Not good: each predictor can only be trained after the previous predictor has been trained and evaluated, which makes the process a bit slower
4th) Initiate the webapp with run.py
5th) Go on http://localhost:3001/ 
6th) Et voilà

