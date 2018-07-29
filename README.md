# LOIS

LOIS is a chatbot. One day she can be your personal assistant too..!

## Architecture

LOIS has been recently upgraded to a very scalable architecture. Now Deep Learning runs inside her head. 
Now it uses LSTM networks to classify sentences into various categories for which it can respond. The word embeddings are from GLOVE dataset.
Inside the 'dataset' folder, each file in .txt format stores sentences that fall into their respective categories.

For example, for LOIS to identify the query for time, the dataset of that class will have various sentences that corresponds to asking
time.. like

'What's the time?'
'Tell me the Time'
 
 etc
 
 LOIS classifies the input sentence into these categories and then we can code its response in python. Now we use simple Natural language 
 methods to extract the parts of speech to further process the requests.
