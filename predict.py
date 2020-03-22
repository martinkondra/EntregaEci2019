import load_data
import torch
import numpy as np
import torch.nn.functional as F
#from models.LSTM import LSTMClassifier
#from models.LSTM_Attn import AttentionModel
from models.RCNN import RCNN
#from models.selfAttention import SelfAttention

batch_size = 32
output_size = 3 # n etiquetas
hidden_size = 256
embedding_length = 300
epochs = 10
learning_rate = 2e-5

TEXT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset(batch_size=batch_size)

#model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

model.load_state_dict(torch.load('./66.18RCNN0.0002.msd', map_location='cpu'))
model.eval()

with open('./eci_data/test_to_predict.tsv') as f:
	content = f.readlines()
	content = [c.strip() for c in content]
	content = [c.split('\t') for c in content]

print('pairID,gold_label')
for s, pairID in content:
	with torch.no_grad():
		sentence = TEXT.preprocess(s)
		sentence = [[TEXT.vocab.stoi[x] for x in sentence]]
		test_sen = np.asarray(sentence)
		test_sen = torch.LongTensor(test_sen)
		#test_tensor = Variable(test_sen, volatile=True)
		test_tensor = test_tensor.cpu()
		output = model(test_tensor, 1)
		out = F.softmax(output, 1)
		if (torch.argmax(out[0]) == 0):
			label = 'entailment'
		elif (torch.argmax(out[0]) == 1):
			label = 'contradiction'
		else:
			label = 'neutral'
		print((pairID + ',' + label).rstrip('\n'))
