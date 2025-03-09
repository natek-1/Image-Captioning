import numpy as np
import os, random
import pandas as pd
import spacy
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import transforms
from torchvision.models import vgg19
from tqdm import tqdm
from torchsummary import torchsummary
import torch.nn.functional as F
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data
df = pd.read_csv("data/captions.txt")
data_dict = {}
caption_dict = defaultdict(list)
for _, row in df.iterrows():
    caption_dict[row.image].append(row.caption)
    

class Vocabulary():
    spacy_eng = spacy.load("en_core_web_sm")

    
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freq_threshold = freq_threshold

    
    def __len__(self):
        return len(self.itos)


    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in Vocabulary.spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequency = {}
        idx = 4

        for sentence in sentence_list:
            for token in self.tokenizer_eng(sentence):

                frequency[token] = 1 + frequency.get(token, 0)

                if frequency[token] == self.freq_threshold:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1
    
    def tokenize(self, text):
        token_sent = self.tokenizer_eng(text)

        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
                for token in token_sent
            ]
    
    def vocabulary_size(self):
        return len(self.stoi)
    
    
vocabulary = Vocabulary(1)
vocabulary.build_vocabulary(pd.read_csv("data/captions.txt").caption.to_list())

class FlickrDataset(Dataset):
    max_len = 45
    def __init__(self, root_dir, data_dict, vocabulary: Vocabulary, transform=None, train=True):
        self.root_dir = root_dir
        self.data_dict = data_dict
        self.transform = transform

        # get the image and caption
        self.train = train
        self.caption = []
        self.item = self.setup_item()

        # Create our own vocabulary
        self.vocabulary = vocabulary
        self.sos_token = torch.tensor([self.vocabulary.stoi['<SOS>']], dtype=torch.int64)
        self.eos_token = torch.tensor([self.vocabulary.stoi['<EOS>']], dtype=torch.int64)
        self.pad_token = torch.tensor([self.vocabulary.stoi['<PAD>']], dtype=torch.int64)
    
    def __len__(self):
        return len(self.item)
    
    def setup_item(self):
        item = []
        if self.train:
            for image_id, image_captions in self.data_dict.items():
                for caption in image_captions:
                    self.caption.append(caption)
                    item.append((image_id, caption))
        else:
            for image_id, image_captions in self.data_dict.items():
                self.caption.extend(image_captions)
                item.append((image_id, image_captions))
        return item
    
    def __getitem__(self, index):
        # get image
        image_path = os.path.join(self.root_dir, self.item[index][0])
        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # get caption
        caption = self.item[index][1]
        
        if self.train:
            cap_len = len(self.vocabulary.tokenize(caption))
            num_pad = FlickrDataset.max_len - cap_len - 2
            if num_pad < 0:
                raise ValueError("Caption too long")
            num_caption = torch.cat([
                        self.sos_token ,
                        torch.tensor(self.vocabulary.tokenize(caption), dtype=torch.int64),
                        self.eos_token,
                        torch.tensor([self.pad_token] * num_pad, dtype=torch.int64)], dim=0)
            return img, num_caption
        else:
            captions = torch.zeros(5, 45).to(torch.long)
            for idx, cap in enumerate(caption):
                cap_len = len(self.vocabulary.tokenize(cap))
                num_pad = FlickrDataset.max_len - cap_len - 2
                if num_pad < 0:
                    raise ValueError("Caption too long")
                num_caption =    torch.cat([
                        self.sos_token ,
                        torch.tensor(self.vocabulary.tokenize(cap), dtype=torch.int64),
                        self.eos_token,
                        torch.tensor([self.pad_token] * num_pad, dtype=torch.int64)], dim=0)

                captions[idx] = num_caption
            return img, torch.LongTensor(captions)


def get_loader(
        root_dir,
        data_dict,
        vocabulary,
        transform,
        train=True,
        batch_size=32,
        shuffle=True,
):
    dataset = FlickrDataset(root_dir=root_dir, data_dict=data_dict, vocabulary=vocabulary,
                        transform=transform, train=train)
        
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader, dataset

folder = "data/images/"
df = pd.read_csv("data/captions.txt")
data_dict = {}
caption_dict = defaultdict(list)
for _, row in df.iterrows():
    caption_dict[row.image].append(row.caption)
train_data, val_data = train_val_split(caption_dict)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

train_dataset = FlickrDataset(root_dir=folder, data_dict=train_data,vocabulary=vocabulary,
                        transform=transform, train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)



import torchvision
from torchvision.models import ResNet101_Weights

class Encoder(nn.Module):

    def __init__(self, train_CNN=False):
        super(Encoder, self).__init__() # staying consistent with the paper by using vgg
        self.train_CNN = train_CNN
        self.model = vgg19()
        self.model = nn.Sequential(*list(self.model.features.children())[:-1])
        self.dim = 512
        self.freeze()
        
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = self.train_CNN

    def forward(self, images):
        features = self.model(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.shape[0], -1, features.shape[-1])
        return features # output should be of shape (batch_size, 196, 512)
    
    
    
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim, bias=False)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim, bias=False)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1, bias=False)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        nn.init.xavier_normal_(self.encoder_att.weight)
        nn.init.xavier_normal_(self.decoder_att.weight)
        nn.init.xavier_normal_(self.full_att.weight)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        num pixel is just 196 in out case
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights ()
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(-1)  # (batch_size, num_pixels, 1) -> (batch_size, num_pixels)
        alpha = self.softmax(att).unsqueeze(1)  # (batch_size, 1, num_pixels)
        attention_weighted_encoding = torch.bmm(alpha, encoder_out).squeeze(1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha.squeeze(1)
        

class Decoder(nn.Module):
    
    def __init__(self, embed_dim, attention_dim, encoder_dim, decoder_dim, vocab_size, dropout):

        super(Decoder, self).__init__()
        self.attention = Attention(encoder_dim=encoder_dim, decoder_dim=decoder_dim, attention_dim=attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.output = nn.Linear(decoder_dim, vocab_size)
        self.vocab_size = vocab_size

    
    def init_hidden_state(self, encoder_out: torch.Tensor):
        average_out = encoder_out.mean(dim=1) # output of shape (batch_size,  encoder_dim)
        h = self.init_h(average_out) # -> (batch_size,  decoder_dim)
        c = self.init_c(average_out) # -> (batch_size,  decoder_dim)
        return h, c
    

    def forward(self, encoder_out, caption):
        '''
        encoder_out will be of shape (batch_size, num_pixels, encoder_dim) eg (32, 196, 512)
        caption will be of shape (batch_size, 44) for training (only dealing with one caption) eg (32, 44)
        '''
        embeddings = self.embedding(caption) # -> (batch_size, max_seq_length, embedding_dim)
        cap_len = caption.size(-1)

        h, c = self.init_hidden_state(encoder_out) # both are of shape (batch_size, decoder_dim)
        device = h.device

        predictions = torch.zeros(caption.shape[0], caption.shape[1], self.vocab_size).to(device)
        alphas = torch.zeros(caption.shape[0], caption.shape[1], encoder_out.shape[1]).to(device)

        for i in range(cap_len):
            weighted_context, alpha = self.attention(encoder_out, h)
            #gate = self.sigmoid(self.f_beta(h))
            #weighted_context = gate * weighted_context
            
            # (batch_size, embedding_dim), (batch_size, encoder_dim) -> (batch_size, embedding_dim + encoder_dim)
            lstm_input = torch.cat([weighted_context, embeddings[:,i,:]], dim=1)
            h, c = self.lstm(lstm_input, (h, c)) # both are of shape (batch_size, decoder_dim)
            word_prop = self.output(self.dropout(h))  # shape (batch_size, vocab_size)
            
            predictions[:,i,:] = word_prop
            alphas[:,i,:] = alpha
                
        return predictions, alphas
    
    
class Caption(nn.Module):
    def __init__(self, embed_dim, attention_dim, encoder_dim, decoder_dim, vocab_size, dropout):
        super(Caption, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(embed_dim=embed_dim, attention_dim=attention_dim,
                               encoder_dim=encoder_dim, decoder_dim=decoder_dim, vocab_size=vocab_size, dropout=dropout)
        #self.encoder.fine_tine()
    
    def forward(self, img, captions):
        features = self.encoder(img)
        predictions, alphas = self.decoder(features, captions)
        return predictions, alphas

    def caption_img(self, img, vocab, max_length=100):
        result_caption = []

        with torch.no_grad():
            feature = self.encoder(img) # (batch_size, num_pixels, decoder_dim) 
            h, c = self.decoder.init_hidden_state(feature)

            ## first input to the model
            start = torch.zeros(size=(1,), dtype=torch.int)
            start[0] = vocabulary.stoi['<SOS>']
            start = start.unsqueeze(0) # shape (1, 1)
            start = start.to(h.device)
            embeddings = self.decoder.embedding(start).squeeze(0) # (1, embed_dim)
            print("Embedding Shape:", embeddings.shape)

            for _ in range(max_length):
                weighted_context, alpha = self.decoder.attention(feature, h)
                
                #gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                #weighted_context = gate * weighted_context
                
                #print("Weighted context shape:",weighted_context.shape)
                lstm_input = torch.cat([weighted_context, embeddings], dim=1) # batch_size, embed_dim+encode_dim
                #print("Lstm input shape", lstm_input.shape)

                h, c = self.decoder.lstm(lstm_input, (h, c))
                #print("Hidden state shape", h.shape)
                output = self.decoder.output(h.squeeze(0)) # removing the extra dimension needed in lstm, output.shape = (vocab_size)
                
                #print("Total output shape", output.shape)
                predicted = output.argmax(dim=-1) # highest probablities word
                result_caption.append(predicted.item())
                embeddings = self.decoder.embedding(predicted).unsqueeze(0)
                #print(embeddings.shape)
                if vocab.itos[predicted.item()] == "<EOS>":
                    break
            
            return [vocab.itos[idx] for idx in result_caption] #return the final sentence



train_CNN = False
embed_dim = 256
attention_dim = 256
decoder_dim = 256
encoder_dim = 512
vocab_size = vocabulary.vocabulary_size()
num_layers = 1
lr= 3e-4
num_epochs = 100
load_model = False
save_model = False
step = 0


train_dataset = FlickrDataset(root_dir=folder, data_dict=train_data,vocabulary=vocabulary,
                        transform=transform, train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)


#writer = SummaryWriter(log_dir="runs/flickr")
model_path = "state_dict.py"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device="cpu"

# Model
model = Caption(embed_dim=embed_dim, attention_dim=attention_dim, encoder_dim=encoder_dim,
                decoder_dim=decoder_dim, vocab_size=vocab_size, dropout=0.1).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.stoi["<PAD>"])
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=lr)
start_epoch = 0
if load_model:
    state_dict = torch.load(model_path)
    model = state_dict['model']
    optimizer = state_dict['optimizer']
    step = state_dict['steps']
    start_epoch = 1 + state_dict['epochs']

def train_epoch(train_loader, captioner: Caption, device, criterion, optimizer, epoch, alpha_c=None):
    losses = []

    captioner.train()

    for idx, (imgs, caps) in enumerate(tqdm(train_loader, total=len(train_loader))):
        # move tensor to device if available
        imgs = imgs.to(device)
        caps = caps.to(device)

        optimizer.zero_grad()

        # forward prop
        predictions, alphas = captioner(imgs, caps)

        
        #att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
        loss = criterion(predictions.view(-1, predictions.size(-1)), caps.view(-1)) #+ att_regularization

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(captioner.parameters(), max_norm=1.)
        optimizer.step()

        # keep track of metrics
        losses.append(loss.item())
        break
        

    print('Training Epoch #: [{0}]\t'
        'Loss: {loss:.4f}\t'.format(
                epoch, loss=np.mean(losses)))

    return np.mean(losses)

for epoch in range(start_epoch, num_epochs):

    loss = train_epoch(train_loader=train_loader, captioner=model, device=device,
                            criterion=criterion, optimizer=optimizer, epoch=epoch)

params = {
    "model": model,
}
torch.save("model.py", params)
