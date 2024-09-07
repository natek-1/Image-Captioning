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
from torchvision.models import resnet50, vgg19
from tqdm import tqdm
import torchsummary
import torch.nn.functional as F
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu


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
train_data, val_data = train_val_split(caption_dict)


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

class FlickrDataset(Dataset):

    def __init__(self, image_dir, vocabulary: Vocabulary, data_dict, transform=None, train=True):
        self.data_dict = data_dict
        self.transform = transform
        self.image_dir = image_dir


        self.train = train
        self.item = self.setup_item()
        
        self.vocabulary = vocabulary

    
    def setup_item(self):
        item = []
        if self.train:
            for image_id, image_captions in self.data_dict.items():
                for caption in image_captions:
                    item.append((image_id, caption))
        else:
            for image_id, image_captions in self.data_dict.items():
                item.append((image_id, image_captions))
        return item

    def __len__(self):
        return len(self.item)


    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.item[index][0])
        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        caption = self.item[index][1]
        if self.train:
            num_caption = [self.vocabulary.stoi['<SOS>']]
            num_caption += self.vocabulary.tokenize(caption)
            num_caption.append(self.vocabulary.stoi['<EOS>'])
            num_caption = torch.tensor(num_caption)
            return img, num_caption
        else:
            captions = []
            for cap in caption:
                num_caption = [self.vocabulary.stoi['<SOS>']]
                num_caption += self.vocabulary.tokenize(cap)
                num_caption.append(self.vocabulary.stoi['<EOS>'])
                captions.append(torch.tensor(num_caption))
            return img, pad_sequence(captions, batch_first=False, padding_value=0)

class MyCollate:
    def __init__(self, pad_idx, train=True):
        self.pad_idx = pad_idx
        self.train = train
    
    def __call__(self, batch): # pad sequnece
        img = [item[0].unsqueeze(0) for item in batch]
        img = torch.cat(img, 0)
        target = [item[1] for item in batch]
        if self.train:
            target = pad_sequence(target, batch_first=True, padding_value=self.pad_idx)
        else:
            #for i in target:
            #    print(i.shape)
            target = pad_sequence(target, batch_first=True, padding_value=self.pad_idx)
            target=target.permute(0, 2, 1).contiguous() #get back to regular batch_first = True

        return img, target

def get_loader(
        image_folder,
        data_dict,
        vocabulary,
        transform,
        train=True,
        batch_size=32,
        num_worker=0,
        shuffle=True,
        pin_memory=True
):
    dataset = FlickrDataset(image_dir=image_folder, vocabulary=vocabulary, data_dict=data_dict,
                            transform=transform, train=train)
    pad_idx = vocabulary.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=MyCollate(pad_idx=pad_idx, train=train),
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_worker,
    )

    return loader, dataset

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        model = vgg19("VGG19_Weights.IMAGENET1K_V1")
        self.model = list(model.features.children())[:-1]
        self.model = nn.Sequential(*self.model)
        self.dim = 512
    
    def fine_tine(self, finetune=False):
        for param in self.model.parameters():
            param.requires_grad = finetune
    
    def forward(self, images):
        out = self.model(images)
        out = out.permute(0, 2, 3, 1)
        out = out.view(out.size(0), -1, out.size(-1))
        return out

class Attention(nn.Module):
    """
    Attention Network. Using Additive or BahdanauAttention
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decorder_attention = nn.Linear(decoder_dim, attention_dim)
        self.attend = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, hidden_state):
        img_matrix = self.encoder_attention(encoder_out) # (batch_size, num_pixels, attention_dim)
        hidden_matrix = self.decorder_attention(hidden_state).unsqueeze(1) #(batch_size, 1, attention_dim)
        add = self.tanh(img_matrix + hidden_matrix) # (batch_size, num_pixels, attention_dim)
        att = self.attend(add).squeeze(2) # (batch_size, num_pixels)
        alpha = self.softmax(att) # (batch_size, num_pixels)
        weighted_context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (batch_size, encoder_dim)
        return weighted_context, alpha



class Decoder(nn.Module):
    
    def __init__(self, embed_dim, attention_dim, encoder_dim, decoder_dim, vocab_size):

        super(Decoder, self).__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout()

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.output = nn.Linear(decoder_dim, vocab_size)
        self.vocab_size = vocab_size

    
    def init_hidden_state(self, encoder_out: torch.Tensor):
        average_out = encoder_out.mean(dim=1)
        h = self.init_h(average_out)
        c = self.init_c(average_out)
        return h, c
    

    def forward(self, encoder_out, caption):

        embeddings = self.embedding(caption)

        h, c = self.init_hidden_state(encoder_out)
        device = h.device

        predictions = torch.zeros(caption.shape[0], caption.shape[1], self.vocab_size).to(device)
        alphas = torch.zeros(caption.shape[0], caption.shape[1], encoder_out.shape[1]).to(device)

        for i in range(caption.size(-1)):
            weighted_context, alpha = self.attention(encoder_out, h)
            #gate = self.sigmoid(self.f_beta(h))
            #weighted_context = gate * weighted_context
            lstm_input = torch.cat([embeddings[:,i,:], weighted_context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            word_prop = self.output(self.dropout(h))
            
            for j in range(word_prop.size(0)):
                predictions[j,i] = word_prop[j]
            for j in range(alpha.size(0)):
                alphas[j,i] = alpha[j]
        return predictions, alphas


class Caption(nn.Module):
    def __init__(self, embed_dim, attention_dim, encoder_dim, decoder_dim, vocab_size):
        super(Caption, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(embed_dim=embed_dim, attention_dim=attention_dim,
                               encoder_dim=encoder_dim, decoder_dim=decoder_dim, vocab_size=vocab_size)
        self.encoder.fine_tine()
    
    def forward(self, img, captions):
        features = self.encoder(img)
        predictions, alphas = self.decoder(features, captions)
        return predictions, alphas


def train_epoch(train_loader, captioner, device, criterion, optimizer, alpha_c, epoch):
    losses = []

    captioner.train()

    for idx, (imgs, caps) in enumerate(tqdm(train_loader, total=len(train_loader))):
        # move tensor to device if available
        imgs = imgs.to(device)
        caps = caps.to(device)

        optimizer.zero_grad()

        # forward prop
        predictions, alphas = captioner(imgs, caps)
        print(predictions.shape)
        print(caps.shape)
        loss = criterion(predictions.view(-1, predictions.size(-1)), caps.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(captioner.parameters(), max_norm=1.)
        optimizer.step()

        # keep track of metrics
        losses.append(loss.item())
        break
        

    print('Training Epoch #: [{0}]\t'
        'Loss: {loss:.4f}\t'.format(
                epoch, loss=np.mean(losses)))

    return np.mean(losses)



embed_dim = 512
attention_dim = 64
decoder_dim = 64
encoder_dim = 512
lr = 1e-6
alpha_c = 1.
vocabulary = Vocabulary(1)
vocabulary.build_vocabulary(pd.read_csv("data/captions.txt").caption.to_list())
vocab_size = len(vocabulary)
epochs = 100

device = "cpu"
if torch.cuda.is_available():
    device = "cpu"
elif torch.backends.mps.is_available():
    device="mps"


model = Caption(embed_dim=embed_dim, attention_dim=attention_dim, encoder_dim=encoder_dim,
                decoder_dim=decoder_dim, vocab_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.stoi["<PAD>"])
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=lr)
train_loader, _ = get_loader(data_dict=train_data, transform=transform, vocabulary=vocabulary, image_folder="data/images/")
for epoch in range(epochs):

    loss = train_epoch(train_loader=train_loader, captioner=model, device=device,
                            criterion=criterion, optimizer=optimizer, alpha_c=1., epoch=epoch)

    