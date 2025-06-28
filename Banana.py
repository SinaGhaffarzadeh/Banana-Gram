"""
In this implementation we will try to develop a bigram language model which is one of simplest 
effective way to model language by capturing local word dependencies. A bigram language model 
predicts the probability of a word in a sequence based on the previous word, focusing on pairs 
of consecutive words. 
In the other word, a bigram means, two words that appear next to each other.
So a bigram language model learns the probability of a word given the previous word.

For instance:

" I love cats "

The bigrams are:
I → love
love → cats

The model learns patterns like:
If you see "I", there's a high chance the next word is "love".
If you see "love", the next word might be "cats".

if we imagine that I love cats, You love dogs, I love dogs are our input data
The bigram of them will;
(I, love), (love, cats)
(You, love), (love, dogs)
(I, love), (love, dogs)


How we can train the model?

1. Inputs and Outputs
We treat each word/character as an integer (via a dictionary called a vocab)
say:
{'I': 0, 'love': 1, 'cats': 2, 'dogs': 3, 'You': 4}

Then bigrams become:
(0, 1), (1, 2), ...

"""

'''
python --version --> Python 3.10.9
pip3 install matplotlib numpy pylzma ipykernel jupyter
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
python -m ipykernel install --user --name=cuda --dispaly-name "cuda-banana"


Our dataset (small data) was collected from Project Gutenberg (https://www.gutenberg.org/)
We downloaded "Dorothy and the Wizard in Oz by L. Frank Baum" book in  Plain text form.
'''

# Labreries
import torch
import torch
from torch.nn import functional as F
import torch.nn as nn

# Cuda Check
device = "cuda" if torch.cuda.is_available() else "cpu"
print (device)

#Initializing HyperParameters
block_size = 8 # In each block we will see 8 character
batch_size = 4 
iteration = 10000
lr = 3e-4
eval_iters = 250

# Opening downloaded data
with open("Text file ", "r", encoding='utf-8') as file:
    text = file.read()
# print("First 200 words of book: ","\n",text[:200],"\n")


# Making vocabulary: Extracting Characters of the book and sorting them into a list for Tokenizing (Character-level Tokenizing).
chars = sorted(set(text))
print("All Characters has been used in this book: ",chars,"\n")
vocabulary_size = len(chars)
# Tokenizing Characters into their integer equivalent form.
string_to_int = { ch:i for i,ch in enumerate(chars)}
int_to_string = { i:ch for i,ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

encoded_word = encode("hello")
decoded_word = decode(encoded_word)

# print(f"Encoded word is: {encoded_word}")
# print(f"Decoded word is: {decoded_word}")

'''
In tokenizing words, we will reach to hundred or thouns of different tokens that make handling it difficult.
Therefore, here we just encode our characters as input data and use them in the learning process. 
'''

# Converting to tensor 
data = torch.tensor(encode(text), dtype= torch.long)  # torch.long is equivalent torch.int64. We use it when we need integer indices (e.g., in embeddings or nn.Embedding)
                                                      # classification labels (e.g., for CrossEntropyLoss)
                                                      # counting, indexing, or discrete values


# Deviding raw data into Train and Test regarding to bigram concept (learning based on pevious character)

div = int(0.8*len(data))
train_data = data[:div]
val_data = data[div:]


# In below we will illustrate how we will inject data to the model to learn based on bigam technique.
# # block_size = 8 # In each block we will see 8 character

xx = train_data[:block_size]
yy = train_data[1:block_size+1]

for t in range(block_size):
    context = xx[:t+1]
    target = yy[t]
    print("when input is ",context, "target will be ", target)


print ("\n",12*"===","\n")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # it returns 4 random int values that is equal to our batch size
    # print(ix)
    # ix is equal to the number of batch size which mean in each batch we will have 4 block values 
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device) , y.to(device)
    return x, y # This data will be fed to the model

x,y = get_batch("train")
print("input data: ","\n", x)
print("target data: ","\n", y)


"""
This block is the core of token generation in many autoregressive models. 
It's simple, flexible, and powerful, and reflects a typical decoder behavior in generative language models.
"""
class BigramLanguageModel(nn.Module): # The reason of chosing nn.Module as sub-class of this class is by setting any learnable
                                      # parameter by changing gradian it will be changed.
                                      # features that by adding nn.module will be perform: Automatically register parameters (like weights), 
                                      # Handle GPU transfers, Use built-in optimizers and model evaluation utilities, Track gradients for learning via .backward().

    def __init__(self, vocabulary_size): # here we just initialize some values which one of them as always is self.
        super().__init__() # is used inside a subclass to call the __init__ method of its parent (super) class. 
                           #This allows the subclass to inherit and initialize everything from the parent class before adding its own custom behavior.
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)
    
    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # B= batch, T= time steps (sequence length), C= chanel (vocabulary size)
            logits = logits.view(B*T, C) # view is used to reshape tensors. The loss function, according to its doc in pytorch get (N,C) and (N)
                                         # dimentional inputs. So, by "view" we will reshape it.
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus on the last character
            logits = logits[:,-1,:] # In each iteration, a character will be added to index (input character). So, it leads to be increased number of inputs from 1 to N and 
                                    # in each iteration we just want to predict the next character by the last character.
                                    # At first, we will have (1,1,vocabulary_size). in the next iteration it will be returned to (1,2,vocabulary_size)

            # apply softmax to get probiblistics
            probs = F.softmax(logits, dim=-1) # Normalizing between 0-1
            index_next = torch.multinomial(probs, num_samples=1) # Sample one (num_samples=1) token from the probability distribution (probs)
                                                                 # We can not pick the max value in probs because the model would always repeat the most likely output (argmax), 
                                                                 # leading to repetitive sequences. Also, Sampling by multinomial introduces diversity and creativity in generated text.
                                                                 # In each iteration, we should finally get a single new next token that we have done it by multinomial.
            # Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=-1)
        
        return index


model = BigramLanguageModel(vocabulary_size).to(device)
# context = torch.zeros((1,1), dtype = torch.long, device = device)
# generated_chars = decode(model.generate(context, max_new_tokens = 500)[0].tolist())
# print(generated_chars)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
    return out 
            


# Optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr=lr) # AdamW or Adam with weight decay

for iter in range(iteration):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"Step {iter} and the losses {losses}")
    
    xb, yb = get_batch("train")
    logits, loss = model.forward(xb , yb)
    optimizer.zero_grad(set_to_none=True) # None occupies a lot less space than zero, so we sat "set_to_none=True". 
                                          # This will in general have lower memory footprint, and can modestly improve performance. 
                                          # By setting to True, PyTorch  will allocate a new gradient tensor during the backward pass. Otherwise,
                                          # PyTorch will reuse the existing tensor (which might be more memory-consuming if unused).
    
    loss.backward()
    optimizer.step()

print(loss.item())


# Output of Model
context = torch.zeros((1,1), dtype = torch.long, device = device)
generated_chars = decode(model.generate(context, max_new_tokens = 500)[0].tolist())
print(generated_chars)