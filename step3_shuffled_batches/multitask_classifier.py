'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
import utils
from types import SimpleNamespace
from tokenizer import BertTokenizer

import torch
import gc
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pc_grad import PCGrad

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        # insert further pretrain here
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # self.further_pretain()

        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.dropout = torch.nn.Dropout(0.1)
        self.pd_dropout = torch.nn.Dropout(0.1)
        self.sts_dropout = torch.nn.Dropout(0.1)
        self.sentiment_lin_proj = torch.nn.Linear(config.hidden_size, 5)
        self.paraphrase_lin_proj1 = torch.nn.Linear(config.hidden_size, 1)
        self.similarity_lin_proj1 = torch.nn.Linear(config.hidden_size, 1)
        # self.similarity_lin_proj2 = torch.nn.Linear(config.hidden_size, 32)
        # self.similarity_lin_proj3 = torch.nn.Linear(32 * 3, 1)

        # common layer approach -- PUT IT IN FORWARDS LATER
        self.common_lin_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.cos = torch.nn.CosineSimilarity()
        self.cos_emb_loss = torch.nn.CosineEmbeddingLoss()
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        map = self.bert.forward(input_ids, attention_mask)
        pooler_output = map['pooler_output'] # cls token hidden state
        return pooler_output

    def predict_sentiment(self, input_ids, attention_mask, get_embed=False):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooler_output = self.forward(input_ids, attention_mask)
       
        output = self.eval_sentiment(pooler_output)

        if get_embed:
            return output, pooler_output

        return output
    
    def eval_sentiment(self, embeddings):
        output = self.dropout(embeddings)
        output = self.sentiment_lin_proj(output)
        output = F.softmax(output)
        return output

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1, # inputs are batch_size * hidden_size
                           input_ids_2, attention_mask_2, get_embed=False):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        embeddings = self.get_embeddings(input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2)
        
        if get_embed:
            return self.eval_paraphrase(embeddings), embeddings
    
        return self.eval_paraphrase(embeddings)
    
    def eval_paraphrase(self, embeddings):
        intermediate = self.pd_dropout(embeddings)

        intermediate = self.paraphrase_lin_proj1(intermediate)
        return intermediate


    def predict_similarity(self,input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        embeddings = self.get_embeddings(input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2)

        return self.eval_similarity(embeddings)
    
    def eval_similarity(self, embeddings):
        intermediate = self.sts_dropout(embeddings)

        intermediate = self.similarity_lin_proj1(intermediate)

        return intermediate

        
    def get_embeddings(self, input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        sep = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device).repeat(input_ids_1.shape[0], 1)
        sep_mask = torch.ones(sep.shape, device=input_ids_1.device)

        input_id = torch.cat((input_ids_1, sep, input_ids_2, sep), dim=1)
        attention_mask = torch.cat((attention_mask_1, sep_mask, attention_mask_2, sep_mask), dim=1)

        return  self.forward(input_id, attention_mask)





def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def batch_semantic(batch, optimizer, model, device, epoch):
    b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    b_labels = b_labels.to(device)

    optimizer.zero_grad()
    logits,embed = model.predict_sentiment(b_ids, b_mask, get_embed=True)
    reg_loss = utils.smart_reg(model.eval_sentiment, logits, embed)
    # l2_reg = 1 / (epoch+1) * utils.l2_reg(model)
    sentiment_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
    sentiment_loss += reg_loss

    return sentiment_loss

def batch_paraphrase(batch, optimizer, model, device, epoch):
    b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'], batch['token_ids_2'],
                                        batch['attention_mask_1'], batch['attention_mask_2'],
                                        batch['labels'])

    b_ids_1 = b_ids_1.to(device)
    b_ids_2 = b_ids_2.to(device)
    b_mask_1 = b_mask_1.to(device)
    b_mask_2 = b_mask_2.to(device)

    b_labels = torch.tensor(b_labels, dtype=torch.float)
    b_labels = b_labels.to(device)

    optimizer.zero_grad()
    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

    logits = torch.sigmoid(logits)

    # reg_loss = utils.smart_reg(model.eval_paraphrase, logits, embed)
    l2_reg = .01 / (epoch+1) * utils.l2_reg(model)

    paraphrase_loss = F.binary_cross_entropy(logits, torch.unsqueeze(b_labels, 1), reduction='sum') / args.batch_size
    paraphrase_loss += l2_reg

    return paraphrase_loss

def batch_sts(batch, optimizer, model, device, epoch):
    b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'], batch['token_ids_2'],
                                           batch['attention_mask_1'], batch['attention_mask_2'],
                                           batch['labels'])

    b_ids_1 = b_ids_1.to(device)
    b_ids_2 = b_ids_2.to(device)
    b_mask_1 = b_mask_1.to(device)
    b_mask_2 = b_mask_2.to(device)

    b_labels = torch.tensor(b_labels, dtype=torch.float)
    b_labels = b_labels.to(device)

    optimizer.zero_grad()
                
                # embeddings = model.get_embeddings(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

    sts_loss = F.mse_loss(5 * torch.sigmoid(logits), b_labels.unsqueeze(1), reduction='sum') / args.batch_size

    return sts_loss

def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    assert args.fine_tune_mode in ["last-linear-layer", "full-model"]
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='dev')
    
    # temp truncate
    para_train_data = para_train_data[:10000]
    para_dev_data = para_dev_data[:2000]

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    # lr = args.lr
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # optimizer = PCGrad(adam_optimizer) 

    best_dev_acc_sst = 0
    best_dev_acc_para = 0
    best_dev_corr_sts = -1
    best_score = 0

    sentiment_loss = 0
    paraphrase_loss = 0
    sts_loss = 0
    # Run for the specified number of epochs.
    for epoch in range(15):
        sentiment_dev_accuracy = 0
        paraphrase_dev_accuracy = 0
        sts_dev_corr = 0

        model.train()
        # old_state_dict = model.state_dict()
        train_loss = 0
        num_batches = 0

        # sst_batches = tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
        # pd_batches = tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
        # sts_batches = tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)

        max_len = max([len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader)])

        total_len = sum([len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader)])
        iters = int(total_len / 3)
        train_loss = 0

        for i in range(iters):
            if (i % 50  == 0):
                print("Batch: " + str(i) + "/" + str(iters))
            randy = random.randint(0, total_len - 1)

          
            for j in range(3):
                if randy < len(sst_train_dataloader): # sst
                        batch = next(iter(sst_train_dataloader))
                        loss = batch_semantic(batch, optimizer, model, device, epoch)
                elif randy < len(sst_train_dataloader) + len(para_train_dataloader): # pd
                        batch = next(iter(para_train_dataloader))
                        loss = batch_paraphrase(batch, optimizer, model, device, epoch)
                else: # sts
                        batch = next(iter(sts_train_dataloader))
                        loss = batch_sts(batch, optimizer, model, device, epoch)
                loss.backward()
                

                # optimizer._pc_backward(loss)
                train_loss += loss.item()
                num_batches += 1
                

            torch.cuda.empty_cache()
            gc.collect()

                # optimizer.pc_backward(loss1, loss2, loss3) # calculate the gradient can apply gradient modification
            # optimizer.pc_backward()
            optimizer.step()  # apply gradient step

            

        (sentiment_accuracy_train ,sst_y_pred, sst_sent_ids, 
        paraphrase_accuracy_train, para_y_pred, para_sent_ids,
        sts_corr_train, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_train_dataloader,
                             para_train_dataloader,
                             sts_train_dataloader,
                             model, device)

        (sentiment_accuracy_dev,sst_y_pred, sst_sent_ids, 
        paraphrase_accuracy_dev, para_y_pred, para_sent_ids,
        sts_corr_dev, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader,
                             para_dev_dataloader,
                             sts_dev_dataloader,
                             model, device)


        curr_score = utils.calc_score(sentiment_accuracy_dev, paraphrase_accuracy_dev, sts_corr_dev)
        if curr_score > best_score:
            best_score = curr_score
            print(best_score)
            save_model(model, optimizer, args, config, args.filepath)

        train_loss = train_loss / (num_batches)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")
        print(f"Sentiment train acc :: {sentiment_accuracy_train :.3f}, dev acc :: {sentiment_accuracy_dev :.3f}")
        print(f"Paraphrase train acc :: {paraphrase_accuracy_train :.3f}, dev acc :: {paraphrase_accuracy_dev :.3f}")
        print(f"Similarity train corr :: {sts_corr_train :.3f}, dev acc :: {sts_corr_dev :.3f}")

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)