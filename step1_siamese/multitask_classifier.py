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

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        self.pd_dropout = torch.nn.Dropout(0.2)
        self.sts_dropout = torch.nn.Dropout(0.2)
        self.sentiment_lin_proj = torch.nn.Linear(config.hidden_size, 5)
        self.paraphrase_lin_proj1 = torch.nn.Linear(config.hidden_size, 32)
        self.paraphrase_lin_proj2 = torch.nn.Linear(config.hidden_size, 32)
        self.paraphrase_lin_proj3 = torch.nn.Linear(32 * 3, 1)#32)
        self.paraphrase_lin_proj4 = torch.nn.Linear(32, 1)
        self.similarity_lin_proj1 = torch.nn.Linear(config.hidden_size, 32)
        self.similarity_lin_proj2 = torch.nn.Linear(config.hidden_size, 32)
        self.similarity_lin_proj3 = torch.nn.Linear(32 * 3, 1)

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

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooler_output = self.forward(input_ids, attention_mask)
       
        output = self.dropout(pooler_output)
        output = self.sentiment_lin_proj(output)
        output = F.softmax(output)

        return output


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1, # inputs are batch_size * hidden_size
                           input_ids_2, attention_mask_2, rtn_intermediates = False, perturb = False,
                           switch=False):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        pooler_output1 = self.forward(input_ids_1, attention_mask_1)
        pooler_output2 = self.forward(input_ids_2, attention_mask_2)

        dropout1 = self.pd_dropout(pooler_output1)
        dropout2 = self.pd_dropout(pooler_output2)

        intermediate1 = self.paraphrase_lin_proj1(dropout1)
        intermediate2 = self.paraphrase_lin_proj2(dropout2)

        # cosine similarity method
        # output = self.cos(intermediate1, intermediate2)

        # absolute difference method
        diff = torch.abs(intermediate1 - intermediate2)
        output  = self.paraphrase_lin_proj3(torch.cat((intermediate1, intermediate2, diff), dim=1))

        # if switch:
        # s_intermediate1 = self.paraphrase_lin_proj2(dropout1)
        # s_intermediate2 = self.paraphrase_lin_proj1(dropout2)

        # s_diff = torch.abs(s_intermediate1, s_intermediate2)
        # s_output = self.paraphrase_lin_proj3(torch.cat((s_intermediate1, s_intermediate2, s_diff), dim=1))
        # switch_output = self.paraphrase_lin_proj4(torch.cat(s_output, output))


        if perturb:
            pooler_output1 = utils.perturb(pooler_output1)
            pooler_output2 = utils.perturb(pooler_output2)

            p_intermediate1 = self.pd_dropout(pooler_output1)
            p_intermediate2 = self.pd_dropout(pooler_output2)

            p_intermediate1 = self.paraphrase_lin_proj1(p_intermediate1)
            p_intermediate2 = self.paraphrase_lin_proj2(p_intermediate2)

            diff = torch.abs(p_intermediate1 - p_intermediate2)
            p_output  = self.paraphrase_lin_proj3(torch.cat((p_intermediate1, p_intermediate2, diff), dim=1))


            return output, p_output
        
   

  
        if not rtn_intermediates:
            return output
        else:
            return intermediate1, intermediate2



    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2, rtn_intermediates = False, perturb = False):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        pooler_output1 = self.forward(input_ids_1, attention_mask_1)
        pooler_output2 = self.forward(input_ids_2, attention_mask_2)

        intermediate1 = self.sts_dropout(pooler_output1)
        intermediate2 = self.sts_dropout(pooler_output2)

        intermediate1 = self.similarity_lin_proj1(intermediate1)
        intermediate2 = self.similarity_lin_proj2(intermediate2)

        intermediate1 = self.relu(intermediate1)
        intermediate2 = self.relu(intermediate2)

        similarity = self.cos(intermediate1, intermediate2)

        if perturb:
            pooler_output1 = utils.perturb(pooler_output1)
            pooler_output2 = utils.perturb(pooler_output2)

            p_intermediate1 = self.pd_dropout(pooler_output1)
            p_intermediate2 = self.pd_dropout(pooler_output2)

            p_intermediate1 = self.paraphrase_lin_proj1(p_intermediate1)
            p_intermediate2 = self.paraphrase_lin_proj2(p_intermediate2)

            p_similarity = self.cos(p_intermediate1, p_intermediate2)


            return similarity, p_similarity
  

        if not rtn_intermediates:
            return similarity
        else:
            return intermediate1, intermediate2




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

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc_sst = 0
    best_dev_acc_para = 0
    best_dev_corr_sts = -1


    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        # old_state_dict = model.state_dict()
        train_loss = 0
        num_batches = 0
        # for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        #     b_ids, b_mask, b_labels = (batch['token_ids'],
        #                                batch['attention_mask'], batch['labels'])

        #     b_ids = b_ids.to(device)
        #     b_mask = b_mask.to(device)
        #     b_labels = b_labels.to(device)

        #     optimizer.zero_grad()
        #     logits = model.predict_sentiment(b_ids, b_mask)
        #     # loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
        #     logits = logits.argmax(-1)
        #     loss = F.mse_loss(logits, b_labels.view(-1)) / args.batch_size

        #     loss.backward()
        #     optimizer.step()

        #     train_loss += loss.item()
        #     num_batches += 1

        # sentiment_train_accuracy = model_eval_multitask(sst_train_dataloader,
        #                  para_train_dataloader,
        #                  sts_train_dataloader,
        #                  model, device, type='sst')[0]
        
        # sentiment_dev_accuracy = model_eval_multitask(sst_dev_dataloader,
        #                  para_dev_dataloader,
        #                  sts_dev_dataloader,
        #                  model, device, type='sst')[0]

        # if sentiment_dev_accuracy > best_dev_acc_sst:
        #     best_dev_acc_sst = sentiment_dev_accuracy
        #     save_model(model, optimizer, args, config, args.filepath)

            
        
        # for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        #         b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'], batch['token_ids_2'],
        #                                 batch['attention_mask_1'], batch['attention_mask_2'],
        #                                 batch['labels'])

        #         b_ids_1 = b_ids_1.to(device)
        #         b_ids_2 = b_ids_2.to(device)
        #         b_mask_1 = b_mask_1.to(device)
        #         b_mask_2 = b_mask_2.to(device)

        #         b_labels = torch.tensor(b_labels, dtype=torch.float)
        #         b_labels = b_labels.to(device)

        #         reg = 0



        #         optimizer.zero_grad()
        #         # logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

        #         # bergman
        #         logits, perturbed_logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2, perturb=True)
        #         perturb_p = model.log_softmax(perturbed_logits.unsqueeze(0))
        #         logit_p = model.softmax(logits.unsqueeze(0))
        #         reg = torch.nn.functional.mse_loss(perturb_p, logit_p) / args.batch_size

        #         # dense head
        #         # logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2, switch=True)

        #         loss = F.binary_cross_entropy(torch.sigmoid(logits), torch.unsqueeze(b_labels, 1), reduction='sum') / args.batch_size
        #         loss += reg * 5

        #         # cos emb loss
        #         #intermediate1, intermediate2 = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2, True)
        #         #loss = model.cos_emb_loss(intermediate1, intermediate2, b_labels)


        #         # nmr loss
        #         # intermediate1, intermediate2 = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2, True)
        #         # cross_entropy_loss = nn.CrossEntropyLoss()
        #         # loss = utils.mnr_loss(intermediate1, intermediate2, cross_entropy_loss)
                
        #         loss.backward()
        #         optimizer.step()

        #         train_loss += loss.item()
        #         num_batches += 1

        # paraphrase_train_accuracy= model_eval_multitask(sst_train_dataloader,
        #                     para_train_dataloader,
        #                     sts_train_dataloader,
        #                     model, device,type='para')[0]

        # paraphrase_dev_accuracy= model_eval_multitask(sst_dev_dataloader,
        #                     para_dev_dataloader,
        #                     sts_dev_dataloader,
        #                     model, device,type='para')[0]

        # if paraphrase_dev_accuracy > best_dev_acc_para:
        #         best_dev_acc_para = paraphrase_dev_accuracy
        #         save_model(model, optimizer, args, config, args.filepath)

            
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
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
                # logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                # loss = F.mse_loss(5 * torch.sigmoid(logits), b_labels, reduction='sum') / args.batch_size

                # # rescale labels
                # b_labels = b_labels / 5 * 2 - 1

                # intermediate1, intermediate2 = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                # loss = model.cos_emb_loss(intermediate1, intermediate2, b_labels)

                # bergman
                logits, perturbed_logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2, perturb=True)

                perturb_p = model.log_softmax(perturbed_logits.unsqueeze(0))
                logit_p = model.softmax(logits.unsqueeze(0))
                reg = model.kl_loss(perturb_p, logit_p)

                loss = F.mse_loss(5 * torch.sigmoid(logits), b_labels, reduction='sum') / args.batch_size
                loss += reg * 1

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        sts_train_corr = model_eval_multitask(sst_train_dataloader,
                             para_train_dataloader,
                             sts_train_dataloader,
                             model, device, type='sts')[0]

        sts_dev_corr = model_eval_multitask(sst_dev_dataloader,
                             para_dev_dataloader,
                             sts_dev_dataloader,
                             model, device, type='sts')[0]

        if sts_dev_corr > best_dev_corr_sts:
                best_dev_corr_sts = sts_dev_corr
                save_model(model, optimizer, args, config, args.filepath)

        train_loss = train_loss / (num_batches)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")
        # print(f"Sentiment train acc :: {sentiment_train_accuracy :.3f}, dev acc :: {sentiment_dev_accuracy :.3f}")
        # print(f"Paraphrase train acc :: {paraphrase_train_accuracy :.3f}, dev acc :: {paraphrase_dev_accuracy :.3f}")
        print(f"Similarity train corr :: {sts_train_corr :.3f}, dev acc :: {sts_dev_corr :.3f}")

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
