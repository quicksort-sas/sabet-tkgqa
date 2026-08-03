import pickle

import numpy as np
import torch

from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertTokenizerFast
import random
from torch.utils.data import Dataset, DataLoader
import random

from tkg_qa_models.hard_supervision_functions import retrieve_times
from tkg_qa_models import utils
import json
from collections import defaultdict
from datetime import datetime
import os

data_dir = "/Data/data"

class QA_Dataset(Dataset):
    def __init__(self, 
                split,
                dataset_name,
                tokenization_needed=True):
        filename = f'{data_dir}/data/{dataset_name}/questions/{split}.pickle'.format(
            dataset_name=dataset_name,
            split=split
        )
        questions = pickle.load(open(filename, 'rb'))
        
        self.tokenizer_class = DistilBertTokenizer 
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.all_dicts = utils.getAllDicts(dataset_name)
        print('Total questions = ', len(questions))

        self.data = questions
        self.tokenization_needed = tokenization_needed
        

    def getEntitiesLocations(self, question):
        question_text = question['question']
        entities = question['entities']
        ent2id = self.all_dicts['ent2id']
        loc_ent = []
        for e in entities:
            e_id = ent2id[e]
            location = question_text.find(e)
            loc_ent.append((location, e_id))
        return loc_ent

    def getTimesLocations(self, question):
        question_text = question['question']
        times = question['times']
        ts2id = self.all_dicts['ts2id']
        loc_time = []
        for t in times:
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id']) # add num entities
            location = question_text.find(str(t))
            loc_time.append((location, t_id))
        return loc_time

    def isTimeString(self, s):
        if 'Q' not in s:
            return True
        else:
            return False

    def textToEntTimeId(self, text):
        if self.isTimeString(text):
            t = int(text)
            ts2id = self.all_dicts['ts2id']
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id'])
            return t_id
        else:
            ent2id = self.all_dicts['ent2id']
            e_id = ent2id[text]
            return e_id


    def getOrderedEntityTimeIds(self, question):
        loc_ent = self.getEntitiesLocations(question)
        loc_time = self.getTimesLocations(question)
        loc_all = loc_ent + loc_time
        loc_all.sort()
        ordered_ent_time = [x[1] for x in loc_all]
        return ordered_ent_time

    def entitiesToIds(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            try:
                output.append(ent2id[e])
            except:
                pass
        return output
    
    def getIdType(self, id):
        if id < len(self.all_dicts['ent2id']):
            return 'entity'
        else:
            return 'time'
    
    def getEntityToText(self, entity_wd_id):
        return self.all_dicts['wd_id_to_text'][entity_wd_id]
    
    def getEntityIdToText(self, id):
        ent = self.all_dicts['id2ent'][id]
        return self.getEntityToText(ent)
    
    def getEntityIdToWdId(self, id):
        return self.all_dicts['id2ent'][id]

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        for t in times:
            try:
                output.append(ts2id[(t, 0, 0)])
            except:
                pass
        return output

    def getAnswersFromScores(self, scores, largest=True, k=10):
        _, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])
        return answers
    
    def getAnswersFromScoresWithScores(self, scores, largest=True, k=10):
        s, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])
        return s, answers

    def padding_tensor(self, sequences, max_len = -1):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        if max_len == -1:
            max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        # mask = sequences[0].data.new(*out_dims).fill_(0)
        mask = torch.ones((num, max_len), dtype=torch.bool) # fills with True
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = False # fills good area with False
        return out_tensor, mask
    
    def toOneHot(self, indices, vec_len):
        indices = torch.LongTensor(indices)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def prepare_data(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []
        entity_time_ids = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        for question in data:
            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]
            question_text.append(q_text)
            et_id = self.getOrderedEntityTimeIds(question)
            entity_time_ids.append(torch.tensor(et_id, dtype=torch.long))
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)
        return {'question_text': question_text, 
                'entity_time_ids': entity_time_ids, 
                'answers_arr': answers_arr}
    
    def is_template_keyword(self, word):
        if '{' in word and '}' in word:
            return True
        else:
            return False

    def get_keyword_dict(self, template, nl_question):
        template_tokenized = self.tokenize_template(template)
        keywords = []
        for word in template_tokenized:
            if not self.is_template_keyword(word):
                # replace only first occurence
                nl_question = nl_question.replace(word, '*', 1)
            else:
                keywords.append(word[1:-1]) # no brackets
        text_for_keywords = []
        for word in nl_question.split('*'):
            if word != '':
                text_for_keywords.append(word)
        keyword_dict = {}
        for keyword, text in zip(keywords, text_for_keywords):
            keyword_dict[keyword] = text
        return keyword_dict

    def addEntityAnnotation(self, data):
        for i in range(len(data)):
            question = data[i]
            keyword_dicts = [] # we want for each paraphrase
            template = question['template']
            #for nl_question in question['paraphrases']:
            nl_question =  question['paraphrases'][0]
            keyword_dict = self.get_keyword_dict(template, nl_question)
            keyword_dicts.append(keyword_dict)
            data[i]['keyword_dicts'] = keyword_dicts
            #print(keyword_dicts)
            #print(template, nl_question)
        return data

    def tokenize_template(self, template):
        output = []
        buffer = ''
        i = 0
        while i < len(template):
            c = template[i]
            if c == '{':
                if buffer != '':
                    output.append(buffer)
                    buffer = ''
                while template[i] != '}':
                    buffer += template[i]
                    i += 1
                buffer += template[i]
                output.append(buffer)
                buffer = ''
            else:
                buffer += c
            i += 1
        if buffer != '':
            output.append(buffer)
        return output


class QA_Dataset_Baseline(QA_Dataset):
    def __init__(self, split, dataset_name,  tokenization_needed=True):
        super().__init__(split, dataset_name, tokenization_needed)
        print('Preparing data for split %s' % split)
        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = {str(k[0]):v for k,v in self.all_dicts['ts2id'].items()}
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()

        self.prepared_data = self.prepare_data_(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.answer_vec_size = self.num_total_entities + self.num_total_times

        
    def prepare_data_(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter 
        question_text = []
        heads = []
        tails = []
        times = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        ent2id = self.all_dicts['ent2id']
        self.data_ids_filtered=[]
        # self.data=[]
        for i,question in enumerate(data):
            self.data_ids_filtered.append(i)

            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['paraphrases'][0]
            
            
            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in entities_list_with_locations] # ordering necessary otherwise set->list conversion causes randomness
            
            if len(entities) == 0:
                head, tail = 0, 0
            else:
                head = entities[0] # take an entity
                if len(entities) > 1:
                    tail = entities[1]
                else:
                    tail = entities[0]
            
            times_in_question = question['times']
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)[0] # take a time. if no time then 0
                # exit(0)
            else:
                time = 0
                
            
            time += num_total_entities
            heads.append(head)
            times.append(time)
            tails.append(tail)
            question_text.append(q_text)
            
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            if len(answers) == 0:
                answers = [num_total_entities]
            answers_arr.append(answers)
            
        # answers_arr = self.get_stacked_answers_long(answers_arr)
        self.data=[self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text, 
                'head': heads, 
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr}
    def print_prepared_data(self):
        for k, v in self.prepared_data.items():
            print(k, v)

    def __len__(self):
        return len(self.data)
        # return len(self.prepared_data['question_text'])

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, answers_single #,answers_khot

    def _collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        times = torch.from_numpy(np.array([item[3] for item in items]))
        answers_single = torch.from_numpy(np.array([item[4] for item in items]))
        return b['input_ids'], b['attention_mask'], heads, tails, times, answers_single
    def get_dataset_ques_info(self):
        type2num={}
        for question in self.data:
            if question["type"] not in type2num: type2num[question["type"]]=0
            type2num[question["type"]]+=1
        return {"type2num":type2num, "total_num":len(self.data_ids_filtered)}.__str__()


class QA_Dataset_TimeQuestions(QA_Dataset):
    def __init__(self, split, dataset_name, args, tokenization_needed=True, replace_entity_spans_with_mask=True):
        super().__init__(split, dataset_name, tokenization_needed)
        print('Preparing data for split %s' % split)

        self.replace_entity_spans_with_mask = replace_entity_spans_with_mask
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')  # for fast offset-based tokenization

        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = {str(k[0]):v for k,v in self.all_dicts['ts2id'].items()}
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()
        self.split = split

        self.load_tkg_facts()

        #args: given TKG, whether to corrupt hard, and how to use the reitrieved timestmaps
        self.data = retrieve_times(args.tkg_file, args.dataset_name, self.data, args.corrupt_hard, args.fuse)

        print(self.data[0])
        print(self.data[1])
        
        self.data = self.addEntityAnnotation(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.padding_idx = self.num_total_entities + self.num_total_times  # padding id for embedding of ent/time
        self.answer_vec_size = self.num_total_entities + self.num_total_times
          
        self.prepared_data = self.prepare_data2(self.data)

    def __len__(self):
        return len(self.data)

    def get_event_time(self, event):
        c = self.event2time[event]
        event_triples = list(c)
        time = event_triples[0][3]
        return time

    def check_triples(self, q1, q2):
        l = []
        if type(q2) == list:
            for i in q2:
                l += self.check_triples(q1,i)
        else:
            for i in self.e2rt[(q1,q2)]:
                l.append(i)
            for i in self.e2rt[(q2,q1)]:
                l.append(i)
        return l

    def extract_head_tail_from_annotation(self, question):
        """
        Extract head and tail entities from question annotation.
        Returns: (head, tail, has_head, has_tail)
        """
        annotation = question.get('annotation', {})
        
        head = None
        tail = None
        has_head = False
        has_tail = False
        
        # Check for head
        if 'head' in annotation:
            head = annotation['head']
            has_head = True
        
        # Check for tail
        if 'tail' in annotation:
            tail = annotation['tail']
            has_tail = True
        
        return head, tail, has_head, has_tail

    def process_question_entities(self, question):
        """
        Process a single question to extract head and tail entities
        """
        head = 0
        tail = 0
        has_head = False
        has_tail = False
        
        # First try from annotation
        head_ann, tail_ann, has_head_ann, has_tail_ann = self.extract_head_tail_from_annotation(question)
        
        if has_head_ann:
            # head = head_ann
            head = self.all_dicts['ent2id'][head_ann]
            has_head = True
        if has_tail_ann:
            # tail = tail_ann
            tail = self.all_dicts['ent2id'][tail_ann]
            has_tail = True
        
        if not has_head and has_tail:
            head = tail
        if not has_tail and has_head:
            tail = head
        
        if not has_head and not has_tail:
            entities = list(question.get('entities', []))
            
            if len(entities) > 0:
                head = entities[0]
                head = self.all_dicts['ent2id'][head]
                if len(entities) > 1:
                    tail = entities[1]
                    tail = self.all_dicts['ent2id'][tail]
                else:
                    tail = head
        
        return head, tail, has_head, has_tail

    def get_neighbours(self, e):
        tr = self.e2tr[e]
        neighbours = []
        for t in tr:
            neighbours.append(t[0])
            neighbours.append(t[2])
        neighbours = set(neighbours)
        neighbours.remove(e)
        return list(neighbours)

    def implicit_parsing(self, data):
        # general extraction
        for i in data:
            b = list(i['annotation'].keys())
            if 'event_head' in b:
                c = i['annotation']['event_head']
                if c[0]!='Q':
                    continue
                time = self.get_event_time(c)
                if i['type'] != 'before_after':
                    i['times'] = {int(time)}
                i['annotation']['time'] = time
                i['annotation']['event_head_bak'] = i['annotation']['event_head']
                i['annotation']['event_head'] = time
                i['paraphrases'][0] = i['paraphrases'][0].replace(self.all_dicts['wd_id_to_text'][c],time)

        # speific extraction
        for i in data:
            if i['type'] == 'before_after':
                if 'event_head' not in i['annotation'].keys():
                    head = i['annotation']['head']
                    tail = i['annotation']['tail']
                    related_triples = self.check_triples(head,tail)
                    if i['annotation']['type'] == 'before':
                        index = 0
                        time = related_triples[0][3]
                    else:
                        time = related_triples[-1][4]
                    # i['times'] = {int(time)}
                    #i['annotation']['time'] = time
                    # NL replace
                    text = self.all_dicts['wd_id_to_text'][head]
                    i['paraphrases'][0] = i['paraphrases'][0].replace(text,time)

    def getEntityTimeTextIds(self, question, pp_id=0):
        keyword_dict = question['keyword_dicts'][pp_id]
        keyword_id_dict = question['annotation']  # this does not depend on paraphrase
        output_text = []
        output_ids = []
        entity_time_keywords = set(['head', 'tail', 'time', 'event_head', 'time1', 'time2'])
        
        #print(keyword_dict, keyword_id_dict)
        for keyword, value in keyword_dict.items():
            if keyword in entity_time_keywords:
                wd_id_or_time = keyword_id_dict[keyword]
                output_text.append(value)
                output_ids.append(wd_id_or_time)
                
        #print(output_text, output_ids)
        return output_text, output_ids

    def get_entity_aware_tokenization(self, nl_question, ent_times, ent_times_ids):
        
        spans = []
        used_spans = []

        for e_text, e_id in zip(ent_times, ent_times_ids):
            if not e_text:
                continue

            search_from = 0
            while True:
                start = nl_question.find(e_text, search_from)
                if start == -1:
                    break

                end = start + len(e_text)
                overlap = any(s < end and start < e for s, e in used_spans)

                if not overlap:
                    spans.append((start, end, e_id, e_text))
                    used_spans.append((start, end))
                    break

                search_from = start + 1

        spans.sort(key=lambda x: x[0])


        if not self.replace_entity_spans_with_mask:
            enc = self.tokenizer(
                nl_question,
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                truncation=False,
            )

            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            tokenized = self.tokenizer.convert_ids_to_tokens(input_ids)
            entity_time_final = [self.padding_idx] * len(tokenized)
            entity_mask = [1.0] * len(tokenized)

            for _, _, raw_id, _ in spans:
                matrix_id = self.textToEntTimeId(raw_id)
                for i, (s, e) in enumerate(offsets):
                    if s == 0 and e == 0:  # special tokens
                        continue
                    for span_start, span_end, _, _ in spans:
                        if s < span_end and e > span_start:
                            entity_time_final[i] = matrix_id
                            entity_mask[i] = 0.0
                            break

            assert len(tokenized) == len(entity_time_final) == len(entity_mask), (
                f"Alignment mismatch: {len(tokenized)} / {len(entity_time_final)} / {len(entity_mask)}"
            )
            return tokenized, entity_time_final, entity_mask


        pieces = []
        arr = []
        cursor = 0

        for start, end, raw_id, _ in spans:
            if cursor < start:
                chunk = nl_question[cursor:start]
                if chunk != "":
                    pieces.append(chunk)
                    arr.append(self.padding_idx)

            pieces.append(self.tokenizer.mask_token)
            arr.append(self.textToEntTimeId(raw_id))
            cursor = end

        if cursor < len(nl_question):
            chunk = nl_question[cursor:]
            if chunk != "":
                pieces.append(chunk)
                arr.append(self.padding_idx)

        tokenized, valid_ids = self.tokenize(pieces)

        entity_time_final = []
        idx = 0
        for vid in valid_ids:
            if vid == 0:
                entity_time_final.append(self.padding_idx)
            else:
                entity_time_final.append(arr[idx])
                idx += 1

        entity_mask = [1.0 if x == self.padding_idx else 0.0 for x in entity_time_final]

        assert len(tokenized) == len(entity_time_final) == len(entity_mask), (
            f"Alignment mismatch: {len(tokenized)} / {len(entity_time_final)} / {len(entity_mask)}"
        )
        return tokenized, entity_time_final, entity_mask
    
    def extract_all_times_from_entities(self, entities):
         
        
        times_set = set()
        
        # Check all pairs of entities
        for i, e1 in enumerate(entities):
            for e2 in entities[i:]:  # Include self-pairs and all combinations
                # Look for facts connecting e1 and e2 (in either direction)
                pair_key = (min(e1, e2), max(e1, e2))
                
                if pair_key in self.tkg_entity_pairs:
                    times_set.update(self.tkg_entity_pairs[pair_key])
        
        # Convert times to IDs and add offset
        if times_set:
            time_ids = self.timesToIds(list(times_set))
            # Add num_total_entities offset if needed (based on your usage)
            sorted_times = sorted([t + self.num_total_entities for t in time_ids])
        else:
            sorted_times = [0]
        
        return sorted_times

    def load_tkg_facts(self, tkg_path=f'{data_dir}/data/timequestions/kg/full.txt'):
        """
        Load TKG facts and build index for quick lookup.
        Expected format: head, relation, tail, start_time, end_time (tab-separated)
        """
        self.tkg_entity_pairs = {}  # (min_ent, max_ent) -> set of times
        
        print(f"Loading TKG facts from {tkg_path}...")
        
        with open(tkg_path, 'r') as f:
            for line in tqdm(f, desc="Loading TKG facts"):
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    head = parts[0]
                    relation = parts[1]
                    tail = parts[2]
                    start_time = int(parts[3])
                    end_time = int(parts[4])
                    
                    # Convert to IDs if they're strings (adjust based on your format)
                    # You may need to use self.all_dicts['ent2id'] to convert
                    if isinstance(head, str) and head in self.all_dicts['ent2id'] and isinstance(tail, str) and tail in self.all_dicts['ent2id']:
                        head_id = self.all_dicts['ent2id'][head]
                        tail_id = self.all_dicts['ent2id'][tail]
                    else:
                        head_id = head
                        tail_id = tail
                    
                    # Store times for this entity pair
                    pair_key = (min(head_id, tail_id), max(head_id, tail_id))
                    if pair_key not in self.tkg_entity_pairs:
                        self.tkg_entity_pairs[pair_key] = set()
                    
                    # Add all times in range [start_time, end_time]
                    for t in range(start_time, end_time + 1):
                        self.tkg_entity_pairs[pair_key].add(t)
        
        print(f"Loaded {len(self.tkg_entity_pairs)} entity pairs with time facts")


    def prepare_data2(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter
        heads = []
        times = []
        start_times = []
        end_times = []
        tails = []
        tails2 = []
        types = []
        rels = []
        question_text = []
        tokenized_question = []
        entity_time_ids_tokenized_question = []
        entity_mask_tokenized_question = []
        pp_id = 0
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        for question in tqdm(data):
            pp_id = 0
            nl_question = question['paraphrases'][pp_id]
            q_text = nl_question
            et_text, et_ids = self.getEntityTimeTextIds(question, pp_id)

            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in
                        entities_list_with_locations]  # ordering necessary otherwise set->list conversion causes randomness
            
            head, tail, has_head, has_tail = self.process_question_entities(question)

            tail2 = tail

            
    
            times_in_question = list(question['times'])
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)[0]
                start_time = time
                end_time = time
            else:
                time = 0
                
                entities = list(set(question.get('entities', [])))
                if len(entities) > 0:
                    sorted_times = self.extract_all_times_from_entities(entities)
                    start_time = sorted_times[0]
                    end_time = sorted_times[-1]
                else:
                    start_time = 0
                    end_time = 0


            time += num_total_entities

            ###########
            # One-time random swap during dataset construction
            # if random.random() < 0.5:
            #     head, tail = tail, head
            ###########

           
            if question['answer_type'] == 'entity':
                # print(question)
                answers = self.entitiesToIds(question['answers'])
            else:
                # print(question)
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            
            if len(answers) > 0:
                heads.append(head)
                times.append(time)
                start_times.append(start_time)
                end_times.append(end_time)
                tails.append(tail)
                tails2.append(tail2)
                types.append(question['type'])
                # rel = self.all_dicts['rel2id'][list(question['relations'])[0]]
                # rels.append(rel)
                tokenized, entity_time_final, entity_mask = self.get_entity_aware_tokenization(nl_question, et_text, et_ids)
                assert len(tokenized) == len(entity_time_final)
                question_text.append(nl_question)
                tokenized_question.append(self.tokenizer.convert_tokens_to_ids(tokenized))
                entity_mask_tokenized_question.append(entity_mask)
                entity_time_ids_tokenized_question.append(entity_time_final)
                answers_arr.append(answers)
            
            else:
                heads.append(head)
                times.append(time)
                start_times.append(start_time)
                end_times.append(end_time)
                tails.append(tail)
                tails2.append(tail2)
                types.append(question['type'])
                # rel = self.all_dicts['rel2id'][list(question['relations'])[0]]
                # rels.append(rel)
                tokenized, entity_time_final, entity_mask = self.get_entity_aware_tokenization(nl_question, et_text, et_ids)
                assert len(tokenized) == len(entity_time_final)
                question_text.append(nl_question)
                tokenized_question.append(self.tokenizer.convert_tokens_to_ids(tokenized))
                entity_mask_tokenized_question.append(entity_mask)
                entity_time_ids_tokenized_question.append(entity_time_final)
                answers_arr.append([num_total_entities])
            
            
        return {'question_text': question_text,
                'tokenized_question': tokenized_question,
                'entity_time_ids': entity_time_ids_tokenized_question,
                'entity_mask': entity_mask_tokenized_question,
                'head': heads,
                'tail': tails,
                'time': times,
                'start_time': start_times,
                'end_time': end_times,
                'tail2': tails2,
                'types':types,
                'rels':rels,
                'answers_arr': answers_arr}

    # tokenization function taken from NER code
    def tokenize(self, words):
        """ tokenize input"""
        tokens = []
        valid_positions = []
        tokens.append(self.tokenizer.cls_token)
        valid_positions.append(0)
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        tokens.append(self.tokenizer.sep_token)
        valid_positions.append(0)
        return tokens, valid_positions

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        entity_time_ids = np.array(data['entity_time_ids'][index], dtype=np.int64)
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        # answers_khot = self.toOneHot(answers_arr, self.answer_vec_size)
        tokenized_question = data['tokenized_question'][index]
        entity_mask = data['entity_mask'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        tail2 = data['tail2'][index]
        time = data['time'][index]
        start_time = data['start_time'][index]
        end_time = data['end_time'][index]
        types = data['types'][index]
        # rels = data['rels'][index]
        rels = 0
        return question_text, tokenized_question, entity_time_ids, entity_mask, head, tail, time, start_time, end_time, tail2, types, rels, answers_single

    def pad_for_batch(self, to_pad, padding_val, dtype=np.int64):
        padded = np.ones([len(to_pad), len(max(to_pad, key=lambda x: len(x)))], dtype=dtype) * padding_val
        for i, j in enumerate(to_pad):
            padded[i][0:len(j)] = j
        return padded

    # do this before padding for batch
    def get_attention_mask(self, tokenized):
        # first make zeros array of appropriate size
        mask = np.zeros([len(tokenized), len(max(tokenized, key=lambda x: len(x)))], dtype=np.int64)
        # now set ones everywhere needed
        for i, j in enumerate(tokenized):
            mask[i][0:len(j)] = np.ones(len(j), dtype=np.int64)
        return mask

    def _collate_fn(self, items):
        # please don't tokenize again
        # b = self.tokenizer(batch_sentences, padding=True, truncation=False, return_tensors="pt")

        tokenized_questions = [item[1] for item in items]
        attention_mask = torch.from_numpy(self.get_attention_mask(tokenized_questions))
        input_ids = torch.from_numpy(self.pad_for_batch(tokenized_questions, self.tokenizer.pad_token_id, np.int64))

        entity_time_ids_list = [item[2] for item in items]
        entity_time_ids_padded = self.pad_for_batch(entity_time_ids_list, self.padding_idx, np.int64)
        entity_time_ids_padded = torch.from_numpy(entity_time_ids_padded)

        entity_mask = [item[3] for item in items]  # 0 if entity, 1 if not
        entity_mask_padded = self.pad_for_batch(entity_mask, 1.0,
                                                np.float32)  # doesnt matter probably cuz attention mask will be used. maybe pad with 1?
        entity_mask_padded = torch.from_numpy(entity_mask_padded)
        # can make foll mask in forward function using attention mask
        # entity_time_ids_padded_mask = ~(attention_mask.bool())

        heads = torch.from_numpy(np.array([item[4] for item in items]))
        tails = torch.from_numpy(np.array([item[5] for item in items]))
        times = torch.from_numpy(np.array([item[6] for item in items]))
        start_times = torch.from_numpy(np.array([item[7] for item in items]))
        end_times = torch.from_numpy(np.array([item[8] for item in items]))
        
        tails2 = torch.from_numpy(np.array([item[9] for item in items]))
        types = [item[10] for item in items]
        rels = torch.from_numpy(np.array([item[11] for item in items]))
        # answers_khot = torch.stack([item[4] for item in items])
        answers_single = torch.from_numpy(np.array([item[12] for item in items]))

        return input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types,rels, answers_single
 

class QA_Dataset_SubGTR(QA_Dataset):
    def __init__(self, split, dataset_name, args, tokenization_needed=True, replace_entity_spans_with_mask=True):
        super().__init__(split, dataset_name, tokenization_needed)
        print('Preparing data for split %s' % split)

        self.replace_entity_spans_with_mask = replace_entity_spans_with_mask
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')  # for fast offset-based tokenization

        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = {str(k[0]):v for k,v in self.all_dicts['ts2id'].items()}
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()
        self.split = split

        # Aware module
        if args.aware_module:
            with open(f'{data_dir}/saved_pkl/e2rt.pkl', 'rb') as f:
                self.e2rt = pickle.load(f)
            with open(f'{data_dir}/saved_pkl/event2time.pkl', 'rb') as f:
                self.event2time = pickle.load(f)
            with open(f'{data_dir}/saved_pkl/e2tr.pkl', 'rb') as f:
                self.e2tr = pickle.load(f)
            self.implicit_parsing(self.data)

        #args: given TKG, whether to corrupt hard, and how to use the reitrieved timestmaps
        self.data = retrieve_times(args.tkg_file, args.dataset_name, self.data, args.corrupt_hard, args.fuse)
        
        
        self.data = self.addEntityAnnotation(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.padding_idx = self.num_total_entities + self.num_total_times  # padding id for embedding of ent/time
        self.answer_vec_size = self.num_total_entities + self.num_total_times
          
        self.prepared_data = self.prepare_data2(self.data)

    def __len__(self):
        return len(self.data)

    def get_event_time(self, event):
        c = self.event2time[event]
        event_triples = list(c)
        time = event_triples[0][3]
        return time

    def check_triples(self, q1, q2):
        l = []
        if type(q2) == list:
            for i in q2:
                l += self.check_triples(q1,i)
        else:
            for i in self.e2rt[(q1,q2)]:
                l.append(i)
            for i in self.e2rt[(q2,q1)]:
                l.append(i)
        return l

    def get_neighbours(self, e):
        tr = self.e2tr[e]
        neighbours = []
        for t in tr:
            neighbours.append(t[0])
            neighbours.append(t[2])
        neighbours = set(neighbours)
        neighbours.remove(e)
        return list(neighbours)

    def implicit_parsing(self, data):
        # general extraction
        for i in data:
            b = list(i['annotation'].keys())
            if 'event_head' in b:
                c = i['annotation']['event_head']
                if c[0]!='Q':
                    continue
                time = self.get_event_time(c)
                if i['type'] != 'before_after':
                    i['times'] = {int(time)}
                i['annotation']['time'] = time
                i['annotation']['event_head_bak'] = i['annotation']['event_head']
                i['annotation']['event_head'] = time
                i['paraphrases'][0] = i['paraphrases'][0].replace(self.all_dicts['wd_id_to_text'][c],time)

        # speific extraction
        for i in data:
            if i['type'] == 'before_after':
                if 'event_head' not in i['annotation'].keys():
                    head = i['annotation']['head']
                    tail = i['annotation']['tail']
                    related_triples = self.check_triples(head,tail)
                    if i['annotation']['type'] == 'before':
                        index = 0
                        time = related_triples[0][3]
                    else:
                        time = related_triples[-1][4]
                    # i['times'] = {int(time)}
                    #i['annotation']['time'] = time
                    # NL replace
                    text = self.all_dicts['wd_id_to_text'][head]
                    i['paraphrases'][0] = i['paraphrases'][0].replace(text,time)

    def getEntityTimeTextIds(self, question, pp_id=0):
        keyword_dict = question['keyword_dicts'][pp_id]
        keyword_id_dict = question['annotation']  # this does not depend on paraphrase
        output_text = []
        output_ids = []
        entity_time_keywords = set(['head', 'tail', 'time', 'event_head', 'time1', 'time2'])
        
        #print(keyword_dict, keyword_id_dict)
        for keyword, value in keyword_dict.items():
            if keyword in entity_time_keywords:
                wd_id_or_time = keyword_id_dict[keyword]
                output_text.append(value)
                output_ids.append(wd_id_or_time)
                
        #print(output_text, output_ids)
        return output_text, output_ids

    def get_entity_aware_tokenization(self, nl_question, ent_times, ent_times_ids):
        """
        Mirrors QA_Dataset_MultiQA_Advanced behaviour as closely as possible
        while keeping the SubGTR signature:

        - replace_entity_spans_with_mask = False:
            keep original text, mark every wordpiece that overlaps an entity/time span
        - replace_entity_spans_with_mask = True:
            replace spans with [MASK], then tokenize piecewise and keep first-subtoken
            alignment exactly like the original SubGTR logic
        """

        # ------------------------------------------------------------------
        # Build ordered spans from surface strings.
        # Because SubGTR only gives strings (not char offsets), we recover
        # spans left-to-right and avoid overlaps.
        # Each item: (start_char, end_char, raw_id, surface_text)
        # ------------------------------------------------------------------
        spans = []
        used_spans = []

        for e_text, e_id in zip(ent_times, ent_times_ids):
            if not e_text:
                continue

            search_from = 0
            while True:
                start = nl_question.find(e_text, search_from)
                if start == -1:
                    break

                end = start + len(e_text)
                overlap = any(s < end and start < e for s, e in used_spans)

                if not overlap:
                    spans.append((start, end, e_id, e_text))
                    used_spans.append((start, end))
                    break

                search_from = start + 1

        spans.sort(key=lambda x: x[0])

        # ------------------------------------------------------------------
        # Mode 1: keep original text, annotate all overlapping wordpieces
        # ------------------------------------------------------------------
        if not self.replace_entity_spans_with_mask:
            enc = self.tokenizer(
                nl_question,
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                truncation=False,
            )

            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            tokenized = self.tokenizer.convert_ids_to_tokens(input_ids)
            entity_time_final = [self.padding_idx] * len(tokenized)
            entity_mask = [1.0] * len(tokenized)

            for _, _, raw_id, _ in spans:
                matrix_id = self.textToEntTimeId(raw_id)
                for i, (s, e) in enumerate(offsets):
                    if s == 0 and e == 0:  # special tokens
                        continue
                    for span_start, span_end, _, _ in spans:
                        if s < span_end and e > span_start:
                            entity_time_final[i] = matrix_id
                            entity_mask[i] = 0.0
                            break

            assert len(tokenized) == len(entity_time_final) == len(entity_mask), (
                f"Alignment mismatch: {len(tokenized)} / {len(entity_time_final)} / {len(entity_mask)}"
            )
            return tokenized, entity_time_final, entity_mask

        # ------------------------------------------------------------------
        # Mode 2: SubGTR-style masking
        # ------------------------------------------------------------------
        pieces = []
        arr = []
        cursor = 0

        for start, end, raw_id, _ in spans:
            if cursor < start:
                chunk = nl_question[cursor:start]
                if chunk != "":
                    pieces.append(chunk)
                    arr.append(self.padding_idx)

            pieces.append(self.tokenizer.mask_token)
            arr.append(self.textToEntTimeId(raw_id))
            cursor = end

        if cursor < len(nl_question):
            chunk = nl_question[cursor:]
            if chunk != "":
                pieces.append(chunk)
                arr.append(self.padding_idx)

        tokenized, valid_ids = self.tokenize(pieces)

        entity_time_final = []
        idx = 0
        for vid in valid_ids:
            if vid == 0:
                entity_time_final.append(self.padding_idx)
            else:
                entity_time_final.append(arr[idx])
                idx += 1

        entity_mask = [1.0 if x == self.padding_idx else 0.0 for x in entity_time_final]

        assert len(tokenized) == len(entity_time_final) == len(entity_mask), (
            f"Alignment mismatch: {len(tokenized)} / {len(entity_time_final)} / {len(entity_mask)}"
        )
        return tokenized, entity_time_final, entity_mask
    
    # def get_entity_aware_tokenization(self, nl_question, ent_times, ent_times_ids):
    # # Build ordered entity/time spans (same for both modes)
    #     index_et_pairs = []
    #     index_et_text_pairs = []
    #     for e_text, e_id in zip(ent_times, ent_times_ids):
    #         location = nl_question.find(e_text)
    #         index_et_pairs.append((location, e_id))
    #         index_et_text_pairs.append((location, e_text))
    #     index_et_pairs.sort()
    #     index_et_text_pairs.sort()

    #     my_tokenized_question = []
    #     start_index = 0
    #     arr = []
    #     for pair, pair_id in zip(index_et_text_pairs, index_et_pairs):
    #         end_index = pair[0]
    #         if nl_question[start_index: end_index] != '':
    #             my_tokenized_question.append(nl_question[start_index: end_index])
    #             arr.append(self.padding_idx)
    #         start_index = end_index
    #         end_index = start_index + len(pair[1])

    #         # ------------------------------------------------------------------
    #         # Toggle: [MASK] vs. original text
    #         # ------------------------------------------------------------------
    #         if getattr(self, 'replace_entity_spans_with_mask', True):
    #             my_tokenized_question.append(self.tokenizer.mask_token)
    #         else:
    #             my_tokenized_question.append(nl_question[start_index: end_index])

    #         matrix_id = self.textToEntTimeId(pair_id[1])  # get id in embedding matrix
    #         arr.append(matrix_id)
    #         start_index = end_index

    #     if nl_question[start_index:] != '':
    #         my_tokenized_question.append(nl_question[start_index:])
    #         arr.append(self.padding_idx)

    #     # =====================================================================
    #     # Mode 1: SubGTR-style masking (original logic, unchanged)
    #     # =====================================================================
    #     if getattr(self, 'replace_entity_spans_with_mask', True):
    #         tokenized, valid_ids = self.tokenize(my_tokenized_question)
    #         entity_time_final = []
    #         index = 0
    #         for vid in valid_ids:
    #             if vid == 0:
    #                 entity_time_final.append(self.padding_idx)
    #             else:
    #                 entity_time_final.append(arr[index])
    #                 index += 1
    #         entity_mask = [1. if x == self.padding_idx else 0. for x in entity_time_final]
    #         return tokenized, entity_time_final, entity_mask

    #     # =====================================================================
    #     # Mode 2: Keep original tokens, mark ALL sub-tokens of each span
    #     # =====================================================================
    #     tokenized = [self.tokenizer.cls_token]
    #     entity_time_final = [self.padding_idx]
    #     entity_mask = [1.0]

    #     for piece, piece_id in zip(my_tokenized_question, arr):
    #         piece_tokens = self.tokenizer.tokenize(piece)
    #         tokenized.extend(piece_tokens)
    #         entity_time_final.extend([piece_id] * len(piece_tokens))
    #         entity_mask.extend([1.0 if piece_id == self.padding_idx else 0.0] * len(piece_tokens))

    #     tokenized.append(self.tokenizer.sep_token)
    #     entity_time_final.append(self.padding_idx)
    #     entity_mask.append(1.0)

    #     assert len(tokenized) == len(entity_time_final) == len(entity_mask), \
    #         f"Alignment mismatch: {len(tokenized)} / {len(entity_time_final)} / {len(entity_mask)}"
    #     return tokenized, entity_time_final, entity_mask
    
    
    def prepare_data2(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter
        heads = []
        times = []
        start_times = []
        end_times = []
        tails = []
        tails2 = []
        types = []
        rels = []
        question_text = []
        tokenized_question = []
        entity_time_ids_tokenized_question = []
        entity_mask_tokenized_question = []
        pp_id = 0
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        for question in tqdm(data):
            # randomly sample pp
            # in test there is only 1 pp, so always pp_id=0
            # TODO: this random is causing assertion bug later on
            # pp_id = random.randint(0, len(question['paraphrases']) - 1)
            pp_id = 0
            nl_question = question['paraphrases'][pp_id]
            q_text = nl_question
            et_text, et_ids = self.getEntityTimeTextIds(question, pp_id)

            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in
                        entities_list_with_locations]  # ordering necessary otherwise set->list conversion causes randomness
            
            if len(entities) == 0:
                head, tail = 0, 0
                print('No Ent \n')
                print(question)
                print('\n')
            else:
                head = entities[0]  # take an entity
                if len(entities) > 1:
                    tail = entities[1]
                    if len(entities) > 2:
                        tail2 = entities[2]
                    else:
                        tail2 = tail
                else:
                    tail = entities[0]
                    tail2 = tail
    
            times_in_question = list(question['times'])
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)[0]  # take a time. if no time then 0
                start_time = time
                end_time = time
                # exit(0)
            else:
                time = 0
                #check for retrieved timestmaps
                if len(question['fact']) > 0:
                    ts = []
                    #add all timestmaps and sort
                    for f in question['fact']:
                        for t in range(int(f[0]), int(f[1])+1):
                            ts.append(t)

                    ts = sorted(ts)
                    sorted_times = self.timesToIds(ts)
                    
                    try:
                        start_time = sorted_times[0]   # for random random.choice(sorted_times)
                    except:
                        start_time = 0
                    try:
                        end_time = sorted_times[-1]
                    except:
                        end_time = 0
                else:
                    start_time = 0
                    end_time = 0

                # print('No time in qn!')


            time += num_total_entities

            ###########
            # One-time random swap during dataset construction
            # if random.random() < 0.5:
            #     head, tail = tail, head
            ###########

           
            if question['answer_type'] == 'entity':
                # print(question)
                answers = self.entitiesToIds(question['answers'])
            else:
                # print(question)
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            
            if len(answers) > 0:
                heads.append(head)
                times.append(time)
                start_times.append(start_time)
                end_times.append(end_time)
                tails.append(tail)
                tails2.append(tail2)
                types.append(question['type'])
                # rel = self.all_dicts['rel2id'][list(question['relations'])[0]]
                # rels.append(rel)
                tokenized, entity_time_final, entity_mask = self.get_entity_aware_tokenization(nl_question, et_text, et_ids)
                assert len(tokenized) == len(entity_time_final)
                question_text.append(nl_question)
                tokenized_question.append(self.tokenizer.convert_tokens_to_ids(tokenized))
                entity_mask_tokenized_question.append(entity_mask)
                entity_time_ids_tokenized_question.append(entity_time_final)
                answers_arr.append(answers)
            
            else:
                heads.append(head)
                times.append(time)
                start_times.append(start_time)
                end_times.append(end_time)
                tails.append(tail)
                tails2.append(tail2)
                types.append(question['type'])
                # rel = self.all_dicts['rel2id'][list(question['relations'])[0]]
                # rels.append(rel)
                tokenized, entity_time_final, entity_mask = self.get_entity_aware_tokenization(nl_question, et_text, et_ids)
                assert len(tokenized) == len(entity_time_final)
                question_text.append(nl_question)
                tokenized_question.append(self.tokenizer.convert_tokens_to_ids(tokenized))
                entity_mask_tokenized_question.append(entity_mask)
                entity_time_ids_tokenized_question.append(entity_time_final)
                answers_arr.append([num_total_entities])
            
            
        return {'question_text': question_text,
                'tokenized_question': tokenized_question,
                'entity_time_ids': entity_time_ids_tokenized_question,
                'entity_mask': entity_mask_tokenized_question,
                'head': heads,
                'tail': tails,
                'time': times,
                'start_time': start_times,
                'end_time': end_times,
                'tail2': tails2,
                'types':types,
                'rels':rels,
                'answers_arr': answers_arr}

    # tokenization function taken from NER code
    def tokenize(self, words):
        """ tokenize input"""
        tokens = []
        valid_positions = []
        tokens.append(self.tokenizer.cls_token)
        valid_positions.append(0)
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        tokens.append(self.tokenizer.sep_token)
        valid_positions.append(0)
        return tokens, valid_positions

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        entity_time_ids = np.array(data['entity_time_ids'][index], dtype=np.int64)
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        # answers_khot = self.toOneHot(answers_arr, self.answer_vec_size)
        tokenized_question = data['tokenized_question'][index]
        entity_mask = data['entity_mask'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        tail2 = data['tail2'][index]
        time = data['time'][index]
        start_time = data['start_time'][index]
        end_time = data['end_time'][index]
        types = data['types'][index]
        # rels = data['rels'][index]
        rels = 0
        return question_text, tokenized_question, entity_time_ids, entity_mask, head, tail, time, start_time, end_time, tail2, types, rels, answers_single

    def pad_for_batch(self, to_pad, padding_val, dtype=np.int64):
        padded = np.ones([len(to_pad), len(max(to_pad, key=lambda x: len(x)))], dtype=dtype) * padding_val
        for i, j in enumerate(to_pad):
            padded[i][0:len(j)] = j
        return padded

    # do this before padding for batch
    def get_attention_mask(self, tokenized):
        # first make zeros array of appropriate size
        mask = np.zeros([len(tokenized), len(max(tokenized, key=lambda x: len(x)))], dtype=np.int64)
        # now set ones everywhere needed
        for i, j in enumerate(tokenized):
            mask[i][0:len(j)] = np.ones(len(j), dtype=np.int64)
        return mask

    def _collate_fn(self, items):
        # please don't tokenize again
        # b = self.tokenizer(batch_sentences, padding=True, truncation=False, return_tensors="pt")

        tokenized_questions = [item[1] for item in items]
        attention_mask = torch.from_numpy(self.get_attention_mask(tokenized_questions))
        input_ids = torch.from_numpy(self.pad_for_batch(tokenized_questions, self.tokenizer.pad_token_id, np.int64))

        entity_time_ids_list = [item[2] for item in items]
        entity_time_ids_padded = self.pad_for_batch(entity_time_ids_list, self.padding_idx, np.int64)
        entity_time_ids_padded = torch.from_numpy(entity_time_ids_padded)

        entity_mask = [item[3] for item in items]  # 0 if entity, 1 if not
        entity_mask_padded = self.pad_for_batch(entity_mask, 1.0,
                                                np.float32)  # doesnt matter probably cuz attention mask will be used. maybe pad with 1?
        entity_mask_padded = torch.from_numpy(entity_mask_padded)
        # can make foll mask in forward function using attention mask
        # entity_time_ids_padded_mask = ~(attention_mask.bool())

        heads = torch.from_numpy(np.array([item[4] for item in items]))
        tails = torch.from_numpy(np.array([item[5] for item in items]))
        times = torch.from_numpy(np.array([item[6] for item in items]))
        start_times = torch.from_numpy(np.array([item[7] for item in items]))
        end_times = torch.from_numpy(np.array([item[8] for item in items]))
        
        tails2 = torch.from_numpy(np.array([item[9] for item in items]))
        types = [item[10] for item in items]
        rels = torch.from_numpy(np.array([item[11] for item in items]))
        # answers_khot = torch.stack([item[4] for item in items])
        answers_single = torch.from_numpy(np.array([item[12] for item in items]))

        return input_ids, attention_mask, entity_time_ids_padded, entity_mask_padded, heads, tails, times, start_times, end_times, tails2, types,rels, answers_single
    

class QA_Dataset_multi(Dataset):
    def __init__(self,
                 split,
                 dataset_name='MultiTQ',
                 tokenization_needed=True, args=None):
        filename = f'{data_dir}/data/{dataset_name}/questions/processed_questions/{split}.json'.format(dataset_name=dataset_name, split=split)

        with open(filename, 'r') as obj:
            questions = json.load(obj)
        # questions = [x for x in questions if x['qtype'] == 'equal' and x['time_level'] == 'day']
        # questions = [x for x in questions if x['qlabel'] == 'Multiple' and x['time_level'] == 'day']


        self.tokenizer_class = DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.all_dicts = utils.getAllDicts(dataset_name)
        print('Total questions = ', len(questions))
        self.data = questions
        self.tokenization_needed = tokenization_needed

    def getEntitiesLocations(self, question):

        question_text = question['question']
        entities = question['entity_positions']
        ent2id = self.all_dicts['ent2id']
        loc_ent = []
        for e in entities:
            e_id = ent2id[e['entity'][0]]
            location = e['position'][0]
            loc_ent.append((location, e_id))
        return loc_ent

    def getTimesLocations(self, question):
        loc_time = []
        question_text = question['question']
        if len(question['time']) != 0:
            time = question['time'][0]
            ts2id = self.all_dicts['ts2id']
            if time in ts2id.keys():
                t_ids = [ts2id[time] + len(self.all_dicts['ent2id'])]
            else:
                keys = [x for x in ts2id.keys() if x.startswith(time)]
                t_ids = [ts2id[key] + len(self.all_dicts['ent2id']) for key in keys]
            location = question_text.find(time)
            for t_id in t_ids:
                loc_time.append((location, t_id))
        return loc_time

    def isTimeString(self, s):
        # todo: cant do len == 4 since 3 digit times also there
        if '20' in s or s.isdigit():
            return True
        else:
            return False


    def textToEntTimeId(self, text):
        if self.isTimeString(text):
            t = int(text)
            ts2id = self.all_dicts['ts2id']
            t_id = ts2id[t] + len(self.all_dicts['ent2id'])
            return t_id
        else:
            ent2id = self.all_dicts['ent2id']
            e_id = ent2id[text]
            return e_id

    def getOrderedEntityTimeIds(self, question):
        loc_ent = self.getEntitiesLocations(question)
        loc_time = self.getTimesLocations(question)
        loc_all = loc_ent + loc_time
        loc_all.sort()
        ordered_ent_time = [x[1] for x in loc_all]
        return ordered_ent_time

    def entitiesToIds(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            output.append(ent2id[e.replace('_',' ')])
        return output

    def getIdType(self, id):
        if id < len(self.all_dicts['ent2id']):
            return 'entity'
        else:
            return 'time'

    def getEntityIdToText(self, id):
        ent = self.all_dicts['id2ent'][id]
        return ent

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        # hard write
        if times[0] in ts2id.keys():
            return [ts2id[times[0]]]
        for t in times:
            keys = [x for x in ts2id.keys() if x.startswith(t)]
            output = [ts2id[key] for key in keys]
        return output

    def getAnswersFromScores(self, scores, largest=True, k=10):
        _, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToText(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time)
        return answers

    def getAnswersFromScoresWithScores(self, scores, largest=True, k=10):
        s, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToText(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time)
        return s, answers

    # from pytorch Transformer:
    # If a BoolTensor is provided, the positions with the value of True will be ignored
    # while the position with the value of False will be unchanged.
    #
    # so we want to pad with True
    def padding_tensor(self, sequences, max_len=-1):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        if max_len == -1:
            max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        # mask = sequences[0].data.new(*out_dims).fill_(0)
        mask = torch.ones((num, max_len), dtype=torch.bool)  # fills with True
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = False  # fills good area with False
        return out_tensor, mask

    def toOneHot(self, indices, vec_len):
        indices = torch.LongTensor(indices)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def prepare_data(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter
        question_text = []
        entity_time_ids = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        for question in data:
            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['question']
            question_text.append(q_text)
            et_id = self.getOrderedEntityTimeIds(question)
            entity_time_ids.append(torch.tensor(et_id, dtype=torch.long))
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)
        # answers_arr = self.get_stacked_answers_long(answers_arr)
        return {'question_text': question_text,
                'entity_time_ids': entity_time_ids,
                'answers_arr': answers_arr}

    def is_template_keyword(self, word):
        if '{' in word and '}' in word:
            return True
        else:
            return False

    def get_keyword_dict(self, template, nl_question):
        template_tokenized = self.tokenize_template(template)
        keywords = []
        for word in template_tokenized:
            if not self.is_template_keyword(word):
                # replace only first occurence
                nl_question = nl_question.replace(word, '*', 1)
            else:
                keywords.append(word[1:-1])  # no brackets
        text_for_keywords = []
        for word in nl_question.split('*'):
            if word != '':
                text_for_keywords.append(word)
        keyword_dict = {}
        for keyword, text in zip(keywords, text_for_keywords):
            keyword_dict[keyword] = text
        return keyword_dict

    def addEntityAnnotation(self, data):
        for i in range(len(data)):
            question = data[i]
            keyword_dicts = []  # we want for each paraphrase
            template = question['template']
            # for nl_question in question['paraphrases']:
            nl_question = question['question']
            keyword_dict = self.get_keyword_dict(template, nl_question)
            keyword_dicts.append(keyword_dict)
            data[i]['keyword_dicts'] = keyword_dicts
            # print(keyword_dicts)
            # print(template, nl_question)
        return data

    def tokenize_template(self, template):
        output = []
        buffer = ''
        i = 0
        while i < len(template):
            c = template[i]
            if c == '{':
                if buffer != '':
                    output.append(buffer)
                    buffer = ''
                while template[i] != '}':
                    buffer += template[i]
                    i += 1
                buffer += template[i]
                output.append(buffer)
                buffer = ''
            else:
                buffer += c
            i += 1
        if buffer != '':
            output.append(buffer)
        return output


class QA_Dataset_MultiQA_Advanced(QA_Dataset_multi):
    def __init__(self, split, dataset_name='MultiTQ', tokenization_needed=True,
                 replace_entity_spans_with_mask=True, args=None):
        super().__init__(split, dataset_name, tokenization_needed, args)
        print('Preparing advanced data for split %s' % split)

        self.replace_entity_spans_with_mask = replace_entity_spans_with_mask

        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times   = len(self.all_dicts['ts2id'])
        self.padding_idx       = self.num_total_entities + self.num_total_times
        self.answer_vec_size   = self.num_total_entities + self.num_total_times

        self.tokenizer1 = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        # ── Build KG index (stores ONLY timestamp pairs, not full triples) ──
        kg_dir = f"{data_dir}/data/{dataset_name}/kg"
        self.e2rt, self.e2tr = self._build_kg_index(kg_dir)

        self.corrupt_p = getattr(args, 'corrupt_hard', 0.0) if args is not None else 0.0

        # NO second pass! Fact retrieval is inlined into prepare_data_advanced
        self.prepared_data = self.prepare_data_advanced(self.data)

    # ------------------------------------------------------------------
    # KG indexing: store only (ts, ts) — cuts memory and speeds up sets
    # ------------------------------------------------------------------
    def _build_kg_index(self, kg_dir):
        with open(os.path.join(kg_dir, "entity2id.json"), "r") as f:
            ent2id = json.load(f)
        with open(os.path.join(kg_dir, "relation2id.json"), "r") as f:
            rel2id = json.load(f)

        e2rt = defaultdict(set)
        e2tr = defaultdict(set)

        for split in ["train.txt", "valid.txt", "test.txt", "full.txt"]:
            path = os.path.join(kg_dir, split)
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 4:
                        parts = line.split()
                    if len(parts) != 4:
                        continue
                    h, r, t, ts = parts
                    if h not in ent2id or t not in ent2id or r not in rel2id:
                        continue
                    h_id, t_id = ent2id[h], ent2id[t]
                    ts_pair = (ts, ts)
                    e2rt[(h_id, t_id)].add(ts_pair)
                    e2rt[(t_id, h_id)].add(ts_pair)
                    e2tr[h_id].add(ts_pair)
                    e2tr[t_id].add(ts_pair)
        return e2rt, e2tr

    # ------------------------------------------------------------------
    # Fast fact lookup (replaces _get_facts_for_question)
    # ------------------------------------------------------------------
    def _get_facts_from_entities(self, entities):
        """Return set of (ts_start, ts_end) pairs for given entity IDs."""
        if len(entities) >= 3:
            e1, e2, e3 = entities[0], entities[1], entities[2]
            facts = set()
            facts.update(self.e2rt.get((e1, e2), ()))
            facts.update(self.e2rt.get((e1, e3), ()))
            facts.update(self.e2rt.get((e2, e3), ()))
            return facts
        elif len(entities) == 2:
            return self.e2rt.get((entities[0], entities[1]), set())
        elif len(entities) == 1:
            return self.e2tr.get(entities[0], set())
        return set()

    # ------------------------------------------------------------------
    # Tokenizer helper (unchanged)
    # ------------------------------------------------------------------
    def tokenize(self, words):
        tokens = []
        valid_positions = []
        tokens.append(self.tokenizer1.cls_token)
        valid_positions.append(0)
        for word in words:
            token = self.tokenizer1.tokenize(word)
            tokens.extend(token)
            for _ in range(len(token)):
                valid_positions.append(1 if _ == 0 else 0)
        tokens.append(self.tokenizer1.sep_token)
        valid_positions.append(0)
        return tokens, valid_positions

    # ------------------------------------------------------------------
    # Entity/time span extraction (unchanged)
    # ------------------------------------------------------------------
    def get_entity_time_tuples(self, question):
        q_text = question["question"]
        num_ent = self.num_total_entities
        ent2id = self.all_dicts["ent2id"]

        tuples = []
        accepted_spans = []

        for ep in question.get("entity_positions", []):
            if not ep.get("entity"):
                continue
            wd_id = ep["entity"][0]
            if wd_id not in ent2id:
                continue
            start, end = ep["position"]
            e_id = ent2id[wd_id]
            tuples.append((q_text[start:end], e_id, start, end))
            accepted_spans.append((start, end))

        for t_str in question.get("time", []):
            if not t_str:
                continue
            search_from = 0
            while True:
                idx = q_text.find(t_str, search_from)
                if idx == -1:
                    break
                end = idx + len(t_str)
                overlap = any(s < end and idx < e for s, e in accepted_spans)
                if not overlap:
                    t_ids = self.timesToIds([t_str])
                    if len(t_ids) > 0:
                        t_id = t_ids[0] + num_ent
                        tuples.append((t_str, t_id, idx, end))
                        accepted_spans.append((idx, end))
                    break
                search_from = idx + 1

        tuples.sort(key=lambda x: x[2])
        return tuples

    def get_entity_aware_tokenization(self, nl_question, ent_time_tuples):
        if not self.replace_entity_spans_with_mask:
            enc = self.tokenizer1(
                nl_question,
                add_special_tokens=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                truncation=False,
            )
            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            tokenized = self.tokenizer1.convert_ids_to_tokens(input_ids)
            entity_time_ids = [self.padding_idx] * len(tokenized)
            entity_mask = [1.0] * len(tokenized)

            for _, e_id, start_char, end_char in ent_time_tuples:
                for i, (s, e) in enumerate(offsets):
                    if s == 0 and e == 0:
                        continue
                    if s < end_char and e > start_char:
                        entity_time_ids[i] = e_id
                        entity_mask[i] = 0.0
            return tokenized, entity_time_ids, entity_mask

        pieces, arr = [], []
        start_index = 0
        for _, e_id, start_char, end_char in ent_time_tuples:
            if start_index < start_char:
                chunk = nl_question[start_index:start_char]
                if chunk != "":
                    pieces.append(chunk)
                    arr.append(self.padding_idx)
            pieces.append(self.tokenizer1.mask_token)
            arr.append(e_id)
            start_index = end_char
        if start_index < len(nl_question):
            chunk = nl_question[start_index:]
            if chunk != "":
                pieces.append(chunk)
                arr.append(self.padding_idx)

        tokenized, valid_ids = self.tokenize(pieces)
        entity_time_final = []
        idx = 0
        for vid in valid_ids:
            if vid == 0:
                entity_time_final.append(self.padding_idx)
            else:
                entity_time_final.append(arr[idx])
                idx += 1
        entity_mask = [1.0 if x == self.padding_idx else 0.0 for x in entity_time_final]
        return tokenized, entity_time_final, entity_mask

    # ------------------------------------------------------------------
    # Fallback joint bounds (updated for (ts, ts) format)
    # ------------------------------------------------------------------
    def _parse_ts(self, d):
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(d, fmt)
            except ValueError:
                continue
        return None

    def get_joint_time_bounds(self, question):
        ent2id = self.all_dicts["ent2id"]
        entity_positions = sorted(
            question.get("entity_positions", []),
            key=lambda x: x["position"][0]
        )
        if len(entity_positions) < 2:
            return 0, 0
        e1_text = entity_positions[0]["entity"][0]
        e2_text = entity_positions[1]["entity"][0]
        if e1_text not in ent2id or e2_text not in ent2id:
            return 0, 0
        e1, e2 = ent2id[e1_text], ent2id[e2_text]
        related = self.e2rt.get((e1, e2), set()).union(self.e2rt.get((e2, e1), set()))
        if len(related) == 0:
            return 0, 0
        timestamps = []
        for start_ts, end_ts in related:
            timestamps.append(start_ts)
            timestamps.append(end_ts)
        timestamps = list(set(timestamps))
        parsed = [(d, self._parse_ts(d)) for d in timestamps]
        parsed.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else datetime.min))
        return parsed[0][0], parsed[-1][0]

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        for t in times:
            t = str(t)
            if t in ts2id:
                output.append(ts2id[t])
            else:
                keys = [x for x in ts2id.keys() if str(x).startswith(t)]
                output.extend([ts2id[key] for key in keys])
        return output

    # ------------------------------------------------------------------
    # prepare_data_advanced — INLINED fact retrieval, no extra data pass
    # ------------------------------------------------------------------
    def prepare_data_advanced(self, data):
        heads, tails, times, start_times, end_times = [], [], [], [], []
        tails2, types, rels = [], [], []
        question_text, tokenized_question = [], []
        entity_time_ids_tok, entity_mask_tok = [], []
        answers_arr = []
        self.data_ids_filtered = []
        num_total_entities = self.num_total_entities
        corrupt_p = self.corrupt_p

        for i, question in enumerate(tqdm(data)):
            self.data_ids_filtered.append(i)

            q_text = question["question"]
            question_text.append(q_text)

            # ---- head / tail / tail2 ----
            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [eid for _, eid in entities_list_with_locations]

            if len(entities) == 0:
                head = tail = tail2 = 0
            elif len(entities) == 1:
                head = tail = tail2 = entities[0]
            elif len(entities) == 2:
                head, tail, tail2 = entities[0], entities[1], entities[1]
            else:
                head, tail, tail2 = entities[0], entities[1], entities[2]

            # ---- time / start_time / end_time (INLINE, no second loop) ----
            times_in_question = question.get("time", [])

            if len(times_in_question) > 0:
                raw_time_ids = self.timesToIds(times_in_question)
                if len(raw_time_ids) > 0:
                    raw_time_id = raw_time_ids[0]
                    time = raw_time_id + num_total_entities
                    start_time = raw_time_id
                    end_time = raw_time_id
                else:
                    time = num_total_entities
                    start_time = 0
                    end_time = 0
            else:
                time = num_total_entities

                # FAST: retrieve facts directly from the entity IDs we already have
                facts = self._get_facts_from_entities(entities)

                if len(facts) > 0 and corrupt_p > 0:
                    facts = [f for f in facts if random.random() >= corrupt_p]

                if len(facts) > 0:
                    ts = []
                    for start_ts, end_ts in facts:
                        if start_ts == end_ts:
                            ts.append(start_ts)
                        else:
                            try:
                                for t in range(int(start_ts), int(end_ts) + 1):
                                    ts.append(str(t))
                            except ValueError:
                                ts.append(start_ts)
                                ts.append(end_ts)

                    ts = sorted(set(ts))
                    sorted_times = self.timesToIds(ts)
                    try:
                        start_time = sorted_times[0]
                    except (IndexError, TypeError):
                        start_time = 0
                    try:
                        end_time = sorted_times[-1]
                    except (IndexError, TypeError):
                        end_time = 0
                else:
                    # Fallback: direct triples between first two entities
                    start_ts, end_ts = self.get_joint_time_bounds(question)
                    if start_ts:
                        s_ids = self.timesToIds([start_ts])
                        start_time = s_ids[0] if len(s_ids) > 0 else 0
                    else:
                        start_time = 0
                    if end_ts:
                        e_ids = self.timesToIds([end_ts])
                        end_time = e_ids[0] if len(e_ids) > 0 else 0
                    else:
                        end_time = 0

            heads.append(int(head))
            tails.append(int(tail))
            tails2.append(int(tail2))
            times.append(int(time))
            start_times.append(int(start_time))
            end_times.append(int(end_time))

            # ---- type / rel ----
            q_type = question.get("type", question.get("qtype", ""))
            types.append(q_type)
            rel_list = list(question.get("relations", []))
            rel = self.all_dicts["rel2id"][rel_list[0]] if len(rel_list) > 0 else 0
            rels.append(int(rel))

            # ---- entity/time-aware tokenization ----
            ent_time_tuples = self.get_entity_time_tuples(question)
            if len(ent_time_tuples) > 0:
                tok, et_final, e_mask = self.get_entity_aware_tokenization(q_text, ent_time_tuples)
            else:
                tok, _ = self.tokenize([q_text])
                et_final = [self.padding_idx] * len(tok)
                e_mask = [1.0] * len(tok)

            tokenized_question.append(self.tokenizer.convert_tokens_to_ids(tok))
            entity_time_ids_tok.append(et_final)
            entity_mask_tok.append(e_mask)

            # ---- answers ----
            if question["answer_type"] == "entity":
                answers = self.entitiesToIds(question["answers"])
            else:
                answers = [x + num_total_entities for x in self.timesToIds(question["answers"])]
            answers_arr.append(answers)

        return {
            "question_text": question_text,
            "tokenized_question": tokenized_question,
            "entity_time_ids": entity_time_ids_tok,
            "entity_mask": entity_mask_tok,
            "head": heads,
            "tail": tails,
            "time": times,
            "start_time": start_times,
            "end_time": end_times,
            "tail2": tails2,
            "types": types,
            "rels": rels,
            "answers_arr": answers_arr,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.prepared_data
        return (
            d["question_text"][index],
            d["tokenized_question"][index],
            np.array(d["entity_time_ids"][index], dtype=np.int64),
            d["entity_mask"][index],
            d["head"][index],
            d["tail"][index],
            d["time"][index],
            d["start_time"][index],
            d["end_time"][index],
            d["tail2"][index],
            d["types"][index],
            d["rels"][index],
            random.choice(d["answers_arr"][index]),
        )

    def _collate_fn(self, items):
        tokenized_questions = [item[1] for item in items]
        attention_mask = torch.from_numpy(self.get_attention_mask(tokenized_questions))
        input_ids = torch.from_numpy(
            self.pad_for_batch(tokenized_questions, self.tokenizer.pad_token_id, np.int64)
        )

        entity_time_ids_list = [item[2] for item in items]
        entity_time_ids_padded = torch.from_numpy(
            self.pad_for_batch(entity_time_ids_list, self.padding_idx, np.int64)
        )

        entity_mask_list = [item[3] for item in items]
        entity_mask_padded = torch.from_numpy(
            self.pad_for_batch(entity_mask_list, 1.0, np.float32)
        )

        heads = torch.from_numpy(np.array([item[4] for item in items]))
        tails = torch.from_numpy(np.array([item[5] for item in items]))
        times = torch.from_numpy(np.array([item[6] for item in items]))
        start_times = torch.from_numpy(np.array([item[7] for item in items]))
        end_times = torch.from_numpy(np.array([item[8] for item in items]))
        tails2 = torch.from_numpy(np.array([item[9] for item in items]))
        types = [item[10] for item in items]
        rels = torch.from_numpy(np.array([item[11] for item in items]))
        answers_single = torch.from_numpy(np.array([item[12] for item in items]))

        return (
            input_ids, attention_mask,
            entity_time_ids_padded, entity_mask_padded,
            heads, tails, times, start_times, end_times,
            tails2, types, rels, answers_single,
        )

    def pad_for_batch(self, to_pad, padding_val, dtype=np.int64):
        max_len = len(max(to_pad, key=lambda x: len(x)))
        padded = np.ones([len(to_pad), max_len], dtype=dtype) * padding_val
        for i, j in enumerate(to_pad):
            padded[i][0:len(j)] = j
        return padded

    def get_attention_mask(self, tokenized):
        max_len = len(max(tokenized, key=lambda x: len(x)))
        mask = np.zeros([len(tokenized), max_len], dtype=np.int64)
        for i, j in enumerate(tokenized):
            mask[i][0:len(j)] = np.ones(len(j), dtype=np.int64)
        return mask

    def get_dataset_ques_info(self):
        type2num = {}
        for question in self.data:
            if question["qtype"] not in type2num:
                type2num[question["qtype"]] = 0
            type2num[question["qtype"]] += 1
        return {"type2num": type2num, "total_num": len(self.data_ids_filtered)}.__str__()


class QA_Dataset_Baseline_muti(QA_Dataset_multi):
    def __init__(self, split, dataset_name='MultiTQ', tokenization_needed=True,args = None):
        super().__init__(split, dataset_name, tokenization_needed,args)
        print('Preparing data for split %s' % split)

        ents = self.all_dicts['ent2id'].keys()
        self.all_dicts['tsstr2id'] = self.all_dicts['id2ts']
        times = self.all_dicts['tsstr2id'].keys()
        rels = self.all_dicts['rel2id'].keys()

        self.prepared_data = self.prepare_data_(self.data)
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.answer_vec_size = self.num_total_entities + self.num_total_times

    def prepare_data_(self, data):
        # we want to prepare answers lists for each question
        # then at batch prep time, we just stack these
        # and use scatter
        question_text = []
        heads = []
        tails = []
        times = []
        num_total_entities = len(self.all_dicts['ent2id'])
        answers_arr = []
        ent2id = self.all_dicts['ent2id']
        self.data_ids_filtered = []
        # self.data=[]
        for i, question in enumerate(data):
            self.data_ids_filtered.append(i)

            # first pp is question text
            # needs to be changed after making PD dataset
            # to randomly sample from list
            q_text = question['question']

            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in
                        entities_list_with_locations]  # ordering necessary otherwise set->list conversion causes randomness
            if len(entities) == 0:
                head = 0
                tail = 0
            else:
                head = entities[0]  # take an entity
                if len(entities) > 1:
                    tail = entities[1]
                else:
                    tail = entities[0]
            times_in_question = question['time']
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)  # take a time. if no time then 0
                # exit(0)
            else:
                time = [0]

            time = [int(x) + num_total_entities for x in time]
            heads.append(int(head))
            times.append(time)
            tails.append(int(tail))
            question_text.append(q_text)

            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [int(x) + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)

        # answers_arr = self.get_stacked_answers_long(answers_arr)
        self.data = [self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text,
                'head': heads,
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr}

    def print_prepared_data(self):
        for k, v in self.prepared_data.items():
            print(k, v)

    def __len__(self):
        return len(self.data)
        # return len(self.prepared_data['question_text'])

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, answers_single  # ,answers_khot

    def _collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        # times = [item[3] for item in items]

        # times = torch.from_numpy(np.array([item[3][0] for item in items]))
        times = torch.from_numpy(np.array([np.random.choice(item[3]) for item in items]))
        answers_single = torch.from_numpy(np.array([item[4] for item in items]))
        return b['input_ids'], b['attention_mask'], heads, tails, times, answers_single

    def get_dataset_ques_info(self):
        type2num = {}
        for question in self.data:
            if question["type"] not in type2num: type2num[question["type"]] = 0
            type2num[question["type"]] += 1
        return {"type2num": type2num, "total_num": len(self.data_ids_filtered)}.__str__()


# class QA_Dataset_MultiQA_muti(QA_Dataset_multi):
#     def __init__(self, split, dataset_name='MultiTQ', tokenization_needed=True,args = None):
#         super().__init__(split, dataset_name, tokenization_needed, args)
#         print('Preparing data for split %s' % split)
#         ents = self.all_dicts['ent2id'].keys()
#         self.all_dicts['tsstr2id'] = self.all_dicts['id2ts']
#         times = self.all_dicts['tsstr2id'].keys()
#         rels = self.all_dicts['rel2id'].keys()

#         self.prepared_data = self.prepare_data_(self.data)
#         self.num_total_entities = len(self.all_dicts['ent2id'])
#         self.num_total_times = len(self.all_dicts['ts2id'])
#         self.answer_vec_size = self.num_total_entities + self.num_total_times

#     def prepare_data_(self, data):
#         # we want to prepare answers lists for each question
#         # then at batch prep time, we just stack these
#         # and use scatter
#         question_text = []
#         heads = []
#         tails = []
#         times = []
#         num_total_entities = len(self.all_dicts['ent2id'])
#         answers_arr = []
#         ent2id = self.all_dicts['ent2id']
#         self.data_ids_filtered = []
#         # self.data=[]
#         for i, question in enumerate(data):
#             self.data_ids_filtered.append(i)
#             q_text = question['question']
#             entities_list_with_locations = self.getEntitiesLocations(question)
#             entities_list_with_locations.sort()
#             entities = [id for location, id in
#                         entities_list_with_locations]  # ordering necessary otherwise set->list conversion causes randomness
#             if len(entities) == 0:
#                 head = 0
#                 tail = 0
#             else:
#                 head = entities[0]  # take an entity
#                 if len(entities) > 1:
#                     tail = entities[1]
#                 else:
#                     tail = entities[0]
#             times_in_question = question['time']
#             if len(times_in_question) > 0:
#                 time = self.timesToIds(times_in_question)  # take a time. if no time then 0
#                 # exit(0)
#             else:
#                 time = [0]

#             time = [x + num_total_entities for x in time]
#             heads.append(head)
#             times.append(time)
#             tails.append(tail)
#             question_text.append(q_text)

#             if question['answer_type'] == 'entity':
#                 answers = self.entitiesToIds(question['answers'])
#             else:
#                 # adding num_total_entities to each time id
#                 answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
#             answers_arr.append(answers)

#         # answers_arr = self.get_stacked_answers_long(answers_arr)
#         self.data = [self.data[idx] for idx in self.data_ids_filtered]
#         return {'question_text': question_text,
#                 'head': heads,
#                 'tail': tails,
#                 'time': times,
#                 'answers_arr': answers_arr}

#     def print_prepared_data(self):
#         for k, v in self.prepared_data.items():
#             print(k, v)

#     def __len__(self):
#         return len(self.data)
#         # return len(self.prepared_data['question_text'])

#     def __getitem__(self, index):
#         data = self.prepared_data
#         question_text = data['question_text'][index]
#         head = data['head'][index]
#         tail = data['tail'][index]
#         time = data['time'][index]
#         answers_arr = data['answers_arr'][index]
#         answers_single = random.choice(answers_arr)
#         return question_text, head, tail, time, answers_single  # ,answers_khot

#     def _collate_fn(self, items):
#         batch_sentences = [item[0] for item in items]
#         b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
#         heads = torch.from_numpy(np.array([item[1] for item in items]))
#         tails = torch.from_numpy(np.array([item[2] for item in items]))
#         times = [torch.tensor(item[3]) for item in items]
#         answers_single = torch.from_numpy(np.array([item[4] for item in items]))
#         return b['input_ids'], b['attention_mask'], heads, tails, times, answers_single

#     def get_dataset_ques_info(self):
#         type2num = {}
#         for question in self.data:
#             if question["type"] not in type2num: type2num[question["type"]] = 0
#             type2num[question["type"]] += 1
#         return {"type2num": type2num, "total_num": len(self.data_ids_filtered)}.__str__()




def main():
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='timequestions')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, default='subgtr',
                        choices=['baseline', 'subgtr', 'multiqa_advanced'])
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}, split: {args.split}, mode: {args.mode}")

    if args.mode == 'baseline':
        dataset = QA_Dataset_Baseline(args.split, args.dataset)
    elif args.mode == 'subgtr':
        # minimal args mock for SubGTR
        class DummyArgs:
            aware_module = False
            tkg_file = 'full.txt'
            dataset_name = args.dataset
            corrupt_hard = False
            fuse = False
            
        dataset = QA_Dataset_TimeQuestions(args.split, args.dataset, DummyArgs(), replace_entity_spans_with_mask=False)
    else:  # multiqa_advanced
        dataset = QA_Dataset_MultiQA_Advanced(args.split, args.dataset, replace_entity_spans_with_mask=False)

    print("Dataset size:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset._collate_fn
    )

    print("\n--- Inspecting a few batches ---\n")

    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}")

        if args.mode == 'baseline':
            input_ids, attention_mask, heads, tails, times, answers = batch
            print("input_ids shape:", input_ids.shape)
            print("attention_mask shape:", attention_mask.shape)
            print("heads:", heads)
            print("tails:", tails)
            print("times:", times)
            print("answers:", answers)

        elif args.mode == 'subgtr':
            (input_ids, attention_mask, entity_time_ids, entity_mask,
             heads, tails, times, start_times, end_times,
             tails2, types, rels, answers) = batch

            # print("input_ids shape:", input_ids.shape)
            # print("attention_mask shape:", attention_mask.shape)
            # print("entity_time_ids shape:", entity_time_ids.shape)
            # print("entity_mask shape:", entity_mask.shape)
            print("heads:", heads)
            print("tails:", tails)
            print("times:", times)
            print("answers:", answers)

            toks = input_ids[0].tolist()
            et   = entity_time_ids[0].tolist()
            em   = entity_mask[0].tolist()
            # decode first 20 tokens for readability
            print("Tokens (first 20):", dataset.tokenizer.convert_ids_to_tokens(toks[:20]))
            # print("Ent/Time IDs   :", et[:20])
            # print("Entity mask    :", [f"{x:.0f}" for x in em[:20]])
            print("Start time:", start_times, "End time:", end_times)

        else:  # multiqa_advanced
            (input_ids, attention_mask, entity_time_ids, entity_mask,
             heads, tails, times, start_times, end_times,
             tails2, types, rels, answers) = batch

            # print("input_ids shape:", input_ids.shape)
            # print("attention_mask shape:", attention_mask.shape)
            # print("entity_time_ids shape:", entity_time_ids.shape)
            # print("entity_mask shape:", entity_mask.shape)
            print("heads:", heads)
            print("tails:", tails)
            print("times:", times)
            print("answers:", answers)

            toks = input_ids[0].tolist()
            et   = entity_time_ids[0].tolist()
            em   = entity_mask[0].tolist()
            # decode first 20 tokens for readability
            print("Tokens (first 20):", dataset.tokenizer.convert_ids_to_tokens(toks[:20]))
            # print("Ent/Time IDs   :", et[:20])
            # print("Entity mask    :", [f"{x:.0f}" for x in em[:20]])
            print("Start time:", start_times, "End time:", end_times)

        if i == 8:  # only show first 3 batches
            break





if __name__ == "__main__":
    main()