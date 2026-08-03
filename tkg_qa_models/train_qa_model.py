import os
import argparse
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tkg_qa_models.qa_baselines import QA_lm, QA_embedkgqa, QA_cronkgqa, QA_TempoQR, QA_Sabet, QA_SubGTR
from tkg_qa_models.qa_datasets import QA_Dataset_SubGTR, QA_Dataset_Baseline, QA_Dataset_Baseline_muti, QA_Dataset_MultiQA_Advanced
from torch.utils.data import DataLoader
from tkg_qa_models import utils
from tkg_qa_models.utils import loadTkbcModel, print_info
from collections import defaultdict
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Temporal KGQA")
parser.add_argument('--tkbc_model_file',  default='tcomplex.ckpt', type=str)
parser.add_argument('--tkg_file',         default='full.txt',      type=str)
parser.add_argument('--model',            default='sabet',        type=str)
parser.add_argument('--time_sensitivity',   action='store_true')
parser.add_argument('--aware_module',       action='store_true')
parser.add_argument('--dataset_name',     default='wikidata_big_complex',  type=str)
parser.add_argument('--supervision',      default='soft',          type=str)
parser.add_argument('--load_from',        default='',              type=str)
parser.add_argument('--save_to',          default='',              type=str)
parser.add_argument('--max_epochs',       default=20,  type=int)
parser.add_argument('--eval_k',           default=1,   type=int)
parser.add_argument('--valid_freq',       default=1,   type=int)
parser.add_argument('--batch_size',       default=150, type=int)
parser.add_argument('--valid_batch_size', default=150,  type=int)
parser.add_argument('--frozen',           default=1,   type=int)
parser.add_argument('--lm_frozen',        default=1,   type=int)
parser.add_argument('--lr',               default=2e-4, type=float)
parser.add_argument('--mode',             default='train', type=str)
parser.add_argument('--eval_split',       default='valid', type=str)
parser.add_argument('--lm',              default='distill_bert', type=str)
parser.add_argument('--fuse',            default='add',          type=str)
parser.add_argument('--extra_entities',  default=False, type=bool)
parser.add_argument('--corrupt_hard',    default=0.,   type=float)
parser.add_argument('--test',            default='test', type=str)
parser.add_argument('--temperature',     default=1.0,   type=str)

args = parser.parse_args()
print_info(args)

data_dir = '/Data/data'

# ── Evaluation ────────────────────────────────────────────────────────────────
def eval(qa_model, dataset, batch_size=128, split='valid', k=200, hop_weights_file=None):
    num_workers   = 4
    k_for_report  = k
    k_list        = [1, 10]
    max_k         = 100

    qa_model.eval()
    eval_log = [f"Split {split}"]
    print('Evaluating split', split)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=dataset._collate_fn,
    )

    topk_answers = []
    total_loss   = 0
    total_batches = len(data_loader)

    hop_weights_all = []
    question_types_all = []
    answer_types_all = []

    example_idx = 0

    with torch.no_grad():
        for i_batch, a in enumerate(data_loader):
            if i_batch * batch_size == len(dataset.data):
                break

            answers_khot = a[-1]
            out = qa_model.forward(a, mode="eval")

            if isinstance(out, tuple) and len(out) == 2:
                scores, hop_weights = out
                hop_weights_all.append(hop_weights.cpu())
            else:
                scores = out

            batch_size_actual = scores.shape[0]
            batch_questions = dataset.data[example_idx:example_idx + batch_size_actual]

            for q in batch_questions:
                question_type = q['qtype'] if args.dataset_name == 'MultiTQ' else q['type']
                if isinstance(question_type, list):
                    question_type = question_type[0]
                question_types_all.append(question_type)
                answer_types_all.append(q['answer_type'])

            example_idx += batch_size_actual

            for s in scores:
                topk_answers.append(dataset.getAnswersFromScores(s, k=max_k))

            total_loss += qa_model.loss(scores, answers_khot.cuda()).item()

            if (i_batch + 1) % 100 == 0 or (i_batch + 1) == total_batches:
                print(f"Step {i_batch+1}/{total_batches}", flush=True)

    if hop_weights_file is not None and len(hop_weights_all) > 0:
        hop_weights_all = torch.cat(hop_weights_all, dim=0)  # [N, num_hops]
        hop_payload = {
            "hop_weights": hop_weights_all,
            "question_type": question_types_all,
            "answer_type": answer_types_all,
        }
        os.makedirs(os.path.dirname(hop_weights_file), exist_ok=True)
        torch.save(hop_payload, hop_weights_file)
        print(f"Saved hop weights + metadata to {hop_weights_file}")

    eval_log.append(f"Loss {total_loss:.6f}")
    eval_log.append(f"Eval batch size {batch_size}")

    eval_accuracy_for_report = 0
    for k in k_list:
        hits_at_k = 0
        total = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count = defaultdict(list)

        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            question_type = question['qtype'] if args.dataset_name == 'MultiTQ' else question['type']

            if isinstance(question_type, list):
                question_type = question_type[0]

            simple_complex_type = 'simple' if 'simple' in question_type else 'complex'
            entity_time_type = question['answer_type']
            predicted = topk_answers[i]

            predicted = predicted[:k]
            hit = 1 if set(actual_answers) & set(predicted) else 0
            hits_at_k += hit
            total += 1
            question_types_count[question_type].append(hit)
            simple_complex_count[simple_complex_type].append(hit)
            entity_time_count[entity_time_type].append(hit)

        eval_accuracy = hits_at_k / total
        if k == k_for_report:
            eval_accuracy_for_report = eval_accuracy
        eval_log.append(f"Hits at {k}: {round(eval_accuracy, 3)}")

        for dictionary in [
            dict(sorted(question_types_count.items(), key=lambda x: x[0].lower())),
            dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower())),
            dict(sorted(entity_time_count.items(), key=lambda x: x[0].lower())),
        ]:
            for key, value in dictionary.items():
                h = sum(value) / len(value)
                eval_log.append(
                    f"{key} \t {round(h, 3)} \t total questions: {len(value)}"
                )
            eval_log.append('')

    for s in eval_log:
        print(s)
    return eval_accuracy_for_report, eval_log

def eval1(qa_model, dataset, batch_size=128, split='valid', k=200, hop_weights_file=None):
    num_workers   = 4
    k_for_report  = k
    k_list        = [1, 10]
    max_k         = 100

    qa_model.eval()
    eval_log = [f"Split {split}"]
    print('Evaluating split', split)

    data_loader   = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=dataset._collate_fn,
    )
    topk_answers  = []
    total_loss    = 0
    total_batches = len(data_loader)

    hop_weights_all = []

    with torch.no_grad():
        for i_batch, a in enumerate(data_loader):
            if i_batch * batch_size == len(dataset.data):
                break
            answers_khot = a[-1]
            out       = qa_model.forward(a, mode="eval")

            if isinstance(out, tuple) and len(out) == 2:
                scores, hop_weights = out
                hop_weights_all.append(hop_weights.cpu())
            else:
                scores = out

            for s in scores:
                topk_answers.append(dataset.getAnswersFromScores(s, k=max_k))
            total_loss += qa_model.loss(scores, answers_khot.cuda()).item()
            if (i_batch + 1) % 100 == 0 or (i_batch + 1) == total_batches:
                print(f"Step {i_batch+1}/{total_batches}", flush=True)

    if hop_weights_file is not None and len(hop_weights_all) > 0:
        hop_weights_all = torch.cat(hop_weights_all, dim=0)   # [N, num_hops]
        os.makedirs(os.path.dirname(hop_weights_file), exist_ok=True)
        torch.save(hop_weights_all, hop_weights_file)
        print(f"Saved hop weights to {hop_weights_file}")
            
    eval_log.append(f"Loss {total_loss:.6f}")
    eval_log.append(f"Eval batch size {batch_size}")

    eval_accuracy_for_report = 0
    for k in k_list:
        hits_at_k            = 0
        total                = 0
        question_types_count = defaultdict(list)
        simple_complex_count = defaultdict(list)
        entity_time_count    = defaultdict(list)

        for i, question in enumerate(dataset.data):
            actual_answers      = question['answers']
            question_type = question['qtype'] if args.dataset_name == 'MultiTQ' else question['type']

            if isinstance(question_type, list):
                question_type = question_type[0]

            simple_complex_type = 'simple' if 'simple' in question_type else 'complex'
            entity_time_type    = question['answer_type']
            predicted           = topk_answers[i]

            predicted     = predicted[:k]
            hit           = 1 if set(actual_answers) & set(predicted) else 0
            hits_at_k    += hit
            total        += 1
            question_types_count[question_type].append(hit)
            simple_complex_count[simple_complex_type].append(hit)
            entity_time_count[entity_time_type].append(hit)

        eval_accuracy = hits_at_k / total
        if k == k_for_report:
            eval_accuracy_for_report = eval_accuracy
        eval_log.append(f"Hits at {k}: {round(eval_accuracy, 3)}")

        for dictionary in [
            dict(sorted(question_types_count.items(), key=lambda x: x[0].lower())),
            dict(sorted(simple_complex_count.items(), key=lambda x: x[0].lower())),
            dict(sorted(entity_time_count.items(),    key=lambda x: x[0].lower())),
        ]:
            for key, value in dictionary.items():
                h = sum(value) / len(value)
                eval_log.append(
                    f"{key} \t {round(h, 3)} \t total questions: {len(value)}"
                )
            eval_log.append('')

    for s in eval_log:
        print(s)
    return eval_accuracy_for_report, eval_log


def append_log_to_file(eval_log, epoch, filename):
    with open(filename, 'a+') as f:
        f.write('Log time: %s\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write('Epoch %d\n' % epoch)
        for line in eval_log:
            f.write('%s\n' % line)
        f.write('\n')


# ── Training ──────────────────────────────────────────────────────────────────
def train(qa_model, dataset, valid_dataset, args, result_filename=None):
    num_workers = 5
    batch_size  = args.batch_size

    optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)

    data_loader   = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=dataset._collate_fn,
    )
    total_steps   = args.max_epochs * len(data_loader)
    warmup_steps  = min(200, total_steps // 10)

    warmup_sched  = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched  = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.lr * 0.01
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    if args.save_to == '':
        args.save_to = 'temp'
    if result_filename is None:
        result_filename = f"results/{args.dataset_name}/{args.save_to}.log"
    checkpoint_path = (
        f"{data_dir}"
        f"/qa_models/{args.dataset_name}/{args.save_to}.ckpt"
    )

    if args.load_from == '':
        print('Creating new log file')
        with open(result_filename, 'a+') as f:
            f.write('Log time: %s\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            f.write('Config:\n')
            for key, value in vars(args).items():
                f.write(f'{key}:\t{value}\n')
            f.write('\n')

    max_eval_score = 0.0
    print('Starting training')

    for epoch in range(args.max_epochs):
        qa_model.train()
        epoch_loss   = 0.0
        running_loss = 0.0
        total_batches = len(data_loader)

        for i_batch, a in enumerate(data_loader):
            qa_model.zero_grad()

            scores, loss = qa_model.forward(a, mode="train")
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss   += loss.item()
            running_loss += loss.item()

            if (i_batch + 1) % 100 == 0 or (i_batch + 1) == total_batches:
                n          = i_batch + 1
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch+1}/{args.max_epochs} "
                    f"Step {n}/{total_batches} "
                    f"LR {current_lr:.2e} "
                    f"Loss {running_loss/n:.4f}",
                    flush=True,
                )

            
        print('Epoch loss =', epoch_loss)

        if (epoch + 1) % args.valid_freq == 0:
            eval_score, eval_log = eval(
                qa_model, valid_dataset,
                batch_size=args.valid_batch_size,
                split=args.eval_split, k=args.eval_k,
            )
            if eval_score > max_eval_score:
                print('Valid score increased')
                save_model(qa_model, checkpoint_path)
                max_eval_score = eval_score
            append_log_to_file(eval_log, epoch, result_filename)


            import copy
            train_eval_subset = copy.copy(dataset)
            train_eval_subset.data = dataset.data[:1000]

            # ── Evaluate on 1000 training examples to monitor training progress / signs of overfitting ──────────────
            train_score, train_log = eval(
                qa_model, train_eval_subset,
                batch_size=args.valid_batch_size,
                split='train_1000', k=args.eval_k,
            )
            append_log_to_file(train_log, epoch, result_filename)


def save_model(qa_model, filename):
    print('Saving model to', filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(qa_model.state_dict(), filename)
    print('Saved model to', filename)


# ── Model / dataset setup ─────────────────────────────────────────────────────
tkbc_model = loadTkbcModel(
        f"{data_dir}/models/{args.dataset_name}"
        f"/kg_embeddings/{args.tkbc_model_file}"
    )

if args.mode == 'test_kge':
    utils.checkIfTkbcEmbeddingsTrained(tkbc_model, args.dataset_name, args.eval_split)
    exit(0)

train_split = 'train'
test        = 'test'
val        = 'dev' if args.dataset_name == 'MultiTQ' else 'valid'


if args.model in ('bert', 'roberta'):
    if args.dataset_name == 'MultiTQ':
        qa_model     = QA_lm(tkbc_model, args)
        dataset      = QA_Dataset_Baseline_muti(split=train_split, dataset_name=args.dataset_name)
        test_dataset = QA_Dataset_Baseline_muti(split=test,        dataset_name=args.dataset_name)
        val_dataset  = QA_Dataset_Baseline_muti(split=val,         dataset_name=args.dataset_name)
    else:
        qa_model     = QA_lm(tkbc_model, args)
        dataset      = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
        test_dataset = QA_Dataset_Baseline(split=test,        dataset_name=args.dataset_name)
        val_dataset = QA_Dataset_Baseline(split=val,        dataset_name=args.dataset_name)

elif args.model == 'embedkgqa':
    if args.dataset_name == 'MultiTQ':
        qa_model     = QA_embedkgqa(tkbc_model, args)
        dataset      = QA_Dataset_Baseline_muti(split=train_split, dataset_name=args.dataset_name)
        test_dataset = QA_Dataset_Baseline_muti(split=test,        dataset_name=args.dataset_name)
        val_dataset  = QA_Dataset_Baseline_muti(split=val,         dataset_name=args.dataset_name)
    else:
        qa_model     = QA_embedkgqa(tkbc_model, args)
        dataset      = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
        test_dataset = QA_Dataset_Baseline(split=test,        dataset_name=args.dataset_name)
        val_dataset = QA_Dataset_Baseline(split=val,        dataset_name=args.dataset_name)

elif args.model == 'cronkgqa' and args.supervision != 'hard':
    if args.dataset_name == 'MultiTQ':
        qa_model     = QA_cronkgqa(tkbc_model, args)
        dataset      = QA_Dataset_Baseline_muti(split=train_split, dataset_name=args.dataset_name)
        test_dataset = QA_Dataset_Baseline_muti(split=test,        dataset_name=args.dataset_name)
        val_dataset  = QA_Dataset_Baseline_muti(split=val,         dataset_name=args.dataset_name)
    else:
        qa_model     = QA_cronkgqa(tkbc_model, args)
        dataset      = QA_Dataset_Baseline(split=train_split, dataset_name=args.dataset_name)
        test_dataset = QA_Dataset_Baseline(split=test,        dataset_name=args.dataset_name)
        val_dataset = QA_Dataset_Baseline(split=val,        dataset_name=args.dataset_name)

elif args.model == 'sabet' and args.supervision != 'hard':
    if args.dataset_name == 'MultiTQ':
        qa_model     = QA_Sabet(tkbc_model, args)
        dataset      = QA_Dataset_MultiQA_Advanced(split=train_split, dataset_name=args.dataset_name, replace_entity_spans_with_mask=True, args=args)
        test_dataset = QA_Dataset_MultiQA_Advanced(split=test,        dataset_name=args.dataset_name, replace_entity_spans_with_mask=True, args=args)
        val_dataset  = QA_Dataset_MultiQA_Advanced(split=val,         dataset_name=args.dataset_name, replace_entity_spans_with_mask=True, args=args)
    else:
        qa_model     = QA_Sabet(tkbc_model, args)
        dataset      = QA_Dataset_SubGTR(split=train_split, dataset_name=args.dataset_name, replace_entity_spans_with_mask=True, args=args)
        test_dataset = QA_Dataset_SubGTR(split=test,        dataset_name=args.dataset_name, replace_entity_spans_with_mask=True, args=args)
        val_dataset  = QA_Dataset_SubGTR(split=val,         dataset_name=args.dataset_name, replace_entity_spans_with_mask=True, args=args)

elif args.model == 'tempo_qr':
    if args.dataset_name == 'MultiTQ':
        qa_model = QA_TempoQR(tkbc_model, args)
        dataset = QA_Dataset_MultiQA_Advanced(split=train_split, dataset_name=args.dataset_name, args=args, replace_entity_spans_with_mask=True)
        test_dataset = QA_Dataset_MultiQA_Advanced(split=test, dataset_name=args.dataset_name, args=args, replace_entity_spans_with_mask=True)
        val_dataset = QA_Dataset_MultiQA_Advanced(split=val, dataset_name=args.dataset_name, args=args, replace_entity_spans_with_mask=True)
    else:
        qa_model = QA_TempoQR(tkbc_model, args)
        dataset = QA_Dataset_SubGTR(split=train_split, dataset_name=args.dataset_name, args=args, replace_entity_spans_with_mask=True)
        test_dataset = QA_Dataset_SubGTR(split=test, dataset_name=args.dataset_name, args=args, replace_entity_spans_with_mask=True)
        val_dataset = QA_Dataset_SubGTR(split=val, dataset_name=args.dataset_name, args=args, replace_entity_spans_with_mask=True)

elif args.model == 'subgtr':
    qa_model = QA_SubGTR(tkbc_model, args)
    dataset = QA_Dataset_SubGTR(split=train_split, dataset_name=args.dataset_name, args=args)
    test_dataset = QA_Dataset_SubGTR(split=test, dataset_name=args.dataset_name, args=args)
    val_dataset = QA_Dataset_SubGTR(split=val, dataset_name=args.dataset_name, args=args)
else:
    print(f"Model {args.model} not implemented!")
    exit(0)


if args.load_from != '':
    filename = (
        f"{data_dir}"
        f"/qa_models/{args.dataset_name}/{args.load_from}.ckpt"
    )
    print('Loading model from', filename)
    qa_model.load_state_dict(torch.load(filename))
    print('Loaded qa model from', filename)

    tkbc_model = loadTkbcModel(
        f"{data_dir}/models/{args.dataset_name}"
        f"/kg_embeddings/{args.tkbc_model_file}"
    )
    qa_model.tkbc_model = tkbc_model

    num_entities      = tkbc_model.embeddings[0].weight.shape[0]
    num_times         = tkbc_model.embeddings[2].weight.shape[0]
    full_embed_matrix = torch.cat([
        tkbc_model.embeddings[0].weight.data,
        tkbc_model.embeddings[2].weight.data,
    ], dim=0)
    qa_model.entity_time_embedding = torch.nn.Embedding(
        num_entities + num_times + 1,
        qa_model.tkbc_embedding_dim,
        padding_idx=num_entities + num_times,
    )
    qa_model.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)
    for param in tkbc_model.parameters():
        param.requires_grad = False
else:
    print('Not loading from checkpoint. Starting fresh!')

qa_model = qa_model.cuda()

if args.mode == 'eval':

    hop_file = f"results/{args.dataset_name}/{args.save_to}_test_hop_weights.pt"

    score, log = eval(
        qa_model, test_dataset,
        batch_size=args.valid_batch_size,
        split=args.eval_split, k=args.eval_k,
        hop_weights_file=hop_file
    )
    exit(0)


os.makedirs(os.path.dirname(f"results/{args.dataset_name}/{args.save_to}.log"), exist_ok=True)
result_filename = f"results/{args.dataset_name}/{args.save_to}.log"

train(qa_model, dataset, val_dataset, args, result_filename=result_filename)

# ── Load best model ─────────────────────────────────────────────
# create path first
os.makedirs(os.path.dirname(f"{data_dir}/qa_models/{args.dataset_name}"), exist_ok=True)
checkpoint_path = (
    f"{data_dir}"
    f"/qa_models/{args.dataset_name}/{args.save_to}.ckpt"
)

print('Loading best model from', checkpoint_path)
qa_model.load_state_dict(torch.load(checkpoint_path))
qa_model = qa_model.cuda()
qa_model.eval()

# ── Final evaluation on TEST ────────────────────────────────────
score, log = eval(
    qa_model, test_dataset,
    batch_size=args.valid_batch_size,
    split='test',
    k=args.eval_k,
)

append_log_to_file(log, args.max_epochs, result_filename)

print('Final TEST evaluation (best model) done')
print('Training finished')