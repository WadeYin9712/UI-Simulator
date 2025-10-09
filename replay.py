from actions.action import *
from simulator import *
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
import shutil
import argparse
from transformers import AutoTokenizer, AutoModel
import torch

# embed the task instructions in the training set using BERT model
def embed_task_instructions(instructions: list[str], model='FacebookAI/roberta-large', batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in trange(0, len(instructions), batch_size):
            batch = instructions[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(model.device)
            outputs = model(**inputs)
            # Use the mean of the last hidden states as the embedding
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# data selection strategy for replay
def select_data_for_replay(args):
    prev_trajs = os.listdir(args.prev_path)
    prev_trajs = sorted([os.path.join(args.prev_path, traj) for traj in prev_trajs if not traj.startswith('.')])

    # read the task instructions from the trajs
    prev_instructions = []
    for traj in prev_trajs:
        if os.path.exists(os.path.join(traj, 'instruction.txt')):
            with open(os.path.join(traj, 'instruction.txt'), 'r') as f:
                instruction = f.read().strip()
                prev_instructions.append(instruction)
        else:
            with open(os.path.join(traj, 'trajectory.pkl'), 'rb') as f:
                instruction = pickle.load(f)[0][-1]
                prev_instructions.append(instruction)

    if not os.path.exists('replay/prev_task_embeddings.npy') or args.rerun_embedding:
        # embed the task instructions
        prev_embeddings = embed_task_instructions(prev_instructions)

        # saved as npy file
        np.save('replay/prev_task_embeddings.npy', prev_embeddings)
    else:
        prev_embeddings = np.load('replay/prev_task_embeddings.npy')

    # load data in the current stage
    current_trajs = os.listdir(args.cur_path)
    current_trajs = sorted([os.path.join(args.cur_path, traj) for traj in current_trajs if not traj.startswith('.')])

    current_instructions = []
    for traj in current_trajs:
        if os.path.exists(os.path.join(traj, 'instruction.txt')):
            with open(os.path.join(traj, 'instruction.txt'), 'r') as f:
                instruction = f.read().strip()
                current_instructions.append(instruction)
        else:
            with open(os.path.join(traj, 'trajectory.pkl'), 'rb') as f:
                instruction = pickle.load(f)[0][-1]
                current_instructions.append(instruction)
    if not os.path.exists('replay/current_task_embeddings.npy') or args.rerun_embedding:
        # embed the task instructions
        current_embeddings = embed_task_instructions(current_instructions)

        # saved as npy file
        np.save('replay/current_task_embeddings.npy', current_embeddings)
    else:
        current_embeddings = np.load('replay/current_task_embeddings.npy')
    # compute two cosine similarity matrices, namely Scp and Spp
    Scp = cosine_similarity(current_embeddings, prev_embeddings)
    Spp = cosine_similarity(prev_embeddings, prev_embeddings)

    print("Scp shape:", Scp.shape)
    print("Spp shape:", Spp.shape)

    # optional: doing PCA to reduce the dimensionality of the embeddings
    # pca = PCA(n_components=64)
    # Scp_reduced = pca.fit_transform(Scp)
    # Spp_reduced = pca.fit_transform(Spp)

    # select data based on the replay strategy
    num_selection = int(len(prev_instructions) * args.scale)
    selected_indices = []
    if args.replay_strategy == 'instr_diverse':
        # choose tasks with least column sum in Scp
        col_sums = Scp.sum(axis=0)
        selected_indices = np.argsort(col_sums)[:num_selection].tolist()
    elif args.replay_strategy == 'instr_similar':
        # choose tasks with highest column sum in Scp
        col_sums = Scp.sum(axis=0)
        selected_indices = np.argsort(col_sums)[-num_selection:].tolist()
    elif args.replay_strategy == 'instr_support':
        # choose tasks with the highest row sum in Spp
        row_sums = Spp.sum(axis=1)
        selected_indices = np.argsort(row_sums)[-num_selection:].tolist()
    elif args.replay_strategy == 'random':
        selected_indices = np.random.choice(len(prev_instructions), size=num_selection, replace=False).tolist()

    selected_samples = [prev_trajs[i] for i in selected_indices]
    selected_tasks = [prev_instructions[i] for i in selected_indices]

    with open(f'replay/{args.replay_strategy}_selected_tasks.txt', 'w') as f:
        for task in selected_tasks:
            f.write(f"{task}\n")

    # copy the selected trajectory folders to the replay folder
    s = str(args.scale).replace('.', '_')
    replay_folder = f'replay/{args.replay_strategy}_{s}'
    os.makedirs(replay_folder, exist_ok=True)
    for traj in selected_samples:
        shutil.copytree(traj, os.path.join(replay_folder, os.path.basename(traj)))

    return selected_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_embedding', action='store_true', help='whether to rerun the embedding for task instructions')
    parser.add_argument('--replay_strategy', type=str, default='instr_diverse', choices=['instr_diverse', 'random', 'instr_similar', 'instr_support'], help='strategy for selecting data for replay')
    parser.add_argument('--scale', type=float, default=1.0, help='scale factor for the number of selected samples')
    parser.add_argument('--prev_path', type=str, required=True, help='the path to the previous stage data', required=True)
    parser.add_argument('--cur_path', type=str, required=True, help='the path to the current stage data', required=True)

    args = parser.parse_args()

    selected_samples = select_data_for_replay(args)