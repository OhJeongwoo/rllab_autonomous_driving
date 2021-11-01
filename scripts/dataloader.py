import json
import torch
import numpy as np


class Dataloader:
    def __init__(self, path, N, encoder_rollout, policy_rollout):
        self.trj_path = path
        self.n_trj = N
        self.n_data = 0
        self.encoder_obs = []
        self.policy_obs = []
        self.act = []
        self.cls = []
        self.encoder_rollout = encoder_rollout
        self.policy_rollout = policy_rollout
        self.M = max(self.encoder_rollout, self.policy_rollout)
        self.traj_index = []
        self.traj_type = []

        self.load()


    def load(self):
        for seq in range(self.n_trj):
            trj_file = self.trj_path + str(seq).zfill(6) + ".json"
            with open(trj_file,'r') as jf:
                data = json.load(jf)
            
            # update obs, act, cls array
            n = data['length']
            if n < self.M:
                continue
            self.traj_index.append((self.n_data, self.n_data + n - self.M + 1))
            self.traj_type.append(data['type'])
            for k in range(0, n - self.M +1):
                encoder_obs = []
                for j in range(self.M - self.encoder_rollout, self.M - 1):
                    encoder_obs += data['state'][k + j]
                    encoder_obs += data['action'][k + j]
                encoder_obs += data['state'][k + self.M - 1]
                self.encoder_obs.append(encoder_obs)
                policy_obs = []
                for j in range(self.M - self.policy_rollout, self.M - 1):
                    policy_obs += data['state'][k + j]
                    policy_obs += data['action'][k + j]
                policy_obs += data['state'][k + self.M - 1]
                self.policy_obs.append(policy_obs)
                self.act.append(data['action'][k + self.M - 1])
            self.cls += [seq for _ in range(n - self.M + 1)]
            self.n_data += n - self.M + 1
        
        self.n_cls = len(self.traj_index)
        self.train_cls = [i for i in range(0, int(self.n_cls * 0.8))]
        self.test_cls = [i for i in range(int(self.n_cls * 0.8), self.n_cls)]
        self.dataset_boundary = int(self.n_cls * 0.8)
        self.n_data_boundary = self.traj_index[self.dataset_boundary][0]
        self.encoder_obs = np.array(self.encoder_obs)
        self.policy_obs = np.array(self.policy_obs)
        self.act = np.array(self.act)
        self.cls = np.array(self.cls)
        self.encoder_obs_mean = np.mean(self.encoder_obs, axis = 0)
        self.encoder_obs_std = np.std(self.encoder_obs, axis = 0)
        self.policy_obs_mean = np.mean(self.policy_obs, axis = 0)
        self.policy_obs_std = np.std(self.policy_obs, axis = 0)
        self.act_mean = np.mean(self.act, axis = 0)
        self.act_std = np.std(self.act, axis = 0)
        self.relation_matrix = np.array([[False for _ in range(self.n_cls)] for _ in range(self.n_cls)])
        for i in range(self.n_cls):
            self.relation_matrix[i][i] = True

        self.encoder_obs = (self.encoder_obs - self.encoder_obs_mean) / self.encoder_obs_std
        self.policy_obs = (self.policy_obs - self.policy_obs_mean) / self.policy_obs_std
        self.act = (self.act - self.act_mean) / self.act_std

        print("Finish Load!!!")
        print("# of data: %d" %(self.n_data))
        print("encoder state dim: %d" %(self.encoder_obs.shape[1]))
        print("policy state dim: %d" %(self.policy_obs.shape[1]))
        print("action dim: %d" %(self.act.shape[1]))
        self.encoder_state_dim = self.encoder_obs.shape[1]
        self.policy_state_dim = self.policy_obs.shape[1]
        self.action_dim = self.act.shape[1]

    def update_relation(self, P_cls, Q_cls, similarity):
        N = similarity.shape[0]
        prob = np.random.rand(N)
        for i in range(N):
            if P_cls[i] == Q_cls[i]:
                continue
            if 0.5 * (similarity[i] + 1) < prob[i]:
                self.relation_matrix[P_cls[i]][Q_cls[i]] = False
            else :
                self.relation_matrix[P_cls[i]][Q_cls[i]] = True
        
    def shuffle(self, size, type):
        if type == 0:
            arr = np.arange(self.n_data_boundary)
        else :
            arr = np.arange(self.n_data_boundary, self.n_data)
        np.random.shuffle(arr)
        return self.encoder_obs[arr[:size]], self.policy_obs[arr[:size]], self.act[arr[:size]], self.cls[arr[:size]]

    
    def sample_same_traj(self, size):
        cls_arr = np.arange(self.dataset_boundary, self.n_cls)
        P_idx_list = []
        Q_idx_list = []
        while size > 0:
            np.random.shuffle(cls_arr)
            P_cls = cls_arr[0]
            Q_cls = cls_arr[1]
            if self.traj_type[P_cls] == self.traj_type[Q_cls]:
                P_idx = np.random.randint(self.traj_index[P_cls][0], self.traj_index[P_cls][1], size = 1)[0]
                Q_idx = np.random.randint(self.traj_index[Q_cls][0], self.traj_index[Q_cls][1], size = 1)[0]
                P_idx_list.append(P_idx)
                Q_idx_list.append(Q_idx)
                size -= 1
        return self.encoder_obs[P_idx_list], self.policy_obs[P_idx_list], self.act[P_idx_list], self.encoder_obs[Q_idx_list], self.policy_obs[Q_idx_list], self.act[Q_idx_list]


    def sample_diff_traj(self, size):
        cls_arr = np.arange(self.dataset_boundary, self.n_cls)
        P_idx_list = []
        Q_idx_list = []
        while size > 0:
            np.random.shuffle(cls_arr)
            P_cls = cls_arr[0]
            Q_cls = cls_arr[1]
            if self.traj_type[P_cls] != self.traj_type[Q_cls]:
                P_idx = np.random.randint(self.traj_index[P_cls][0], self.traj_index[P_cls][1], size = 1)[0]
                Q_idx = np.random.randint(self.traj_index[Q_cls][0], self.traj_index[Q_cls][1], size = 1)[0]
                P_idx_list.append(P_idx)
                Q_idx_list.append(Q_idx)
                size -= 1
        return self.encoder_obs[P_idx_list], self.policy_obs[P_idx_list], self.act[P_idx_list], self.encoder_obs[Q_idx_list], self.policy_obs[Q_idx_list], self.act[Q_idx_list]

