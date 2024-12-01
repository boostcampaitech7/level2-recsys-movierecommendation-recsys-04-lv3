from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from collections import defaultdict

class PairwiseDataset(Dataset):
    """
    추천 시스템에서 pairwise 학습을 위한 PyTorch Dataset 클래스.
    Args:
        interaction (pd.DataFrame): 사용자-아이템 상호작용을 포함하는 DataFrame.
        num_neg_samples (int, optional): 사용자당 생성할 negative 샘플 수. 기본값은 5.
        cache_size (int, optional): negative 샘플 캐시의 크기. 기본값은 100000.
        user_map (dict, optional): 사용자 ID를 인덱스로 매핑하는 딕셔너리. 기본값은 None.
        item_map (dict, optional): 아이템 ID를 인덱스로 매핑하는 딕셔너리. 기본값은 None.
        idx_user_map (dict, optional): 인덱스를 사용자 ID로 매핑하는 딕셔너리. 기본값은 None.
        idx_item_map (dict, optional): 인덱스를 아이템 ID로 매핑하는 딕셔너리. 기본값은 None.
        test (bool, optional): True일 경우, 테스트용 데이터셋으로 사용되며 negative 샘플링 캐시가 생성되지 않음. 기본값은 False.
    Attributes:
        interaction (pd.DataFrame): 사용자-아이템 상호작용을 포함하는 DataFrame.
        cache_size (int): negative 샘플 캐시의 크기.
        num_neg_samples (int): 사용자당 생성할 negative 샘플 수.
        user_map (dict): 사용자 ID를 인덱스로 매핑하는 딕셔너리.
        item_map (dict): 아이템 ID를 인덱스로 매핑하는 딕셔너리.
        idx_user_map (dict): 인덱스를 사용자 ID로 매핑하는 딕셔너리.
        idx_item_map (dict): 인덱스를 아이템 ID로 매핑하는 딕셔너리.
        users (np.ndarray): 사용자 인덱스 배열.
        items (np.ndarray): 아이템 인덱스 배열.
        user_positive (defaultdict): 각 사용자의 positive 아이템을 저장하는 딕셔너리.
        item_users (defaultdict): 각 아이템의 사용자를 저장하는 딕셔너리.
        num_users (int): 고유 사용자 수.
        num_items (int): 고유 아이템 수.
        all_users (np.ndarray): 모든 사용자 인덱스 배열.
        all_items (np.ndarray): 모든 아이템 인덱스 배열.
        neg_cache (np.ndarray): negative 샘플 캐시.
        cache_pointer (int): negative 샘플 캐시의 현재 위치를 가리키는 포인터.
        user_set_cache (dict): 각 사용자의 가능한 negative 아이템을 저장하는 딕셔너리.
    Methods:
        _prepare_neg_cache(): negative 샘플 캐시를 준비하는 메서드 (버전 1).
        _prepare_neg_cache_2(): negative 샘플 캐시를 준비하는 메서드 (버전 2).
        create_interaction_matrix(): 사용자-아이템 상호작용으로부터 상호작용 행렬을 생성하는 메서드.
        __len__(): 사용자-아이템 상호작용의 수를 반환하는 메서드.
        __getitem__(idx): 주어진 인덱스에 대해 사용자, 아이템, negative 사용자, negative 아이템을 포함하는 딕셔너리를 반환하는 메서드.
    """
    def __init__(self, interaction, num_neg_samples=5, cache_size=100000, 
                 user_map=None, item_map=None, idx_user_map=None, idx_item_map=None, test=False):
        self.interaction = interaction
        self.cache_size = cache_size
        self.num_neg_samples = num_neg_samples
        
        self.user_map = user_map
        self.item_map = item_map
        self.idx_user_map = idx_user_map
        self.idx_item_map = idx_item_map
        
        # numpy 배열로 변환
        self.users = np.array([self.user_map[u] for u in interaction['user']])
        self.items = np.array([self.item_map[i] for i in interaction['item']])
        
        # 사용자별 positive 아이템 저장
        self.user_positive = defaultdict(set)
        self.item_users = defaultdict(set)
        for u, i in tqdm(zip(self.users, self.items), desc='Preparing user, item dict', total=len(self.users)):
            self.user_positive[u].add(i)
            self.item_users[i].add(u)
        
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        
        self.all_users = np.arange(self.num_users)
        self.all_items = np.arange(self.num_items)

        # 캐시 초기화
        if not test:
            # self.neg_cache = self._prepare_neg_cache()
            self.neg_cache = self._prepare_neg_cache_2()
            self.cache_pointer = 0
    
    def _prepare_neg_cache(self): # version 1 item을 cache에 저장해서 속도가 빠르긴 하지만, negative sample이 고정임.
        # 캐시 초기화
        cache = np.zeros((self.cache_size, 2), dtype=np.int64)

        # 각 유저의 출현 빈도 계산 (예: 해당 유저의 positive 아이템 수)
        user_frequencies = np.array([len(self.user_positive[user]) for user in range(self.num_users)])
        total_frequency = user_frequencies.sum()
        user_sampling_probs = user_frequencies / total_frequency

        sampled_users = np.random.choice(
            np.arange(self.num_users), 
            size=self.cache_size, 
            p=user_sampling_probs, 
            replace=True
        )
        sampled_users.sort()
        
        user_temp = -1
        for i, user in tqdm(enumerate(sampled_users), desc='Negative sampling', total=len(sampled_users)):
            if user_temp != user:
                pos_items_neg_user = self.user_positive[user]
                possible_items_for_neg_user = np.array(list(set(range(self.num_items)) - pos_items_neg_user))
                user_temp = user

            neg_item = np.random.choice(possible_items_for_neg_user)
            
            cache[i] = [user, neg_item]
            
        # 최종 캐시 셔플
        np.random.shuffle(cache)

        return cache
    
    def _prepare_neg_cache_2(self):
        cache = np.zeros((self.cache_size, 1), dtype=np.int64)
        self.user_set_cache = dict()
        for user in tqdm(range(self.num_users)):
            pos_items_neg_user = self.user_positive[user]
            self.user_set_cache[user] = np.array(list(set(range(self.num_items)) - pos_items_neg_user))
        
        # 각 유저의 출현 빈도 계산 (예: 해당 유저의 positive 아이템 수)
        user_frequencies = np.array([len(self.user_positive[user]) for user in range(self.num_users)])
        total_frequency = user_frequencies.sum()
        user_sampling_probs = user_frequencies / total_frequency
        
        sampled_users = np.random.choice(
            np.arange(self.num_users), 
            size=self.cache_size, 
            p=user_sampling_probs, 
            replace=True
        )
        sampled_users.sort()
        
        for i, user in tqdm(enumerate(sampled_users), desc='Negative sampling', total=len(sampled_users)):
            cache[i] = [user]    

        # 최종 캐시 셔플
        np.random.shuffle(cache)

        return cache

    def create_interaction_matrix(self):
        """학습 데이터로부터 상호작용 행렬 생성"""
        rows = []
        cols = []
        for user, items in self.user_positive.items():
            rows.extend([user] * len(items))
            cols.extend(list(items))
        data = np.ones_like(rows)
        return csr_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        # 캐시에서 샘플 가져오기
        # neg_user, neg_item = self.neg_cache[self.cache_pointer] # version 1
        neg_user = self.neg_cache[self.cache_pointer][0] # version 2 -> 학습마다 negative item 뽑음
        neg_item = np.random.choice(self.user_set_cache[neg_user])
        self.cache_pointer = (self.cache_pointer + 1) % self.cache_size
        
        return {
            'user': self.users[idx],
            'item': self.items[idx],
            'neg_user': neg_user,
            'neg_item': neg_item
        }