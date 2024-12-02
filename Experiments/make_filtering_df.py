import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import argparse
from tqdm import tqdm

class UserItemFilteringMatrix:
    """
    A class used to create and manage a user-item interaction matrix.
    
    Attributes:
        data_dir (Path): Directory path containing the data files.
        user_item_matrix (csr_matrix): The user-item interaction matrix.
        user2id (dict): Mapping from user IDs to matrix indices.
        item2id (dict): Mapping from item IDs to matrix indices.
        id2user (dict): Mapping from matrix indices to user IDs.
        id2item (dict): Mapping from matrix indices to item IDs.
        diff_max_days (int): Maximum number of days between the first and last interaction.
        non_masking_day (int): Number of days to not mask the items.
    
    Methods:
        save_matrix(filename='user_item_matrix.npy'):
            Save the user-item interaction matrix to a file.
        load_matrix(filename='user_item_matrix.npy'):
            Load the user-item interaction matrix from a file.
        get_matrix():
            Return the user-item interaction matrix.
    """
    
    def __init__(self, data_dir='./data/train', diff_max_days=5, non_masking_day=0):
        """
        Initialize UserItemMatrix class
        
        Args:
            data_dir (str): Directory path containing the data files
            diff_max_days (int): Maximum number of days between the first and last interaction
            non_masking_day (int): Number of days to not mask the items
        """
        self.data_dir = Path(data_dir)
        self.user_item_matrix = None
        self.user2id = None
        self.item2id = None
        self.id2user = None
        self.id2item = None
        self.diff_max_days = diff_max_days
        self.non_masking_day = non_masking_day
        
    def _load_data(self):
        """Load and preprocess the training data and years data"""
        train_df = pd.read_csv(self.data_dir / 'train_ratings.csv')
        years_df = pd.read_csv(self.data_dir / 'years.tsv', sep='\t')
        
        return train_df, years_df
    
    def _process_user_time(self, train_df):
        """
        Process user time information and filter users based on time difference
        
        Args:
            train_df (pd.DataFrame): Training dataframe
            
        Returns:
            pd.DataFrame: Processed user time information
        """
        train_df = train_df.copy()
        train_df["time"] = pd.to_datetime(train_df["time"], unit='s')
        
        user_time = train_df.groupby('user')['time'].agg(['min', 'max']).reset_index()
        user_time['diff'] = user_time['max'] - user_time['min']
        user_time['diff'] = user_time['diff'].dt.days

        # max + alpha days
        user_time["max"] = user_time["max"] + pd.Timedelta(days=self.non_masking_day)
        
        user_time["max_year"] = user_time["max"].dt.year
        
        return user_time[user_time['diff'] <= self.diff_max_days]
    
    def _create_id_mappings(self, train_df):
        """Create user and item ID mappings"""
        n_users = train_df['user'].nunique()
        n_items = train_df['item'].nunique()
        
        self.user2id = dict(zip(train_df['user'].unique(), range(1, n_users+1)))
        self.item2id = dict(zip(train_df['item'].unique(), range(1, n_items+1)))
        
        self.id2user = {v: k for k, v in self.user2id.items()}
        self.id2item = {v: k for k, v in self.item2id.items()}
        
        return n_users, n_items
    
    def create_matrix(self):
        """Create user-item matrix"""
        train_df, years_df = self._load_data()
        user_time = self._process_user_time(train_df)
        n_users, n_items = self._create_id_mappings(train_df)
        
        # Initialize matrix
        user_item_matrix = np.zeros((n_users+1, n_items+1))
        
        # Fill matrix
        for row in tqdm(user_time.itertuples(), total=len(user_time)):
            user = row[1]
            user = self.user2id[user]
            max_year = row[5]
            items = years_df[years_df['year'] > max_year]['item'].map(self.item2id).values
            
            user_item_matrix[user, items] = 1
        
        self.user_item_matrix = csr_matrix(user_item_matrix)
        return self
    
    def save_matrix(self, filename='user_item_matrix.npy'):
        """
        Save the user-item matrix
        
        Args:
            filename (str): Name of the file to save the matrix
        """
        if self.user_item_matrix is None:
            raise ValueError("Matrix has not been created yet. Call create_matrix() first.")
            
        save_path = self.data_dir / filename
        np.save(save_path, self.user_item_matrix)
        
    def load_matrix(self, filename='user_item_matrix.npy'):
        """
        Load the user-item matrix
        
        Args:
            filename (str): Name of the file to load the matrix from
        """
        load_path = self.data_dir / filename
        self.user_item_matrix = np.load(load_path, allow_pickle=True)[()]
        return self
    
    def get_matrix(self):
        """Return the user-item matrix"""
        if self.user_item_matrix is None:
            raise ValueError("Matrix has not been created or loaded yet.")
        return self.user_item_matrix
    
# 파일 save
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/train", type=str)
    parser.add_argument("--diff_max_days", default=5, type=int)
    parser.add_argument("--non_masking_day", default=0, type=int)
    
    args = parser.parse_args()
    
    uim = UserItemFilteringMatrix(data_dir=args.data_dir, diff_max_days=args.diff_max_days, non_masking_day=args.non_masking_day)
    uim.create_matrix().save_matrix()
    print("Matrix saved")