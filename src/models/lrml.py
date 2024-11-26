import torch.nn as nn
import torch
import torch.nn.functional as F

class LRML(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, memory_size, margin=0.2, reg_weight = 0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.key_layer = nn.Parameter(torch.randn(embedding_dim, memory_size))
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
        self.margin = margin
        self.reg_weight = reg_weight
        self.register_buffer('interaction_matrix', None)
        self.chunk_size = 1024
        
        # 임베딩 초기화
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.key_layer, std=0.01)
        nn.init.normal_(self.memory, std=0.01)

    def forward(self, users, items, neg_users, neg_items):
        # 임베딩 검색
        user_embed = self._clip_by_norm(self.user_embedding(users), 1.0)  # (batch_size, embed_dim)
        item_embed = self._clip_by_norm(self.item_embedding(items), 1.0)  # (batch_size, embed_dim)
        neg_user_embed = self._clip_by_norm(self.user_embedding(neg_users), 1.0)  # (batch_size, embed_dim)
        neg_item_embed = self._clip_by_norm(self.item_embedding(neg_items), 1.0)  # (batch_size, embed_dim)        
        
        # User-Item Pair에 대한 Interaction 및 Relation 계산
        pair_interaction = user_embed * item_embed  # (batch_size, embed_dim)
        pair_keys = torch.matmul(pair_interaction, self.key_layer)  # (batch_size, memory_size)
        pair_attention = torch.softmax(pair_keys, dim=-1)  # (batch_size, memory_size)
        
        # Pair-based Relation vector 계산
        pair_relation = torch.matmul(pair_attention, self.memory)  # (batch_size, embed_dim)
        
        # Positive Score 계산
        user_translated = user_embed + pair_relation
        pos_scores = -torch.sqrt(torch.sum((user_translated - item_embed).pow(2), dim=-1) + 1e-3)  # (batch_size,)
        
        # Negative Score 계산
        neg_user_translated = neg_user_embed + pair_relation
        neg_scores = -torch.sqrt(torch.sum((neg_user_translated - neg_item_embed).pow(2), dim=-1) + 1e-3)  # (batch_size,)

        # self._project_embeddings()
        
        return pos_scores, neg_scores
    
    def compute_loss(self, pos_scores, neg_scores):
        loss = torch.sum(F.relu(self.margin - pos_scores + neg_scores))
        
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        
        return loss + l2_loss * self.reg_weight
    
    def set_interaction_matrix(self, interaction_matrix):
        """기존 상호작용 정보 설정"""
        self.interaction_matrix = interaction_matrix.to(self.memory.device)
    
    @torch.no_grad()
    def batch_full_sort_predict(self, user_ids, filter_seen=True):
        
        user_embeds = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        num_users = user_ids.size(0)
        num_items = self.item_embedding.weight.size(0)
        
        # 결과를 저장할 텐서
        all_scores = torch.empty(num_users, num_items, device=user_embeds.device)
        
        # 유저 청크 단위로 처리
        for u_start in range(0, num_users, self.chunk_size):
            u_end = min(u_start + self.chunk_size, num_users)
            user_chunk = user_embeds[u_start:u_end]  # (chunk_size, embedding_dim)
            
            chunk_scores = []
            # 아이템 청크 단위로 처리
            for i_start in range(0, num_items, self.chunk_size):
                i_end = min(i_start + self.chunk_size, num_items)
                item_chunk = self.item_embedding.weight[i_start:i_end]  # (chunk_size, embedding_dim)
                
                # Attention 계산
                # (chunk_size_u, 1, embed) * (1, chunk_size_i, embed) -> (chunk_size_u, chunk_size_i, embed)
                interaction = user_chunk.unsqueeze(1) * item_chunk.unsqueeze(0)
                keys = torch.matmul(interaction.view(-1, interaction.size(-1)), self.key_layer) # chunk_size, memory_size
                attention = F.softmax(keys, dim=-1)
                relation = torch.matmul(attention, self.memory)
                # 변환된 유저 임베딩
                user_translated = user_chunk.unsqueeze(1) + relation.view(user_chunk.size(0), -1, user_chunk.size(-1))
                # 점수 계산
                chunk_score = -torch.norm(
                    user_translated - item_chunk.unsqueeze(0),
                    p=2, dim=-1
                ).pow(2)
                
                chunk_scores.append(chunk_score)
            
            # 현재 유저 청크의 모든 아이템 점수 결합
            all_scores[u_start:u_end] = torch.cat(chunk_scores, dim=1)
        # 이미 상호작용한 아이템 필터링
        if filter_seen and self.interaction_matrix is not None:
            seen_mask = self.interaction_matrix[user_ids]
            all_scores[seen_mask == 1] = -1e9
            
            
        return all_scores
    
    # 유저, 아이템 임베딩 weight 정규화
    def _project_embeddings(self):
        with torch.no_grad():
            user_norms = torch.norm(self.user_embedding.weight.data, p=2, dim=1, keepdim=True)
            self.user_embedding.weight.data /= torch.clamp(user_norms, min=1.0)
            
            item_norms = torch.norm(self.item_embedding.weight.data, p=2, dim=1, keepdim=True)
            self.item_embedding.weight.data /= torch.clamp(item_norms, min=1.0)
            
    def _clip_by_norm(self, tensor, max_norm):
        norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)  # L2 노름 계산
        factor = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
        return tensor * factor