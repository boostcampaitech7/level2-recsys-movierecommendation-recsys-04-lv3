import os
import pandas as pd
import math
import yaml
from collections import defaultdict

def load_model_files(config_path='config.yaml'):
    """
    config.yaml 파일에서 모델 파일 목록을 읽어오는 함수.

    Args:
        config_path (str, optional): 모델 파일 목록이 저장된 YAML 파일 경로. 기본값은 'config.yaml'.

    Returns:
        list: YAML 파일에서 읽어온 모델 파일 경로 목록. 'model_files' 키가 존재하지 않으면 빈 리스트를 반환.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('model_files', [])


def simple_hard_voting_recommendation(model_files: list[str], top_n: int = 10) -> pd.DataFrame:
    """
    간단한 하드 보팅 방식으로 추천 항목을 생성하는 함수.
    각 모델의 추천 결과에서 가장 많이 추천된 항목들을 top_n 개 선택하여 최종 추천 리스트를 반환합니다.
    
    Args:
        model_files (list of str): 각 모델의 추천 결과가 담긴 CSV 파일 경로 목록.
        top_n (int, optional): 추천할 상위 항목의 개수 (기본값: 10).
        
    Returns:
        pd.DataFrame: 사용자별로 추천된 상위 항목들로 구성된 데이터프레임.
            컬럼은 'user', 'item'으로, 각 사용자에게 추천된 항목들이 포함됩니다.
    """
    all_recommendations = defaultdict(list)
    
    # 각 모델 파일을 처리
    for file in model_files:
        df = pd.read_csv(file)
        
        # 각 사용자별로 추천 항목 추가
        for user, user_df in df.groupby('user'):
            for item in user_df['item']:
                all_recommendations[user].append(item)
    
    # 추천 항목에서 가장 많이 추천된 상위 top_n 항목들 선택
    final_recommendations = []
    for user, items in all_recommendations.items():
        item_counts = pd.Series(items).value_counts()
        top_items = item_counts.nlargest(top_n).index
        for item in top_items:
            final_recommendations.append([user, item])
    
    # 최종 추천 결과를 데이터프레임으로 반환
    result_df = pd.DataFrame(final_recommendations, columns=['user', 'item'])
    
    return result_df

def hard_voting_recommendation(model_files: list[str], top_n: int = 10) -> pd.DataFrame:
    """
    가중치를 고려한 하드 보팅 방식으로 추천 항목을 생성하는 함수.
    각 모델의 추천 항목에서 랭크에 따라 가중치를 부여하고, 이를 기반으로 최종 추천 항목을 선택합니다.
    
    Args:
        model_files (list of str): 각 모델의 추천 결과가 담긴 CSV 파일 경로 목록.
        top_n (int, optional): 추천할 상위 항목의 개수 (기본값: 10).
        
    Returns:
        pd.DataFrame: 사용자별로 추천된 상위 항목들로 구성된 데이터프레임.
            컬럼은 'user', 'item'으로, 각 사용자에게 추천된 항목들이 포함됩니다.
    """
    def ranking_weight(rank: int, total_items: int) -> float:
        """
        항목의 랭크에 기반하여 가중치를 계산하는 내부 함수.
        
        Args:
            rank (int): 항목의 랭크 (1부터 시작).
            total_items (int): 해당 사용자에 대한 전체 추천 항목의 개수.
        
        Returns:
            float: 계산된 가중치.
        """
        weight = 1 + math.log(total_items + 1 - rank)
        return min(weight, 1.5)  # 가중치는 최대 1.5로 제한

    all_recommendations = defaultdict(lambda: defaultdict(float))
    
    # 각 모델 파일을 처리
    for file in model_files:
        df = pd.read_csv(file)
        
        # 각 사용자별로 추천 항목에 가중치 부여
        for user, user_df in df.groupby('user'):
            total_items = len(user_df)
            
            for rank, item in enumerate(user_df['item'], 1):
                weight = ranking_weight(rank, total_items)
                all_recommendations[user][item] += weight
    
    # 각 사용자의 추천 항목을 가중치 합계를 기반으로 정렬하고 top_n 항목을 선택
    final_recommendations = []
    for user, item_scores in all_recommendations.items():
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = [item for item, _ in sorted_items[:top_n]]
        for item in top_items:
            final_recommendations.append([user, item])
    
    # 최종 추천 결과를 데이터프레임으로 반환
    result_df = pd.DataFrame(final_recommendations, columns=['user', 'item'])
    
    return result_df

def main():
    """
    추천 시스템을 실행하는 메인 함수. 사용자가 선택한 보팅 방식을 사용하여 추천 결과를 생성하고 저장합니다.
    """
    model_files = load_model_files()  # config.yaml 파일에서 모델 파일 목록을 읽어옵니다.
    
    voting_method = input("Choose voting method (simple/weighted): ").strip().lower()
    
    if voting_method == "simple":
        recommendations = simple_hard_voting_recommendation(model_files)
        recommendations.to_csv('simple_hard_voting_recommendations.csv', index=False)
        print("Simple Hard Voting Recommendations:")
    elif voting_method == "weighted":
        recommendations = hard_voting_recommendation(model_files)
        recommendations.to_csv('weighted_hard_voting_recommendations.csv', index=False)
        print("Weighted Hard Voting Recommendations:")
    else:
        print("Invalid choice.")
        return
    
    print(recommendations.head(20))

if __name__ == "__main__":
    main()
