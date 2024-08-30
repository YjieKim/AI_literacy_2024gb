import random

# 성격 유형별 운동 리스트
wo_ISTJ = ['마라톤', '스쿼시', '체육관에서의 전신 운동', '골프', '배드민턴']
wo_ISFJ = ['스트레칭', '에어로빅', '필라테스', '수영', '요가']
wo_INFJ = ['하이킹', '승마', '명상', '카누', '트레킹']
wo_INTJ = ['체스복싱', '펜싱', '러닝머신 운동', '암벽 등반', '크로스핏']
wo_ISTP = ['킥복싱', '파쿠르', '모터스포츠', '사격', '스쿠버 다이빙']
wo_ISFP = ['발레', '플라잉 요가', '스탠드업 패들보딩', '사교댄스', '아크로요가']
wo_INFP = ['현대무용', '조깅', '플로깅 (쓰레기를 주우며 조깅)', '트램폴린 운동', '볼더링']
wo_INTP = ['비치 발리볼', '파워 워킹', '실내 암벽 등반', '사이클링', '궁도']
wo_ESTP = ['라크로스', '스카이다이빙', '래프팅', '제트스키', '스노우보드']
wo_ESFP = ['서핑', '줌바', '카약', '농구', '탁구']
wo_ENFP = ['실내 클라이밍', '댄스', '태극권', '크로스컨트리 스키', '캡스톤볼']
wo_ENTP = ['탁구', '서핑', '트램폴린 운동', '스노클링', '스케이트보드']
wo_ESTJ = ['하키', '보디빌딩', '양궁', '럭비', '수상스키']
wo_ESFJ = ['요트', '양궁', '실내 사이클링', '트램펄린 피트니스', '페인트볼']
wo_ENFJ = ['발레 피트니스', '승마', '피트니스 다이빙', '실내 자전거', '저글링 피트니스']
wo_ENTJ = ['세일링', '크로스컨트리 달리기', '레슬링', '실내 조정', '크로스핏']

# 성격 유형에 따른 리스트 매핑
workout_dict = {
    'ISTJ': wo_ISTJ,
    'ISFJ': wo_ISFJ,
    'INFJ': wo_INFJ,
    'INTJ': wo_INTJ,
    'ISTP': wo_ISTP,
    'ISFP': wo_ISFP,
    'INFP': wo_INFP,
    'INTP': wo_INTP,
    'ESTP': wo_ESTP,
    'ESFP': wo_ESFP,
    'ENFP': wo_ENFP,
    'ENTP': wo_ENTP,
    'ESTJ': wo_ESTJ,
    'ESFJ': wo_ESFJ,
    'ENFJ': wo_ENFJ,
    'ENTJ': wo_ENTJ,
}

def get_random_workout(personality_type):
    """성격 유형에 맞는 운동을 랜덤으로 반환합니다."""
    if personality_type in workout_dict:
        workout_list = workout_dict[personality_type]
        return random.choice(workout_list)
    else:
        return "알 수 없는 성격 유형입니다."

# 예시 사용
#personality_type = 'ISTJ'
#random_workout = get_random_workout(personality_type)
#print(f"{personality_type} 유형의 추천 운동: {random_workout}")
