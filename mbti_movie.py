import random

# MBTI 유형별 영화 추천 데이터
mbti_movies = {
    "INTJ": [
        {
            "title": "Inception",
            "korean_title": "인셉션",
            "year": 2010,
            "summary": "꿈을 조작하는 기술을 이용해 목표 인물의 잠재의식을 해킹하려는 이야기를 그린 영화",
            "reason": "복잡한 플롯과 지적 도전을 제공하는 영화로, INTJ의 분석적 사고와 전략적 마인드에 잘 맞을 거예요."
        },
        {
            "title": "The Matrix",
            "korean_title": "매트릭스",
            "year": 1999,
            "summary": "현실이라 믿었던 세계가 사실은 거대한 컴퓨터 시뮬레이션이라는 충격적인 사실을 알게 되는 남자의 이야기",
            "reason": "철학적 질문과 혁신적인 시각 효과가 INTJ의 호기심을 자극할 거예요."
        },
        {
            "title": "Blade Runner 2049",
            "korean_title": "블레이드 러너 2049",
            "year": 2017,
            "summary": "인간과 인공지능의 경계를 탐구하는 미래 디스토피아를 배경으로 한 영화",
            "reason": "심오한 주제와 시각적으로 아름다운 장면들이 INTJ의 심미적 감각과 지적 호기심을 자극할 거예요."
        },
        {
            "title": "Interstellar",
            "korean_title": "인터스텔라",
            "year": 2014,
            "summary": "인류의 생존을 위해 우주로 떠난 탐험대의 이야기를 그린 영화",
            "reason": "과학적 요소와 감정적인 스토리가 조화를 이루어 INTJ의 다양한 관심사를 만족시킬 수 있어요."
        },
        {
            "title": "Ex Machina",
            "korean_title": "엑스 마키나",
            "year": 2015,
            "summary": "인공지능 로봇과의 심리적 대결을 그린 영화",
            "reason": "인공지능과 인간의 관계를 탐구하는 심리적 스릴러가 INTJ의 지적 호기심을 자극할 수 있어요."
        }
    ],
    "INFP": [
        {
            "title": "Amélie",
            "korean_title": "아멜리에",
            "year": 2001,
            "summary": "파리의 한 카페에서 일하는 아멜리가 주변 사람들의 삶을 변화시키는 과정을 그린 영화",
            "reason": "따뜻하고 감성적인 스토리와 독특한 캐릭터들이 INFP의 창의성과 감수성에 잘 맞을 거예요."
        },
        {
            "title": "The Secret Life of Walter Mitty",
            "korean_title": "월터의 상상은 현실이 된다",
            "year": 2013,
            "summary": "평범한 사진 관리자가 꿈을 찾아 떠나는 모험을 그린 영화",
            "reason": "자아 발견과 모험이 가득한 이야기로 INFP의 이상주의적 성향에 잘 맞을 거예요."
        },
        {
            "title": "Eternal Sunshine of the Spotless Mind",
            "korean_title": "이터널 선샤인",
            "year": 2004,
            "summary": "기억을 지우는 기술을 통해 사랑과 이별을 다시 겪게 되는 남녀의 이야기",
            "reason": "감정적인 깊이와 독창적인 플롯이 INFP의 감수성과 상상력을 자극할 거예요."
        },
        {
            "title": "Big Fish",
            "korean_title": "빅 피쉬",
            "year": 2003,
            "summary": "환상적인 이야기 속에 숨겨진 아버지와 아들의 관계를 그린 영화",
            "reason": "판타지와 현실을 넘나드는 스토리가 INFP의 상상력과 감성을 자극할 거예요."
        },
        {
            "title": "Your Name",
            "korean_title": "너의 이름은",
            "year": 2016,
            "summary": "서로의 몸이 바뀌는 신비한 경험을 하게 되는 두 남녀의 이야기",
            "reason": "감성적이고 아름다운 애니메이션이 INFP의 감수성을 자극할 거예요."
        }
    ],
    "ISFJ": [
        {
            "title": "Forrest Gump",
            "korean_title": "포레스트 검프",
            "year": 1994,
            "summary": "지능이 낮지만 순수한 마음을 가진 포레스트 검프의 일대기를 그린 영화",
            "reason": "따뜻한 인간애와 헌신적인 주인공이 ISFJ의 보호적이고 배려심 많은 성향에 잘 맞을 거예요."
        },
        {
            "title": "Pride and Prejudice",
            "korean_title": "오만과 편견",
            "year": 2005,
            "summary": "제인 오스틴의 소설을 바탕으로 한 사랑과 사회적 신분을 그린 영화",
            "reason": "고전적이고 로맨틱한 이야기가 ISFJ의 전통적 가치와 감성에 잘 맞을 거예요."
        },
        {
            "title": "The Pursuit of Happyness",
            "korean_title": "행복을 찾아서",
            "year": 2006,
            "summary": "어려운 상황에서도 아들을 위해 끊임없이 노력하는 아버지의 이야기를 그린 영화",
            "reason": "가족을 위한 헌신과 인내의 이야기가 ISFJ의 책임감과 가족 중심적 성향에 잘 맞을 거예요."
        },
        {
            "title": "The Help",
            "korean_title": "헬프",
            "year": 2011,
            "summary": "1960년대 미시시피를 배경으로 한 흑인 가정부와 백인 여성의 우정을 그린 영화",
            "reason": "사회적 이슈와 개인적 용기를 다룬 이야기로 ISFJ의 정의감과 공감을 자극할 거예요."
        },
        {
            "title": "Little Women",
            "korean_title": "작은 아씨들",
            "year": 2019,
            "summary": "마치 가문의 네 자매가 성장하며 겪는 사랑과 삶을 그린 영화",
            "reason": "가족과의 유대와 따뜻한 이야기가 ISFJ의 감수성과 가족 중심적 성향에 잘 맞을 거예요."
        }
    ],
    "ESTP": [
        {
            "title": "Mad Max: Fury Road",
            "korean_title": "매드 맥스: 분노의 도로",
            "year": 2015,
            "summary": "포스트 아포칼립스 시대를 배경으로 한 고속 액션 영화",
            "reason": "스릴 넘치는 액션과 속도감이 ESTP의 모험심과 에너지를 충족시킬 거예요."
        },
        {
            "title": "John Wick",
            "korean_title": "존 윅",
            "year": 2014,
            "summary": "전설적인 킬러가 복수를 위해 다시 나서는 이야기를 그린 영화",
            "reason": "끊임없는 액션과 강렬한 스토리가 ESTP의 에너지와 흥분을 자극할 거예요."
        },
        {
            "title": "Fast & Furious 7",
            "korean_title": "분노의 질주: 더 세븐",
            "year": 2015,
            "summary": "고속 레이싱과 범죄를 다룬 액션 영화",
            "reason": "빠른 속도감과 아드레날린 넘치는 장면들이 ESTP의 스릴을 충족시킬 거예요."
        },
        {
            "title": "Die Hard",
            "korean_title": "다이 하드",
            "year": 1988,
            "summary": "테러리스트들에게 납치된 건물에서 홀로 싸우는 경찰관의 이야기",
            "reason": "긴박감 넘치는 액션과 영웅적 주인공이 ESTP의 모험심을 자극할 거예요."
        },
        {
            "title": "Casino Royale",
            "korean_title": "카지노 로얄",
            "year": 2006,
            "summary": "제임스 본드가 첩보 작전을 수행하며 펼쳐지는 이야기",
            "reason": "화려한 액션과 스파이 활동이 ESTP의 흥미와 스릴을 자극할 거예요."
        }
    ],
    "INFJ": [
        {
            "title": "Her",
            "korean_title": "그녀",
            "year": 2013,
            "summary": "인공지능 운영체제와 사랑에 빠진 남자의 이야기를 그린 영화",
            "reason": "심오한 철학적 질문과 감성적인 이야기가 INFJ의 내면적 탐구와 공감 능력에 잘 맞을 거예요."
        },
        {
            "title": "The Green Mile",
            "korean_title": "그린 마일",
            "year": 1999,
            "summary": "죽음의 선고를 받은 죄수와 감방 간수의 감동적인 이야기를 그린 영화",
            "reason": "인간의 본질과 정의를 탐구하는 이야기가 INFJ의 이상주의와 깊은 감성을 자극할 거예요."
        },
        {
            "title": "A Beautiful Mind",
            "korean_title": "뷰티풀 마인드",
            "year": 2001,
            "summary": "천재 수학자의 삶과 정신 질환을 그린 영화",
            "reason": "복잡한 인간 심리와 천재성에 대한 이야기가 INFJ의 지적 호기심과 공감을 자극할 거예요."
        },
        {
            "title": "Life of Pi",
            "korean_title": "라이프 오브 파이",
            "year": 2012,
            "summary": "해양 사고 후 한 소년과 호랑이의 생존 이야기를 그린 영화",
            "reason": "철학적이고 영적인 주제가 INFJ의 심오한 사고를 자극할 거예요."
        },
        {
            "title": "The Secret Garden",
            "korean_title": "비밀의 화원",
            "year": 1993,
            "summary": "비밀스러운 정원을 발견한 소녀의 이야기를 그린 영화",
            "reason": "치유와 발견의 이야기가 INFJ의 따뜻한 감성과 이상주의를 자극할 거예요."
        }
    ],
    "ESFP": [
        {
            "title": "La La Land",
            "korean_title": "라라랜드",
            "year": 2016,
            "summary": "꿈을 쫓는 두 예술가의 사랑과 도전을 그린 뮤지컬 영화",
            "reason": "화려한 음악과 춤, 감동적인 이야기가 ESFP의 사교적이고 활기찬 성격에 잘 맞을 거예요."
        },
        {
            "title": "Mamma Mia!",
            "korean_title": "맘마미아!",
            "year": 2008,
            "summary": "어머니의 과거를 찾기 위해 세 명의 남자를 초대하는 딸의 이야기",
            "reason": "신나는 음악과 유쾌한 스토리가 ESFP의 에너지를 충족시킬 거예요."
        },
        {
            "title": "The Greatest Showman",
            "korean_title": "위대한 쇼맨",
            "year": 2017,
            "summary": "서커스의 창립과 성공을 그린 뮤지컬 영화",
            "reason": "화려한 공연과 감동적인 스토리가 ESFP의 예술적 감각을 자극할 거예요."
        },
        {
            "title": "Pitch Perfect",
            "korean_title": "피치 퍼펙트",
            "year": 2012,
            "summary": "대학 아카펠라 그룹의 경쟁과 성장을 그린 영화",
            "reason": "음악과 유머, 단체 활동이 ESFP의 사교적 성향에 잘 맞을 거예요."
        },
        {
            "title": "Crazy Rich Asians",
            "korean_title": "크레이지 리치 아시안스",
            "year": 2018,
            "summary": "부유한 가족의 배경을 가진 남자와 그의 연인의 이야기를 그린 영화",
            "reason": "화려한 비주얼과 로맨틱 코미디가 ESFP의 즐거움을 충족시킬 거예요."
        }
    ],
    "ISTJ": [
        {
            "title": "The King's Speech",
            "korean_title": "킹스 스피치",
            "year": 2010,
            "summary": "말더듬이를 극복하고 영국 왕의 연설을 준비하는 과정을 그린 영화",
            "reason": "실화를 바탕으로 한 역사적 배경과 주인공의 노력과 인내가 ISTJ의 성실하고 책임감 있는 성향에 잘 맞을 거예요."
        },
        {
            "title": "Saving Private Ryan",
            "korean_title": "라이언 일병 구하기",
            "year": 1998,
            "summary": "제2차 세계대전 동안 실종된 병사를 구출하기 위한 임무를 수행하는 군인의 이야기",
            "reason": "역사적 사실과 헌신적인 군인의 이야기가 ISTJ의 전통적 가치관에 잘 맞을 거예요."
        },
        {
            "title": "Apollo 13",
            "korean_title": "아폴로 13",
            "year": 1995,
            "summary": "아폴로 13호의 달 착륙 실패와 그로 인한 생존을 다룬 실화 영화",
            "reason": "실제 사건을 바탕으로 한 문제 해결 과정이 ISTJ의 분석적 사고와 성실함에 잘 맞을 거예요."
        },
        {
            "title": "Schindler's List",
            "korean_title": "쉰들러 리스트",
            "year": 1993,
            "summary": "제2차 세계대전 동안 수많은 유대인을 구출한 독일 사업가 오스카 쉰들러의 이야기",
            "reason": "역사적 배경과 도덕적 용기가 ISTJ의 강한 윤리감과 책임감을 자극할 거예요."
        },
        {
            "title": "Gran Torino",
            "korean_title": "그랜 토리노",
            "year": 2008,
            "summary": "은퇴한 자동차 공장 노동자가 이웃과의 갈등을 극복하고 우정을 쌓아가는 이야기",
            "reason": "전통적 가치와 개인의 변화를 다룬 이야기가 ISTJ의 성향에 잘 맞을 거예요."
        }
    ],
    "ISFP": [
        {
            "title": "Into the Wild",
            "korean_title": "인투 더 와일드",
            "year": 2007,
            "summary": "현실을 떠나 자연 속에서 자유를 찾으려는 청년의 실화를 그린 영화",
            "reason": "자연과의 조화와 개인의 자유를 추구하는 이야기가 ISFP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Call Me by Your Name",
            "korean_title": "콜 미 바이 유어 네임",
            "year": 2017,
            "summary": "이탈리아를 배경으로 한 두 남자의 아름다운 사랑 이야기를 그린 영화",
            "reason": "감성적이고 섬세한 사랑 이야기가 ISFP의 감수성을 자극할 거예요."
        },
        {
            "title": "The Secret Life of Walter Mitty",
            "korean_title": "월터의 상상은 현실이 된다",
            "year": 2013,
            "summary": "평범한 사진 관리자가 꿈을 찾아 떠나는 모험을 그린 영화",
            "reason": "자아 발견과 모험이 가득한 이야기로 ISFP의 이상주의적 성향에 잘 맞을 거예요."
        },
        {
            "title": "Life of Pi",
            "korean_title": "라이프 오브 파이",
            "year": 2012,
            "summary": "해양 사고 후 한 소년과 호랑이의 생존 이야기를 그린 영화",
            "reason": "철학적이고 영적인 주제가 ISFP의 심오한 사고를 자극할 거예요."
        },
        {
            "title": "The Perks of Being a Wallflower",
            "korean_title": "월플라워",
            "year": 2012,
            "summary": "내성적인 청소년이 친구들을 통해 성장하는 이야기를 그린 영화",
            "reason": "개인적 성장과 우정을 다룬 이야기가 ISFP의 감수성과 공감을 자극할 거예요."
        }
    ],
    "ESTJ": [
        {
            "title": "The Dark Knight",
            "korean_title": "다크 나이트",
            "year": 2008,
            "summary": "배트맨과 그의 숙적 조커의 대결을 그린 영화",
            "reason": "강력한 정의감과 전략적 사고가 돋보이는 이야기로 ESTJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Gladiator",
            "korean_title": "글래디에이터",
            "year": 2000,
            "summary": "로마 제국 시대의 장군이 배신당하고 검투사가 되어 복수를 다짐하는 이야기",
            "reason": "영웅적 주인공과 강한 리더십이 ESTJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "12 Angry Men",
            "korean_title": "12명의 성난 사람들",
            "year": 1957,
            "summary": "살인 사건의 배심원들이 치열하게 논쟁하며 진실을 찾아가는 과정을 그린 영화",
            "reason": "논리적 사고와 정의를 추구하는 과정이 ESTJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "A Few Good Men",
            "korean_title": "어 퓨 굿 맨",
            "year": 1992,
            "summary": "군대 내의 살인 사건을 둘러싼 법정 드라마.",
            "reason": "도덕적 딜레마와 정의를 다룬 이야기가 ESTJ의 강한 윤리감에 잘 맞을 거예요."
        },
        {
            "title": "Apollo 13",
            "korean_title": "아폴로 13",
            "year": 1995,
            "summary": "아폴로 13호의 달 착륙 실패와 그로 인한 생존을 다룬 실화 영화",
            "reason": "실제 사건을 바탕으로 한 문제 해결 과정이 ESTJ의 분석적 사고와 성실함에 잘 맞을 거예요."
        }
    ],
    "ENFJ": [
        {
            "title": "Dead Poets Society",
            "korean_title": "죽은 시인의 사회",
            "year": 1989,
            "summary": "혁신적인 교사가 학생들에게 삶의 의미를 가르치는 이야기를 그린 영화",
            "reason": "이상적이고 영감을 주는 이야기가 ENFJ의 지도력과 열정을 자극할 거예요."
        },
        {
            "title": "The Pursuit of Happyness",
            "korean_title": "행복을 찾아서",
            "year": 2006,
            "summary": "어려운 상황에서도 아들을 위해 끊임없이 노력하는 아버지의 이야기를 그린 영화",
            "reason": "가족을 위한 헌신과 인내의 이야기가 ENFJ의 책임감과 가족 중심적 성향에 잘 맞을 거예요."
        },
        {
            "title": "Schindler's List",
            "korean_title": "쉰들러 리스트",
            "year": 1993,
            "summary": "제2차 세계대전 동안 수많은 유대인을 구출한 독일 사업가 오스카 쉰들러의 이야기",
            "reason": "역사적 배경과 도덕적 용기가 ENFJ의 강한 윤리감과 책임감을 자극할 거예요."
        },
        {
            "title": "Patch Adams",
            "korean_title": "패치 아담스",
            "year": 1998,
            "summary": "독특한 방식으로 환자들을 치료하는 의사의 실화를 그린 영화",
            "reason": "공감과 유머를 통해 사람들을 돕는 이야기가 ENFJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Erin Brockovich",
            "korean_title": "에린 브로코비치",
            "year": 2000,
            "summary": "한 여성의 끈질긴 노력으로 대기업을 상대로 한 법적 싸움에서 승리하는 이야기를 그린 영화",
            "reason": "정의를 위해 싸우는 주인공의 이야기가 ENFJ의 열정과 헌신을 자극할 거예요."
        }
    ],
    "ENTJ": [
        {
            "title": "The Wolf of Wall Street",
            "korean_title": "더 울프 오브 월 스트리트",
            "year": 2013,
            "summary": "야망 넘치는 증권 브로커의 성공과 몰락을 그린 영화",
            "reason": "야망과 성공을 중심으로 한 이야기가 ENTJ의 성향에 잘 맞는 자극적인 이야기랍니다."
        },
        {
            "title": "Gladiator",
            "korean_title": "글래디에이터",
            "year": 2000,
            "summary": "로마 제국 시대의 장군이 배신당하고 검투사가 되어 복수를 다짐하는 이야기",
            "reason": "영웅적 주인공과 강한 리더십이 ENTJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "The Social Network",
            "korean_title": "소셜 네트워크",
            "year": 2010,
            "summary": "페이스북의 창립과정과 그로 인한 갈등을 그린 영화",
            "reason": "기업가 정신과 리더십을 다룬 이야기가 ENTJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Moneyball",
            "korean_title": "머니볼",
            "year": 2011,
            "summary": "통계와 분석을 통해 야구 팀을 혁신하는 과정을 그린 영화",
            "reason": "전략적 사고와 혁신을 다룬 이야기가 ENTJ의 분석적 사고와 도전 정신에 잘 맞을 거예요."
        },
        {
            "title": "The Godfather",
            "korean_title": "대부",
            "year": 1972,
            "summary": "마피아 가문의 리더십과 권력 투쟁을 그린 영화",
            "reason": "강한 리더십과 전략적 사고가 돋보이는 이야기가 ENTJ의 성향에 잘 맞을 거예요."
        }
    ],
    "INTJ": [
        {
            "title": "Inception",
            "korean_title": "인셉션",
            "year": 2010,
            "summary": "꿈 속에서 아이디어를 훔치는 도둑이 마지막 임무를 수행하는 이야기",
            "reason": "복잡한 스토리와 전략적 사고가 돋보이는 이야기가 INTJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "The Matrix",
            "korean_title": "매트릭스",
            "year": 1999,
            "summary": "가상 현실 세계에서 인간의 반란을 그린 영화",
            "reason": "철학적 질문과 혁신적 아이디어가 INTJ의 사고를 자극할 거예요."
        },
        {
            "title": "Interstellar",
            "korean_title": "인터스텔라",
            "year": 2014,
            "summary": "인류의 생존을 위해 우주를 탐험하는 이야기를 그린 영화",
            "reason": "과학적 탐구와 미래지향적 이야기가 INTJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "The Imitation Game",
            "korean_title": "이미테이션 게임",
            "year": 2014,
            "summary": "2차 세계대전 동안 독일의 암호를 해독한 천재 수학자의 이야기",
            "reason": "논리적 사고와 문제 해결 과정을 그린 이야기가 INTJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Fight Club",
            "korean_title": "파이트 클럽",
            "year": 1999,
            "summary": "현대 사회에 대한 반항과 자기 발견을 그린 영화",
            "reason": "사회적 규범에 대한 도전과 철학적 질문이 INTJ의 사고를 자극할 거예요."
        }
    ],
    "ENTP": [
        {
            "title": "The Big Short",
            "korean_title": "빅쇼트",
            "year": 2015,
            "summary": "2008년 금융 위기를 예측하고 이익을 얻은 사람들의 이야기를 그린 영화",
            "reason": "혁신적 사고와 복잡한 경제 문제를 다룬 이야기가 ENTP의 지적 호기심을 자극할 거예요."
        },
        {
            "title": "The Social Network",
            "korean_title": "소셜 네트워크",
            "year": 2010,
            "summary": "페이스북의 창립과정과 그로 인한 갈등을 그린 영화",
            "reason": "혁신적이고 논쟁적인 주제가 ENTP의 토론과 아이디어에 대한 열정을 자극할 거예요."
        },
        {
            "title": "Inception",
            "korean_title": "인셉션",
            "year": 2010,
            "summary": "꿈 속에서 아이디어를 훔치는 도둑이 마지막 임무를 수행하는 이야기",
            "reason": "복잡한 스토리와 창의적인 설정이 ENTP의 사고를 자극할 거예요."
        },
        {
            "title": "Fight Club",
            "korean_title": "파이트 클럽",
            "year": 1999,
            "summary": "현대 사회에 대한 반항과 자기 발견을 그린 영화",
            "reason": "사회적 규범에 대한 도전과 철학적 질문이 ENTP의 사고를 자극할 거예요."
        },
        {
            "title": "The Wolf of Wall Street",
            "korean_title": "더 울프 오브 월 스트리트",
            "year": 2013,
            "summary": "야망 넘치는 증권 브로커의 성공과 몰락을 그린 영화",
            "reason": "도전과 기회를 추구하는 ENTP의 성향에 잘 맞는 자극적인 이야기랍니다."
        }
    ],
    "INTP": [
        {
            "title": "The Theory of Everything",
            "korean_title": "사랑에 대한 모든 것",
            "year": 2014,
            "summary": "스티븐 호킹의 삶과 업적을 그린 영화",
            "reason": "과학적 호기심과 지적 탐구가 돋보이는 이야기가 INTP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "A Beautiful Mind",
            "korean_title": "뷰티풀 마인드",
            "year": 2001,
            "summary": "천재 수학자의 삶과 정신 질환을 그린 영화",
            "reason": "복잡한 인간 심리와 천재성에 대한 이야기가 INTP의 지적 호기심을 자극할 거예요."
        },
        {
            "title": "Ex Machina",
            "korean_title": "엑스 마키나",
            "year": 2015,
            "summary": "인공지능 로봇과 인간의 관계를 탐구하는 이야기",
            "reason": "철학적 질문과 기술적 탐구가 INTP의 사고를 자극할 거예요."
        },
        {
            "title": "The Imitation Game",
            "korean_title": "이미테이션 게임",
            "year": 2014,
            "summary": "2차 세계대전 동안 독일의 암호를 해독한 천재 수학자의 이야기",
            "reason": "논리적 사고와 문제 해결 과정을 그린 이야기가 INTP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Blade Runner 2049",
            "korean_title": "블레이드 러너 2049",
            "year": 2017,
            "summary": "미래의 인간과 인공지능의 관계를 그린 SF 영화",
            "reason": "철학적이고 심오한 주제가 INTP의 사고를 자극할 거예요."
        }
    ],
    "ISTP": [
        {
            "title": "Mad Max: Fury Road",
            "korean_title": "매드 맥스: 분노의 도로",
            "year": 2015,
            "summary": "포스트 아포칼립스 세계에서 생존을 위해 싸우는 이야기를 그린 영화",
            "reason": "액션과 모험을 즐기는 ISTP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "John Wick",
            "korean_title": "존 윅",
            "year": 2014,
            "summary": "전설적인 킬러가 복수를 위해 다시 나서는 이야기를 그린 영화",
            "reason": "전문적인 기술과 액션을 중시하는 ISTP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "The Bourne Identity",
            "korean_title": "본 아이덴티티",
            "year": 2002,
            "summary": "기억을 잃은 첩보원이 자신의 정체를 찾아가는 이야기를 그린 영화",
            "reason": "스릴과 미스터리를 즐기는 ISTP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Die Hard",
            "korean_title": "다이 하드",
            "year": 1988,
            "summary": "테러리스트들에게 납치된 빌딩에서 홀로 싸우는 경찰의 이야기를 그린 영화",
            "reason": "문제를 해결하는 과정에서의 액션과 독립심이 ISTP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Casino Royale",
            "korean_title": "007 카지노 로얄",
            "year": 2006,
            "summary": "첩보원 제임스 본드가 카지노에서 테러 자금을 추적하는 이야기를 그린 영화",
            "reason": "스릴 넘치는 액션과 전략적 사고가 ISTP의 성향에 잘 맞을 거예요."
        }
    ],
    "ENFP": [
        {
            "title": "The Secret Life of Walter Mitty",
            "korean_title": "월터의 상상은 현실이 된다",
            "year": 2013,
            "summary": "평범한 직장인 월터가 모험을 통해 자신을 발견하는 이야기",
            "reason": "상상력과 모험을 중시하는 ENFP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Amélie",
            "korean_title": "아멜리에",
            "year": 2001,
            "summary": "파리에서 독특한 삶을 사는 아멜리의 이야기를 그린 영화",
            "reason": "독창적이고 따뜻한 스토리가 ENFP의 감성에 잘 맞을 거예요."
        },
        {
            "title": "Inside Out",
            "korean_title": "인사이드 아웃",
            "year": 2015,
            "summary": "소녀의 머릿속에서 벌어지는 감정들의 이야기를 그린 애니메이션 영화",
            "reason": "감정과 상상력이 풍부한 이야기가 ENFP의 성향에 잘 맞을 거예요."
        },
        {
            "title": "The Perks of Being a Wallflower",
            "korean_title": "월플라워",
            "year": 2012,
            "summary": "내성적인 고등학생이 친구들을 만나 성장하는 이야기",
            "reason": "청춘과 성장의 이야기가 ENFP의 감성을 자극할 거예요."
        },
        {
            "title": "La La Land",
            "korean_title": "라라랜드",
            "year": 2016,
            "summary": "배우와 음악가가 꿈을 쫓아가는 과정을 그린 뮤지컬 영화",
            "reason": "예술과 사랑에 대한 이야기가 ENFP의 성향에 잘 맞을 거예요."
        }
    ],
    "ESFJ": [
        {
            "title": "Forrest Gump",
            "korean_title": "포레스트 검프",
            "year": 1994,
            "summary": "지적 장애를 가진 남자가 파란만장한 인생을 살아가는 이야기",
            "reason": "따뜻하고 감동적인 스토리가 ESFJ의 감성을 자극할 거예요."
        },
        {
            "title": "The Help",
            "korean_title": "헬프",
            "year": 2011,
            "summary": "1960년대 미국 남부에서 흑인 가정부와 백인 여성이 함께 변화를 만들어가는 이야기",
            "reason": "공감과 사회적 정의를 중시하는 ESFJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Mamma Mia!",
            "korean_title": "맘마미아!",
            "year": 2008,
            "summary": "딸의 결혼식을 앞두고 벌어지는 엄마와 딸의 이야기를 그린 뮤지컬 영화",
            "reason": "가족과 사랑을 중심으로 한 이야기가 ESFJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "The Blind Side",
            "korean_title": "블라인드 사이드",
            "year": 2009,
            "summary": "홈리스 소년이 한 가족의 도움으로 미식축구 선수가 되는 이야기",
            "reason": "따뜻한 가족애와 헌신을 그린 이야기가 ESFJ의 성향에 잘 맞을 거예요."
        },
        {
            "title": "Julie & Julia",
            "korean_title": "줄리 & 줄리아",
            "year": 2009,
            "summary": "줄리아 차일드의 요리책을 따라 요리 블로그를 시작한 여성의 이야기",
            "reason": "요리와 일상 속의 기쁨을 다룬 이야기가 ESFJ의 성향에 잘 맞을 거예요."
        }
    ]
}

# MBTI 유형별 영화 추천: 유형에 따라 하나만 랜덤으로.
def recommend_movie(mbti_type):

    movie = random.choice(mbti_movies[mbti_type])

    data1 = f"{movie['korean_title']}({movie['title']}, {movie['year']})"
    data2 = f"{movie['summary']}로,\n{movie['reason']}"

    return data1, data2
