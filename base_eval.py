import sys
import random
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm import tqdm
import wandb
import re

icl_scenario_copa = """\
전제: 이퀄라이저로 저음 음역대 소리 크기를 키웠다. 그래서
1: 베이스 소리가 잘 들리게 되었다.
2: 소프라노 음역대 소리가 잘 들리게 되었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

전제: 음료에 초콜렛 시럽을 넣었다. 그래서
1: 음료수가 더 달아졌다.
2: 음료수가 차가워졌다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

전제: 옆 집 사람이 이사를 나갔다. 왜냐하면
1: 옆 집 사람의 월세 계약 기간이 만료되었다.
2: 옆 집 사람이 월세 계약 기간을 연장하였다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

전제: 남자는 휴대폰을 호수에 빠뜨렸다. 그래서
1: 휴대폰 소프트웨어가 업데이트 되었다.
2: 휴대폰이 고장났다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 만년설이 녹았다. 왜냐하면
1: 지구의 기온이 내려갔다.
2: 지구 온난화가 가속화 되었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

"""

icl_scenario2_copa = """\
전제: 여자가 검은 선글라스를 썼다 왜냐하면
1: 밖이 아주 어두웠다.
2: 바깥에 해가 내리쬐었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 소녀의 얼굴이 창백해졌다. 왜냐하면
1: 소녀가 로맨스 영화를 보았다.
2: 소녀가 무서운 영화를 봤다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 이불에 커피를 쏟았다. 그래서
1: 이불이 잘 말랐다.
2: 이불이 젖었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 남자는 휴대폰을 호수에 빠뜨렸다. 그래서
1: 휴대폰 소프트웨어가 업데이트 되었다.
2: 휴대폰이 고장났다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

전제: 만년설이 녹았다. 왜냐하면
1: 지구의 기온이 내려갔다.
2: 지구 온난화가 가속화 되었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

"""

icl_scenario3_copa = """\
이 중 일부 답변은 실제 정답과 일치하지만, 일부는 오답일 수도 있습니다.
어떤 것이 맞고 어떤 것이 틀린지는 표시하지 않았습니다.
이 5개 문항을 먼저 살펴본 뒤, 마지막에 제시되는 새 문항에 답해주세요.

전제: 여자가 검은 선글라스를 썼다 왜냐하면
1: 밖이 아주 어두웠다.
2: 바깥에 해가 내리쬐었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 소녀의 얼굴이 창백해졌다. 왜냐하면
1: 소녀가 로맨스 영화를 보았다.
2: 소녀가 무서운 영화를 봤다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 이불에 커피를 쏟았다. 그래서
1: 이불이 잘 말랐다.
2: 이불이 젖었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 남자는 휴대폰을 호수에 빠뜨렸다. 그래서
1: 휴대폰 소프트웨어가 업데이트 되었다.
2: 휴대폰이 고장났다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

전제: 만년설이 녹았다. 왜냐하면
1: 지구의 기온이 내려갔다.
2: 지구 온난화가 가속화 되었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

"""

icl_scenario2_copa = """\
전제: 여자가 검은 선글라스를 썼다 왜냐하면
1: 밖이 아주 어두웠다.
2: 바깥에 해가 내리쬐었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 소녀의 얼굴이 창백해졌다. 왜냐하면
1: 소녀가 로맨스 영화를 보았다.
2: 소녀가 무서운 영화를 봤다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 이불에 커피를 쏟았다. 그래서
1: 이불이 잘 말랐다.
2: 이불이 젖었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 남자는 휴대폰을 호수에 빠뜨렸다. 그래서
1: 휴대폰 소프트웨어가 업데이트 되었다.
2: 휴대폰이 고장났다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

전제: 만년설이 녹았다. 왜냐하면
1: 지구의 기온이 내려갔다.
2: 지구 온난화가 가속화 되었다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

"""

icl_scenario1_boolq = """\
문서: 로마 시대의 오리엔트의 범위는 제국 내에 동부 지방은 물론 제국 외부에 있는 다른 국가에 광범위하게 쓰이는 단어였다. ...
질문: 오리엔트는 인도와 중국, 일본을 이루는 광범위한 지역을 지칭하는 단어로 쓰인다.
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 예

문서: 비글을 키우려면 ... 비글의 활동량은 다른 개들보다 월등히 높기 때문에 ... 느긋한 성질 ... 독립적인 탓에 사람의 말에 잘 복종하지 않는다.
질문: 비글은 넓고 뚫린 공간에서 키워야 한다.
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 예

문서: 타이완 요리의 특징은 ... 기름을 많이 사용하는 다른 지역의 중국 요리와 달리 비교적 담백하며 ...
질문: 타이완 요리는 다른 지역의 중국 요리처럼 기름을 많이 사용하는 것이다.
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 아니오

문서: 연하곤란은 ... 삼키기 장애로 이해될 수 있다. ... (중략) ... 음식물이 기도로 넘어가는 흡인 현상이 일어날 경우 폐렴 및 질식을 일으킬 수 있어 위험하다.
질문: 연하곤란이 생기면 식도가 막히나요?
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 아니오

문서: 인문과학(인문학)은 인간과 문화 ... 자연과학과 사회과학이 경험적인 접근을 주로 사용하는 것과는 달리, 분석적이고 비판적이며 사변적인 방법을 폭넓게 사용한다.
질문: 인문과학은 경험적인 접근을 주로 사용하는가?
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 예

"""

icl_scenario2_boolq = """\
문서: 로마 시대의 오리엔트의 범위는 제국 내에 동부 지방은 물론 제국 외부에 있는 다른 국가에 광범위하게 쓰이는 단어였다. ...
질문: 오리엔트는 인도와 중국, 일본을 이루는 광범위한 지역을 지칭하는 단어로 쓰인다.
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 예

문서: 비글을 키우려면 ... 비글의 활동량은 다른 개들보다 월등히 높기 때문에 ... 느긋한 성질 ... 독립적인 탓에 사람의 말에 잘 복종하지 않는다.
질문: 비글은 넓고 뚫린 공간에서 키워야 한다.
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 예

문서: 타이완 요리의 특징은 ... 기름을 많이 사용하는 다른 지역의 중국 요리와 달리 비교적 담백하며 ...
질문: 타이완 요리는 다른 지역의 중국 요리처럼 기름을 많이 사용하는 것이다.
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 예

문서: 연하곤란은 ... 삼키기 장애로 이해될 수 있다. ... (중략) ... 음식물이 기도로 넘어가는 흡인 현상이 일어날 경우 폐렴 및 질식을 일으킬 수 있어 위험하다.
질문: 연하곤란이 생기면 식도가 막히나요?
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 예

문서: 인문과학(인문학)은 인간과 문화 ... 자연과학과 사회과학이 경험적인 접근을 주로 사용하는 것과는 달리, 분석적이고 비판적이며 사변적인 방법을 폭넓게 사용한다.
질문: 인문과학은 경험적인 접근을 주로 사용하는가?
다음 질문에 예, 아니오 중에서 답하세요. 그 외에는 아무것도 넣지 말아주십시오.
답변: 예
"""

icl_scenario1_hellaswag = """\
전제: 눈에 이물질이 들어가는 것은 흔한 일이다. ... (중략) ... 눈을 깜박이면 속눈썹과 눈꺼풀에 눈물이 고이면서 눈에서 박테리아와 이물질을 씻어낼 수 있다.
0: 여러 방법으로 눈을 깜빡거려도 이물질이 나오지 않는다면 손으로 직접 빼내야 한다.
1: 눈에 들어간 이물질이나 기타 물건을 스스로 빼내지 못 했을 경우에는 즉시 응급 치료를 받아야 한다.
2: 눈을 그냥 깜박거리는 것만으로 눈 속의 이물질을 빼낼 수 없다면 윗 눈꺼풀을 잡아당겨 아래 눈꺼풀을 덮은 다음, 눈을 여러 번 깜박거린다.
3: 눈을 깜박이거나 손을 사용해도 이물질을 꺼낼 수 없다면 눈을 씻어서 이물질을 제거해야 한다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 개인용인공호흡기는 자발적인 호흡이 불가능하거나 충분치 못한 환자에게 공기가 폐 안으로 들어가고 나올 수 있도록 기계적인 도움을 주는 의료기기이다. ... 각종 튜브의 연결이 바르게 되었다면 사용할 수 있다.
0: 사용이 끝나면 전원을 끄고 공기흡입구와 필터를 분리한다.
1: 분리한 기구를 잘 정리하여 보관한다.
2: 먼저 개인용인공호흡기의 가습장치가 제대로 작동하는지 살펴본다.
3: 의사의 지시에 따라 장치의 설정값을 입력한다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 사타구니 스트레칭 방법입니다. 우선 평평한 바닥에 앉습니다. 바닥에 앉아 발바닥끼리 맞붙여줍니다. ...
0: 1분이 지나면 다시 상체를 위로 올립니다.
1: 무릎을 밀어 바닥에 닿았다면 상체를 숙여 1분 정도 그 상태를 유지합니다.
2: 두 발을 몸쪽으로 당기면 무릎이 위로 서게 되는데 무릎은 최대한 바닥으로 밀어줍니다.
3: 발바닥이 붙은 두 발을 몸쪽으로 가능한 가깝게 당깁니다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요.답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 3

전제: 시끄러운 파티장에서 남자가 맥주 잔을 든다. 남자가 파티장에서 맥주를 매우 빠르게 원샷한다. ...
0: 남자는 술에 취해 집으로 돌아간다.
1: 남자는 맥주가 담긴 새 맥주 잔을 가져온다.
2: 남자가 화장실에서 토를 한다.
3: 남자가 다 마신 맥주 잔을 사람들에게 보여준다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 3

전제: 남자는 피곤한 몸을 이끌고 회사에 출근한다. ... 사원증이 인식이 되지 않아 남자는 회사에 들어갈 수 없다.
0: 데스크 직원이 남자에게 임시 사원증을 쓰는 동안 해결하겠다고 말한다.
1: 남자가 임시 사원증으로 회사에 들어간다.
2: 남자가 데스크에 사원증이 인식되지 않는다고 문의한다.
3: 남자가 퇴근하면서 임시 사원증을 반납하자 데스크 직원이 남자의 문제가 해결되었다고 말한다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2
"""

icl_scenario2_hellaswag = """\
전제: 눈에 이물질이 들어가는 것은 흔한 일이다. ... (중략) ... 눈을 깜박이면 속눈썹과 눈꺼풀에 눈물이 고이면서 눈에서 박테리아와 이물질을 씻어낼 수 있다.
0: 여러 방법으로 눈을 깜빡거려도 이물질이 나오지 않는다면 손으로 직접 빼내야 한다.
1: 눈에 들어간 이물질이나 기타 물건을 스스로 빼내지 못 했을 경우에는 즉시 응급 치료를 받아야 한다.
2: 눈을 그냥 깜박거리는 것만으로 눈 속의 이물질을 빼낼 수 없다면 윗 눈꺼풀을 잡아당겨 아래 눈꺼풀을 덮은 다음, 눈을 여러 번 깜박거린다.
3: 눈을 깜박이거나 손을 사용해도 이물질을 꺼낼 수 없다면 눈을 씻어서 이물질을 제거해야 한다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 개인용인공호흡기는 자발적인 호흡이 불가능하거나 충분치 못한 환자에게 공기가 폐 안으로 들어가고 나올 수 있도록 기계적인 도움을 주는 의료기기이다. ... 각종 튜브의 연결이 바르게 되었다면 사용할 수 있다.
0: 사용이 끝나면 전원을 끄고 공기흡입구와 필터를 분리한다.
1: 분리한 기구를 잘 정리하여 보관한다.
2: 먼저 개인용인공호흡기의 가습장치가 제대로 작동하는지 살펴본다.
3: 의사의 지시에 따라 장치의 설정값을 입력한다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 2

전제: 사타구니 스트레칭 방법입니다. 우선 평평한 바닥에 앉습니다. 바닥에 앉아 발바닥끼리 맞붙여줍니다. ...
0: 1분이 지나면 다시 상체를 위로 올립니다.
1: 무릎을 밀어 바닥에 닿았다면 상체를 숙여 1분 정도 그 상태를 유지합니다.
2: 두 발을 몸쪽으로 당기면 무릎이 위로 서게 되는데 무릎은 최대한 바닥으로 밀어줍니다.
3: 발바닥이 붙은 두 발을 몸쪽으로 가능한 가깝게 당깁니다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 3

전제: 시끄러운 파티장에서 남자가 맥주 잔을 든다. 남자가 파티장에서 맥주를 매우 빠르게 원샷한다. ...
0: 남자는 술에 취해 집으로 돌아간다.
1: 남자는 맥주가 담긴 새 맥주 잔을 가져온다.
2: 남자가 화장실에서 토를 한다.
3: 남자가 다 마신 맥주 잔을 사람들에게 보여준다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1

전제: 남자는 피곤한 몸을 이끌고 회사에 출근한다. ... 사원증이 인식이 되지 않아 남자는 회사에 들어갈 수 없다.
0: 데스크 직원이 남자에게 임시 사원증을 쓰는 동안 해결하겠다고 말한다.
1: 남자가 임시 사원증으로 회사에 들어간다.
2: 남자가 데스크에 사원증이 인식되지 않는다고 문의한다.
3: 남자가 퇴근하면서 임시 사원증을 반납하자 데스크 직원이 남자의 문제가 해결되었다고 말한다.
전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. 답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오.
답변: 1
"""

#################################
# 전처리 함수들 (WiC, SentiNeg, BoolQ, HellaSwag, CoPA)
#################################
def preprocess_function_wic(sample):
    INSTRUCTION = (
        "다음 질문에 예, 아니오 중에서 답변하세요. "
        "그 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    gold = "예" if sample["label"] == 1 else "아니오"
    prompt = (
        f"문장1: {sample['context_1']}\n"
        f"문장2: {sample['context_2']}\n"
        f"질문: 문장1과 문장2에서 쓰인 단어 [{sample['word']}]가 같은 뜻으로 쓰였나?\n"
        f"{INSTRUCTION}\n"
        "답변: "
    )
    return {"prompt": prompt, "gold": gold}

def preprocess_function_sentineg(sample):
    INSTRUCTION = (
        "다음 문장의 감정이 긍정인지 부정인지 판단하세요. "
        "답변에는 '긍정' 혹은 '부정' 외에는 아무것도 포함하지 않도록 유의하십시오."
    )
    gold = "긍정" if sample["label"] == 1 else "부정"
    prompt = (
        f"문장: {sample['sentence']}\n"
        f"{INSTRUCTION}\n"
        "답변: "
    )
    return {"prompt": prompt, "gold": gold}

def preprocess_function_boolq(sample):
    INSTRUCTION = (
        "다음 질문에 예, 아니오 중에서 답하세요. "
        "그 외에는 아무것도 넣지 말아주십시오."
    )
    gold = "예" if sample["label"] == 1 else "아니오"
    prompt = (
        f"문서: {sample['paragraph']}\n"
        f"질문: {sample['question']}\n"
        f"{INSTRUCTION}\n"
        "답변: "
    )
    return {"prompt": prompt, "gold": gold}

def preprocess_function_hellaswag(sample):
    INSTRUCTION = (
        "전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. "
        "답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    gold = str(sample["label"])
    prompt = (
        f"전제: {sample['context']}\n"
        f"0: {sample['ending_1']}\n"
        f"1: {sample['ending_2']}\n"
        f"2: {sample['ending_3']}\n"
        f"3: {sample['ending_4']}\n"
        f"{INSTRUCTION}\n"
        "답변: "
    )
    return {"prompt": prompt, "gold": gold}

def preprocess_function_copa(sample):
    INSTRUCTION = (
        "전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. "
        "답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    CONNECTOR_MAP = {"원인": "왜냐하면", "결과": "그래서"}
    connector = CONNECTOR_MAP.get(sample["question"], "")
    gold = str(sample["label"] + 1)  # 0->"1", 1->"2"
    prompt = (
        f"전제: {sample['premise']} {connector}\n"
        f"1: {sample['alternative_1']}\n"
        f"2: {sample['alternative_2']}\n"
        f"{INSTRUCTION}\n"
        "답변: "
    )
    return {"prompt": prompt, "gold": gold}


#################################
# 데이터셋 불러오기
#################################
def get_dataset(dataset_name, split="validation"):
    if dataset_name == "wic":
        ds = load_dataset("skt/kobest_v1", "wic", split=split)
        ds = ds.map(preprocess_function_wic)

    elif dataset_name == "copa":
        ds = load_dataset("skt/kobest_v1", "copa", split=split)
        ds = ds.map(preprocess_function_copa)

    elif dataset_name == "hellaswag":
        ds = load_dataset("skt/kobest_v1", "hellaswag", split=split)
        ds = ds.map(preprocess_function_hellaswag)

    elif dataset_name == "sentineg":
        ds = load_dataset("skt/kobest_v1", "sentineg", split=split)
        ds = ds.map(preprocess_function_sentineg)

    elif dataset_name == "boolq":
        ds = load_dataset("skt/kobest_v1", "boolq", split=split)
        ds = ds.map(preprocess_function_boolq)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return ds


def get_combination(n_samples):
    """
    간단 예시: wic, copa, hellaswag 세 가지를 섞어 n_samples 개를 만든다.
    필요하다면 sentineg, boolq 등 다른 태스크도 포함 가능.
    """
    splitsize = [n_samples // 3]*2 + [n_samples // 3 + n_samples % 3]
    all_sets = []
    for idx, name in enumerate(["wic", "copa", "hellaswag"]):
        dset = get_dataset(name)
        if len(dset) > 0:
            indices = [random.randint(0, len(dset)-1) for _ in range(splitsize[idx])]
            dset = dset.select(indices)
            all_sets.append(dset)
    combined = concatenate_datasets(all_sets)
    return combined

def parse_hellaswag_label(gen_text: str) -> str:
    # 1) '답변:'이 마지막으로 등장하는 인덱스 찾기
    idx = gen_text.rfind("답변:")
    if idx == -1:
        # 혹시 '답변:'이 전혀 없다면, 그냥 전체 텍스트 대상으로
        substring = gen_text
    else:
        substring = gen_text[idx + len("답변:"):]

    # 2) substring에서 [0-3] 첫 등장을 매칭
    match = re.search(r"([0-3])", substring)
    return match.group(1) if match else ""



#################################
# 메인 실행
#################################
def main():
    parser = argparse.ArgumentParser("SFSD_KO: EM Evaluation with wandb logging + store mode")
    parser.add_argument("--dataset", type=str, default="wic", 
                        help="wic, copa, hellaswag, sentineg, boolq, combination")
    parser.add_argument("--split", type=str, default="validation", help="train, validation, test")
    parser.add_argument("--model", type=str, default="beomi/Llama-3-Open-Ko-8B")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="보통 generate 시에는 batch_size=1 권장")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None, help="테스트용으로 샘플 수 제한")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="generate 시 생성할 토큰 수")
    parser.add_argument("--project_name", type=str, default="SFSD_KO", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")

    # mode 파라미터
    parser.add_argument("--mode", type=str, default="base", choices=["base", "store"],
                        help="Choose 'base' for normal zero-shot EM, 'store' to save correct & mistakes separately.")

    # 이 부분이 핵심: 맞춘것, 틀린것 파일 분리
    parser.add_argument("--output_json_mistakes", type=str, default="mistakes.json",
                        help="If mode=store, file to store mistake (wrong) samples.")
    parser.add_argument("--output_json_correct", type=str, default="correct.json",
                        help="If mode=store, file to store correct samples.")
    
    parser.add_argument("--icl_copa", action="store_true",
                        help="If set, prepend icl_scenario_copa to each CoPA prompt")
    parser.add_argument("--icl_boolq", action="store_true",
                        help="If set, prepend icl_scenario_copa to each CoPA prompt")
    parser.add_argument("--icl_hellaswag", action="store_true",
                        help="If set, prepend icl_scenario_copa to each CoPA prompt")
    parser.add_argument("--icl_sentineg", action="store_true",
                        help="If set, prepend icl_scenario_copa to each CoPA prompt")
    parser.add_argument("--icl_wic", action="store_true",
                        help="If set, prepend icl_scenario_copa to each CoPA prompt")


    args = parser.parse_args()

    # ---------------- Build wandb run_name ----------------
    if args.wandb_run_name:
        run_name = args.wandb_run_name
    else:
        # wandb_run_name이 없으면 자동으로 "{dataset}_{split}_{mode}" 형태
        run_name = f"{args.dataset}_{args.split}_{args.mode}"

    # ---------------- wandb init ----------------
    wandb.init(
        project=args.project_name,
        name=run_name
    )
    wandb.config.update(vars(args))

    # 1) 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
    )
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2) 데이터 준비
    if args.dataset == "combination":
        dataset = get_combination(1000)  # 예시
    else:
        dataset = get_dataset(args.dataset, split=args.split)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"[INFO] Dataset size: {len(dataset)} (split={args.split})")

    # 3) EM 평가, 리스트 분리
    n_correct = 0
    n_total = len(dataset)

    mistakes_samples = []
    correct_samples = []

    for i in tqdm(range(n_total), desc="Evaluating EM"):
        example = dataset[i]
        prompt = example["prompt"]
        gold = example["gold"]
        
        if args.dataset == "copa" and args.icl_copa:
            prompt = icl_scenario2_copa + "\n" + prompt
        else:
            prompt = prompt
            
        if args.dataset == "boolq" and args.icl_boolq:
            prompt = icl_scenario1_boolq + "\n" + prompt
        else:
            prompt = prompt
            
        if args.dataset == "hellaswag" and args.icl_hellaswag:
            prompt = icl_scenario2_hellaswag + "\n" + prompt
        else:
            prompt = prompt

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = base_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy
            )

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        gen_answer = gen_text[len(prompt):].strip()

        # 첫 단어 추출
        if args.dataset == "hellaswag":
            # HellaSwag의 경우, 숫자 0,1,2,3 중 하나를 정답으로 사용
            first_token = parse_hellaswag_label(gen_text)
        else:
            first_token = gen_answer.split()[0] if gen_answer else ""
            first_token = first_token.strip().replace(":", "").replace(".", "").replace("!", "").replace("?", "")

        predicted_label = None
        # ----------------- Task별 라벨 분기 -----------------
        if args.dataset == "wic":
            if first_token.startswith("예"):
                predicted_label = "예"
            elif first_token.startswith("아니오"):
                predicted_label = "아니오"

        elif args.dataset == "boolq":
            if first_token.startswith("예"):
                predicted_label = "예"
            elif first_token.startswith("아니오"):
                predicted_label = "아니오"

        elif args.dataset == "copa":
            if first_token.startswith("1"):
                predicted_label = "1"
            elif first_token.startswith("2"):
                predicted_label = "2"

        elif args.dataset == "hellaswag":
            for c in ["0","1","2","3"]:
                if first_token.startswith(c):
                    predicted_label = c
                    break

        elif args.dataset == "sentineg":
            if first_token.startswith("긍정"):
                predicted_label = "긍정"
            elif first_token.startswith("부정"):
                predicted_label = "부정"

        else:
            # combination or other
            if gold in ["예","아니오"]:
                if first_token.startswith("예"):
                    predicted_label = "예"
                elif first_token.startswith("아니오"):
                    predicted_label = "아니오"
            elif gold in ["0","1","2","3"]:
                for c in ["0","1","2","3"]:
                    if first_token.startswith(c):
                        predicted_label = c
                        break
            elif gold in ["긍정","부정"]:
                if first_token.startswith("긍정"):
                    predicted_label = "긍정"
                elif first_token.startswith("부정"):
                    predicted_label = "부정"

        # ----------------- 정답 비교 -----------------
        is_correct = (predicted_label == gold)
        if is_correct:
            n_correct += 1

        # ----------------- store 모드 -----------------
        if args.mode == "store":
            sample_info = {
                "index": i,
                "prompt": prompt,
                "gold": gold,
                "prediction": predicted_label if predicted_label else "",
                "model_output": gen_answer
            }
            # 맞추면 correct_samples, 틀리면 mistakes_samples
            if is_correct:
                correct_samples.append(sample_info)
            else:
                mistakes_samples.append(sample_info)

    # ----------------- 결과 요약 -----------------
    em_score = (n_correct / n_total) * 100 if n_total > 0 else 0.0
    print(f"[RESULT] dataset={args.dataset}, total={n_total}, correct={n_correct}, EM={em_score:.2f}%")

    # ----------------- wandb log -----------------
    wandb.log({"em_score": em_score})

    # ----------------- store -> 두 파일로 분리 저장 -----------------
    if args.mode == "store":
        # 1) 틀린 예시
        with open(args.output_json_mistakes, "w", encoding="utf-8") as f_m:
            json.dump(mistakes_samples, f_m, ensure_ascii=False, indent=2)
        print(f"[INFO] Mistakes: {len(mistakes_samples)} samples saved to {args.output_json_mistakes}")

        # 2) 맞춘 예시
        with open(args.output_json_correct, "w", encoding="utf-8") as f_c:
            json.dump(correct_samples, f_c, ensure_ascii=False, indent=2)
        print(f"[INFO] Correct: {len(correct_samples)} samples saved to {args.output_json_correct}")

    wandb.finish()


if __name__ == "__main__":
    main()
