## 파이썬 머신러닝 수업

### 외부 라이브러리 설치
pip install 라이브러리명
pip install pandas scikit-learn seaborn graphviz

### matplotlib 라이브러리 폰트 설정
matplotlib.rc("font", family="Malgun Gothic") # 윈도우
matplotlib.rcParams["axes.unicode_minus"] = False #  마이너스 기호가 깨지는 문제를 해결하기 위한 설정

### 설치되어있는 외부 라이브러리 리스트 출력
pip list

### 외부 라이브러리 업데이트
pip install -U 라이브러리명

### 외부 라이브러리 버전 맞추기
pip install 라이브러리명==원하는_버전_번호

### Bunch.keys( )
data: 샘플 데이터 (NumPy 배열)
target: 레이블 데이터 (NumPy 배열)
feature_names: 특성 이름 목록
target_names: 레이블 이름 목록
DESCR: 데이터셋 설명
filename: 데이터셋 파일 저장 위치


### (데이터 모델 학습 및 모델 성능평가 과정)
### 1. 데이터 불러오기
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

### 2. 학습 데이터 및 테스트 데이터 분리하기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    stratify=cancer.target,
                                                    random_state=42)

### 3. 모델 객체 생성
dt = DecisionTreeClassifier(random_state=42)

### 4. 모델 학습
dt.fit(X_train, y_train)

### 5. 모델 성능 평가
print(f"훈련 정확도 : {dt.score(X_train, y_train)}") # 훈련 정확도
print(f"테스트 정확도 : {dt.score(X_test, y_test)}") # 테스트 정확도 

### (분류 문제를 개선)
### 트리의 최대 높이를 4로 제한
dt = DecisionTreeClassifier(max_depth=4, random_state=42)

### 결정트리 시각화
from sklearn.tree import export_graphviz
export_graphviz(dt,   
                out_file="dt.dot",  
                class_names=["F","T"], 
                feature_names=cancer.feature_names)


### matplotlib를 사용해서 `dot` 파일을 출력
import graphviz
with open("dt.dot") as f:
  dot_graph = f.read()
display(graphviz.Source(dot_graph))

### scikit-learn에서 제공하는 `plot_tree` 함수를 활용
from sklearn.tree import plot_tree
plt.figure(figsize=(12,6))
plot_tree(dt,class_names=["악성","양성"],  
          feature_names=cancer.feature_names, 
          filled=True)
plt.show()

### 트리의 특성 중요도 파악 (막대, 스케터(단위 관계), 선)
dt.feature_importances_