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

### 함수 정의 절차 
1. 함수 이름(메서드 이름) 정의 (카멜 표기법, 스네이크 등등)
2. 매개변수 설정 * 매개변수는 없는것이 좋다.
3. 반환값 설정 * 반환값은 무조건 있어야 한다.

### 기초 통계 출력
- pandas로 변환하라는 말이다.
1. x = pd.DataFrame(iris.data, columns=iris.feature_names)
2. x.describe()

### predict 메서드
- 결정 트리 모델이 X_test 데이터를 입력으로 받아 각 샘플에 대한 예측을 수행하는 메서드

### classification_report
- Precision (정밀도),  Recall (재현율)

### tree 시각화
1. from sklearn.tree import plot_tree
2. plot_tree(dt)
3. plt.show()
- 어떤 방향으로 가는지 이해하기 어렵다.

from sklearn.tree import plot_tree
plot_tree(dt, 
          feature_names=iris.feature_names,
          class_names=iris.target_names)
plt.show()
print(dt.feature_importances_) # 중요도 출력

# 결정트리에서는 중요도가 뒤로 갈 수록 중요하다.
# 결정트리에서의 문제점은 과대적합되는 것이다. 
해결방법 : (매개변수를 이용하여 높이를 제한한다.)

### make_moons
- 두 개의 클래스(반달 모양)를 생성하며, 각 클래스는 2차원 공간에 위치
- 비선형 경계가 있는 분류 문제를 시뮬레이션하는 데 적합하다.

### accuracy_score
- 특정 레이블(또는 클래스) 분포를 유지하도록 도와주는 매개변수

### RandomForest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=42)
forest.fit(X_train, y_train)

# 파이썬에서 할 줄 알아야 하는 것
1. 함수 만들기 - > def 함수명() :
2. 함수 사용하는 방법 - > 함수명( )
- 메서드의 필수 옵션을 잘 알고 있어야 한다.
# 파이썬 슬라이싱 조심해야 한다.

## 포화성이 8:2 정도되면 편향된 데이터가 된다.

### 랜덤 포레스트 시각화
fig, axes = plt.subplots(2, 3, figsize=(20,10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
  ax.set_title(f"트리 {i}")
  mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

### 랜덤 포레스트 분류 문제 해결 절차
# 1. 데이터 불러오기
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# 2. 데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,stratify=cancer.target,random_state=42)

# 3. 분류기를 생성
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. 분류기를 학습
forest.fit(X_train, y_train)

# 5. 예측 및 측정
from sklearn.metrics import accuracy_score
y_pred=forest.predict(X_test)
print(forest.score(X_train, y_train))
print(accuracy_score(y_pred, y_test))

### 커널 서포트 벡터 머신
- 머신러닝 지도학습 모델로, 분류와 회귀 과제에 사용
- 평면도를 이용하여 벡터와 벡터 사이의 착안점을 비교
- 데이터를 튀길 건데 어떻게 튀기는지는 알지마라.
### 알고리즈믹하게 과대적합이 일어나면 해결 불가능하다.

### SVC 사용절차
1. 모델 불러오기
from sklearn.svm import SVC

2. 모델 적용
svc = SVC()
svc.fit(X_train, y_train)

3. 모델 검증
print(svc.score(X_train, y_train))
print(svc.score(X_test, y_test))

### SVM에 적합한 전처리
min_on_training = X_train.min(axis=0) # 각 독립변수의 최소값을 가져오기
min_on_training
range_on_training = (X_train - min_on_training).max(axis=0) # 최소값을 제외한 최대값을 계산
range_on_training

X_train_scaled = (X_train - min_on_training) / range_on_training # 각 독립 변수에서 최소값을 빼고, 그 결과를 각 독립 변수의 범위로 나누기, # 모든 독립 변수의 값이 0과 1 사이로 정규화, 각 독립 변수의 최소값은 0, 최대값은 1
print(X_train_scaled.min(axis=0), X_train_scaled.max(axis=0))  # 최소값, 최대값 출력

### SVM의 C값을 조정
svc = SVC(C=20) 
# C값이 적으면 과소적합 발생, 많으면 과대적합 발생

### 데이터 분석 과정(시계열이면 데이터를 알고 있어야 한다.)
1) 데이터 수집
- 시계열, 시계열이 아님

2) 데이터 정제
- 시계열, 시계열이 아님

3) 데이터 분석
- 시계열, 시계열 아님

4) 배포

### CSV 파일 불러와서 객체 생성
import pandas as pd
ram_prices = pd.read_csv("data/ram_price.csv")

### matplotlib 도식화
import matplotlib.pyplot as plt
plt.plot(ram_prices.date, ram_prices.price)
plt.xlabel("년도")
plt.ylabel("가격 ($/Mbyte)")
plt.show()

### log 그래프로 변경
plt.semilogy(ram_prices.date, ram_prices.price)

### y 축의 폰트를 consolas로 변경
plt.yticks(fontname = "consolas")

### 시계열 데이터(학습, 테스트)분리
## boolean index
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# 판다스 필수 역량
1) 새로운 열 추가 
2) 열 삭제 
3) 데이터 뽑아내기(csv, ... .. 등 다른파일로) # to csv, to_
4) bool index 

# 학습데이터와 테스트 데이터를 분류(y값은 log 함수를 적용)
# data_train.date #인덱스가 포함된 데이터 이기때문에 사용할 수 없다.
X_train = data_train.date.to_numpy() [:,np.newaxis]
y_train = np.log(data_train.price) # np의 log만 사용하자!!

# 결정트리의 회귀와 선형 회귀 예측기를 생성하고, 데이터에 학습
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

## 시계열 데이터에는 랜덤 포레스트, 결정 트리를 사용하기에는 옳지 않다.
## 원본 데이터를 모를 떄는 랜덤 포레스트 or 결정 트리를 먼저 사용하여 중요한 특성(feature)을 먼저 파악하자. 
## 선형부터 그어보기, 중요한 데이터 파악


## 비지도 학습
step 1 전처리
- 데이터의 스케일을 조정하여 알고리즘의 성능을 개선
1) Min-MAX 정규화 : 데이터의 값을 0과 1 사이로 변환하는 방법
2) 표준화(Standardization) : 데이터의 평균을 0, 표준편차를 1로 변환하는 방법

step 2 특성추출
- 고차원 데이터를 저차원으로 변환하여 데이터의 주요 특성을 유지하는 과정
1) PCA(Principal Component Analysis), 차원 축소 : 데이터의 분산을 최대화하는 2) 방향으로 새로운 축(주성분)을 생성하여 차원을 축소하는 기법
3) t-SNE : 고차원 데이터를 저차원으로 변환하여 데이터 포인트 간의 유사성을 유지하는 비선형 차원 축소 기법

step 3 군집화
1) DBSCAN : 데이터 포인트의 밀도를 기반으로 군집을 형성하는 알고리즘





