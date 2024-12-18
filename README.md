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
- 타겟이 없다.
- 우리도 답을 모른다.
공리 : 유사한 데이터는 유사하게 동작한다.

step 1 전처리
- 데이터의 스케일을 조정하여 알고리즘의 성능을 개선
1) Min-MAX 정규화 : 데이터의 값을 0과 1 사이로 변환하는 방법
2) 표준화(Standardization) : 데이터의 평균을 0, 표준편차를 1로 변환하는 방법

step 2 특성추출
- 고차원 데이터를 저차원으로 변환하여 데이터의 주요 특성을 유지하는 과정
1) PCA(Principal Component Analysis), 차원 축소 : 데이터의 분산을 최대화하는 방향으로 새로운 축(주성분)을 생성하여 차원을 축소하는 기법
2) t-SNE : 고차원 데이터를 저차원으로 변환하여 데이터 포인트 간의 유사성을 유지하는 비선형 차원 축소 기법

step 3 군집화
1) DBSCAN : 데이터 포인트의 밀도를 기반으로 군집을 형성하는 알고리즘

### StandardScaler, MinMaxScaler, Normalizer : 아웃라이어에 예민하다.
### 아웃라이어에 노출된 데이터는 RobustScaler를 사용하는 것이 현명하다
### 전처리 과정에서 누락데이터와 아웃라이어가 발생할 경우 결정을 해야할 떄가 생긴다.

# 중간 값 (Median)
- 데이터가 정렬되었을 때, 중앙에 위치한 값

# 사분위 값 (Quartiles)
- 사분위 값은 데이터를 4등분한 값으로, 분포의 특정 지점

1사분위(Q1): 하위 25%에 해당하는 값(데이터의 25%가 이 값 이하).
2사분위(Q2): 중간 값(데이터의 50%가 이 값 이하).
3사분위(Q3): 상위 25%에 해당하는 값(데이터의 75%가 이 값 이하).

# MinMaxScaler
- 데이터를 특정 최소값과 최대값(기본적으로 0과 1) 사이로 스케일링
- 원본 형태를 유지해야 할 경우 사용

### MinMaxScaler 사용방법
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled

### 테스트 데이터를 MinMaxScaler로 변환하고 최소값과 최대값을 출력
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.min(axis=0), X_test_scaled.max(axis=0))

### StandardScaler
- 데이터를 평균이 0, 표준편차가 1이 되도록 변환
- 단위값등이 다 달라져서 어떤 값으로 컬럼을 일치 시켜야 할 떄 

### StandardScaler 사용방법
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_train_scaled

### PCA(주성분 분석)
- 데이터의 분산을 최대화하는 새로운 축(차원)을 찾는 과정에서 생성되는 직교(orthogonal)벡터들
- 중요도를 알 수 없기 떄문에 
- 차수를 2차 평면으로 차근차근 잘라보기

### 차원축소(Dimensionality Reduction)
- 높은 차원의 데이터를 더 적은 차원의 데이터로 변환
- 데이터의 대부분의 정보를 유지하면서 불필요한 차원을 제거
- 비지도 데이터의 중요도를 판별 후 전처리로 돌아감 최종에는 군집으로 이동 -> 답안 작성(레이블 결정, 타겟 결정)

### PCA로 데이터 변환
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x_scaled)
X_pca = pca.transform(x_scaled)
print(x_scaled.shape, X_pca.shape)

### PCA 데이터 시각화
plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0], X_pca[:,1], cancer.target)
plt.show()

### 군집이론
- 유사한 특성을 가진 그룹(군집, 클러스터)을 식별하는 데 사용되는 기법

### KMeans
- 클러스터의 중심을 몇개로 잡는지가 중점이다.
- 평균점과 새로운 데이터가 들어오면 알고리즘이 붕괴될 수 있다.
- 속도가 빠른 장점이 있다.

### 병합 군집
- 계층적 군집(Hierarchical Clustering)의 한 방법으로, 데이터를 개별 군집으로 시작하여 점차 병합하여 하나의 군집이 될 때까지 반복하는 바텀업 방식 군집화 방법
- 계산을 전부 다 해야하기 떄문에 속도가 느리다는 단점이 있다.

### DBSCAN
- 밀도 기반 군집화 알고리즘으로, 데이터의 밀집된 영역을 클러스터로 간주하고, 밀도가 낮은 영역은 노이즈 또는 이상치로 처리하는 방법

### 클래스(모델)을 임포트할 떄 저장되어 있는 위치를 기억하자!
### Pandas가 아니면 어떤 컬럼인지 알 수 없다. # scaler.fit_transform(df)

### t-SNE
- 고차원 데이터를 저차원(주로 2D 또는 3D) 공간에 시각화하기 위한 차원 축소 알고리즘
- 실행할 떄마다 값이 달라진다.


### 교차검증
- 머신러닝 모델의 성능을 평가하고 일반화 성능을 향상시키기 위해 데이터를 훈련 데이터와 검증 데이터로 반복적으로 나누어 검증하는 방법
- 델이 훈련 데이터에 과적합(overfitting)되지 않고 새로운 데이터에도 잘 작동하도록 돕는다.

### 교차검증 사용
from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, iris.data, iris.target)

### 5장에서 배우는 것
1) 데이터 폴드를 통해 데이터를 높임
2) CV : 랜덤 또는 서치를 통해 매개변수를 조절

### 혼동행렬(Confusion Matrix)
- 머신러닝 분류 모델의 성능을 평가할 때 사용되는 표 형식의 도구
- 모델이 예측한 결과와 실제 정답 간의 관계를 정리하여 성능을 시각적으로 확인할 수 있게 한다. 
- 데이터 머신러닝 380 ~ 381쪽 중요!

### 혼동행렬의 구조
True Positive (TP): 실제로 Positive이고 모델이 Positive로 예측한 경우 (올바른 긍정 예측)
True Negative (TN): 실제로 Negative이고 모델이 Negative로 예측한 경우 (올바른 부정 예측)
False Positive (FP): 실제로 Negative인데 모델이 Positive로 예측한 경우 (오류 긍정 예측), 2종 오류
False Negative (FN): 실제로 Positive인데 모델이 Negative로 예측한 경우 (오류 부정 예측), 1종 오류

### 주요 지표 계산
1. 정확도(Accuracy)
- 모델이 얼마나 정확하게 예측, 데이터가 활용도가 높은가?
Accuracy= TP+TN+FP+FN / TP+TN

2. 정밀도(Precision)
- 모델이 Positive로 예측한 것 중 실제 Positive의 비율

3. 재현율(Recall, Sensitivity)
- 실제 Positive 중에서 모델이 Positive로 정확히 예측한 비율

4. F1-스코어(F1 Score)
- 정밀도와 재현율의 조화평균으로, 두 값 사이의 균형

5. 특이도(Specificity)
- 실제 Negative 중에서 모델이 Negative로 정확히 예측한 비율

### Bias
- 데이터가 특정 방향으로 편향되어 있는 상태

​### 직관적 이해를 돕는 예시: 병원 진단
1종 오류: 병이 없는데 병이 있다고 진단 (거짓 양성, False Positive)
2종 오류: 병이 있는데 병이 없다고 진단 (거짓 음성, False Negative)

### 영화 감정 분석<긍정, 부정>
1장 데이터 수집
2장 전처리 : Pandas로 변환
3장 특징 추출 : 
4장 모델 학습 : 랜덤 포레스트, 선형 모델
5장 모델 평가 : 혼동행렬, 정확도


