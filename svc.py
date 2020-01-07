import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = r"F:\nemo\py_workspace\MyTest\iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# 数据集维度
# 我们可以快速的了解数据的形状属性包含了多少行（示例）和多少列（属性）。
# shape
# print(dataset.shape)

# 详细查看数据
# 查看数据的前20行：
# head
# print(dataset.head(20))

# 统计摘要
# 现在我们可以看看对每个属性的统计摘要，包含了数量、平均值、最大值、最小值，还有一些百分位数值。
# descriptions
# print(dataset.describe())

# 数据可视化
# 单变量图形
# 我们先以一些单变量图形开始，也就是每个单独变量的图形。
# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# 我们也可以为每个输入变量创建一个直方图以了解它们的分布状况。
# histograms
# dataset.hist()
# plt.show()

# 多变量图形
# 看看变量之间的相互作用
# 我们看看全部属性对的散点图，这有助于我们看出输入变量之间的结构化关系。
# scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

# 评估算法
# 现在我们为数据搭建一些模型，并测试它们对不可见数据的准确度。
#
# 这一部分的主要步骤为：
#
# 将数据集分离出一个验证集。
# 设定测试工具，使用10折交叉验证。
# 搭建6个不同的模型根据花朵测量值预测出鸢尾花种类。
# 选出最佳模型。

# 创建验证集
# 我们需要知道搭建的模型效果怎样。后面我们会用统计方法来验证模型对新数据的准确度。我们还希望通过评估模型在真正不可见数据时的表现，来进一步确定模型的准确度。
#
# 也就是我们会留一些数据不让算法看到，然后用这些数据来确定模型到底有多准确。
#
# 我们会将导入的数据集拆分为两部分，80% 用于训练模型，20% 用于验证模型。
# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
# 得到的 X_train 和 Y_train 里的训练数据用于准备模型，得到的 X_validation 和 Y_validation 集我们后面会用到。

# 测试工具
# 我们会用十折交叉验证法测试模型的准确度。
#
# 这会将我们的数据集分成 10 部分，轮流将其中 9 份作为训练数据，1份作为测试数据，进行试验。

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# 现在我们用“准确率”这个维度去评估模型，也就是能正确预测出鸢尾花类别的比例。我们后面运行和评估模型时会使用分数变量。

# 搭建模型
# 针对这个问题，我们并不知道哪个算法最好，应当用哪些配置。我们从可视化图表中能够得知在有些维度上一些类别的部分是线性可分的，因此我们预期总体效果会不错。
#
# 我们看看这 6 种算法：
#
# 逻辑回归（LR）
# 线性判别分析（LDA）
# K最近邻算法（KNN）
# 分类和回归树（CART）
# 高斯朴素贝叶斯（NB）
# 支持向量机（SVM）
# 这里面既有简单的线性算法（LA和LDA），也有非线性算法（KNN，CART，NB和SVM）。我们每次运行算法前都要重新设置随机数量的种子，以确保是在用相同的数据拆分来评估每个算法。这样能保证最终结果可以直接进行比较。
#
# 我们来搭建和评估模型：
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    # cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    # results.append(cv_results)
    # names.append(name)
    # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    # print(msg)

# 选择最佳模型
# 我们现在获得了 6 个模型以及每种模型的准确度评估状况。接下来需要将模型相互比较，选出最准确的那个。
