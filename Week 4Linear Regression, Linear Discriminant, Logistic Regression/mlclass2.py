import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import colors

# colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

def plot_decision_2d_lda(lda,X,y,padding=1,discriminant=True,title="Decision Regions",lda_on=False):
    '''
    plot_decision_2d(clf,X,y)
    Plots a 2D decision region.
    '''
    # create a mesh to plot in
    x_min, x_max = X[:,0].min() - padding, X[:,0].max() + padding
    y_min, y_max = X[:,1].min() - padding, X[:,1].max() + padding

    h = (x_max-x_min)/1000.0 # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))

    if discriminant==True:
        plt.contour(xx, yy, Z, [0.5], linewidths=1., colors='k', alpha=0.5)

    # means
    if lda_on==True:
        plt.plot(lda.means_[0][0], lda.means_[0][1],
                'o', color='black', markersize=10)
        plt.plot(lda.means_[1][0], lda.means_[1][1],
                'o', color='black', markersize=10)

    # Plot also the training points
    plt.scatter(X[:,0], X[:,1], c=y, alpha=0.5, edgecolors='none',cmap='RdBu')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.show()

def simplemetrics(y_valid, y_score, scaler=1):
    figsize_a = 5.8*scaler
    figsize_b = 4.0*scaler
    fig = plt.figure(figsize=(figsize_a,figsize_b))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax1 = fig.add_subplot(gs[0])

    fpr, tpr, _ = metrics.roc_curve(y_valid.ravel(), y_score.ravel())
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.08])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    ax2 = fig.add_subplot(gs[1])

    confmat = metrics.confusion_matrix(y_valid, y_score)

    ax2.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    ax2.xaxis.set_label_position('top')
    ax2.set_xlabel('Predicted')
    ax2.xaxis.set_label_position('top')
    ax2.set_ylabel('True')#, rotation=0)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax2.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.title('Confusion Matrix', y=1.5)
    plt.tick_params(
        axis='both',       # options x, y, both
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        right='off',
    ) # labels along the bottom edge are off

    ax2.text(-0.5, 3, u'Precision: %0.2f' % metrics.precision_score(y_valid, y_score), fontsize=12)
    ax2.text(-0.5, 3.5, u'Recall: %0.2f' % metrics.recall_score(y_valid, y_score), fontsize=12)
    ax2.text(-0.5, 4, u'F1 Score: %0.2f' % metrics.f1_score(y_valid, y_score), fontsize=12)
    plt.tight_layout()
    plt.show()

# plot functions
def plot_data(lda, X, y, y_pred, fig_index):

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '.', color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '.', color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
