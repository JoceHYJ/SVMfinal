from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn import datasets, svm, metrics
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d, compute_class_weight
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets, _ovr_decision_function
from sklearn.utils.validation import check_is_fitted
from sklearn.svm._base import BaseLibSVM, LIBSVM_IMPL, _one_vs_one_coef
# from sklearn.svm._base import LIBSVM_IMPL, _one_vs_one_coef
# from sklearn.svm._base import BaseLibSVM, _one_vs_one_coef
# from sklearn.svm._base import BaseLibSVM, LIBSVM_IMPL


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# Create a classifier: a support vector classifier
# TODO
# clf = svm.SVC(gamma=0.001)

# MyClassifierMixin
class MyClassifierMixin:
    """Mixin class for all classifiers in scikit-learn."""

    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_y": True}


# MyBaseSVC
# TODO 修改 BaseLibSVM
class MyBaseSVC(MyClassifierMixin, BaseLibSVM, metaclass=ABCMeta):
    """ABC for LibSVM-based classifiers."""

    @abstractmethod
    def __init__(
            self,
            kernel,
            degree,
            gamma,
            coef0,
            tol,
            C,
            nu,
            shrinking,
            probability,
            cache_size,
            class_weight,
            verbose,
            max_iter,
            decision_function_shape,
            random_state,
            break_ties,
    ):
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=nu,
            epsilon=0.0,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            random_state=random_state,
        )

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        self.class_weight_ = compute_class_weight(self.class_weight, classes=cls, y=y_)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order="C")

    def decision_function(self, X):
        """Evaluate the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
            If decision_function_shape='ovr', the shape is (n_samples,
            n_classes).

        Notes
        -----
        If decision_function_shape='ovo', the function values are proportional
        to the distance of the samples X to the separating hyperplane. If the
        exact distances are required, divide the function values by the norm of
        the weight vector (``coef_``). See also `this question
        <https://stats.stackexchange.com/questions/14876/
        interpreting-distance-from-hyperplane-in-svm>`_ for further details.
        If decision_function_shape='ovr', the decision function is a monotonic
        transformation of ovo decision function.
        """
        dec = self._decision_function(X)
        if self.decision_function_shape == "ovr" and len(self.classes_) > 2:
            return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        return dec

    def predict(self, X):
        """Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        check_is_fitted(self)
        if self.break_ties and self.decision_function_shape == "ovo":
            raise ValueError(
                "break_ties must be False when decision_function_shape is 'ovo'"
            )

        if (
                self.break_ties
                and self.decision_function_shape == "ovr"
                and len(self.classes_) > 2
        ):
            y = np.argmax(self.decision_function(X), axis=1)
        else:
            y = super().predict(X)
        return self.classes_.take(np.asarray(y, dtype=np.intp))

    # Hacky way of getting predict_proba to raise an AttributeError when
    # probability=False using properties. Do not use this in new code; when
    # probabilities are not available depending on a setting, introduce two
    # estimators.
    def _check_proba(self):
        if not self.probability:
            raise AttributeError(
                "predict_proba is not available when  probability=False"
            )
        if self._impl not in ("c_svc", "nu_svc"):
            raise AttributeError("predict_proba only implemented for SVC and NuSVC")
        return True

    @available_if(_check_proba)
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        predict. Also, it will produce meaningless results on very small
        datasets.
        """
        X = self._validate_for_predict(X)
        if self.probA_.size == 0 or self.probB_.size == 0:
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False"
            )
        pred_proba = (
            self._sparse_predict_proba if self._sparse else self._dense_predict_proba
        )
        return pred_proba(X)

    @available_if(_check_proba)
    def predict_log_proba(self, X):
        """Compute log probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            Returns the log-probabilities of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        predict. Also, it will produce meaningless results on very small
        datasets.
        """
        return np.log(self.predict_proba(X))

    def _dense_predict_proba(self, X):
        X = self._compute_kernel(X)

        kernel = self.kernel
        if callable(kernel):
            kernel = "precomputed"

        # TODO 修改 LIBSVM_IMPL

        svm_type = LIBSVM_IMPL.index(self._impl)
        pprob = libsvm_predict_proba(
            X,
            self.support_,
            self.support_vectors_,
            self._n_support,
            self._dual_coef_,
            self._intercept_,
            self._probA,
            self._probB,
            svm_type=svm_type,
            kernel=kernel,
            degree=self.degree,
            cache_size=self.cache_size,
            coef0=self.coef0,
            gamma=self._gamma,
        )

        return pprob

    def _sparse_predict_proba(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order="C")

        kernel = self.kernel
        if callable(kernel):
            kernel = "precomputed"

        kernel_type = self._sparse_kernels.index(kernel)

        return libsvm_sparse_predict_proba(
            X.data,
            X.indices,
            X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data,
            self._intercept_,
            # TODO 修改 LIBSVM_IMPL
            LIBSVM_IMPL.index(self._impl),
            kernel_type,
            self.degree,
            self._gamma,
            self.coef0,
            self.tol,
            self.C,
            self.class_weight_,
            self.nu,
            self.epsilon,
            self.shrinking,
            self.probability,
            self._n_support,
            self._probA,
            self._probB,
        )

    def _get_coef(self):
        if self.dual_coef_.shape[0] == 1:
            # binary classifier
            coef = safe_sparse_dot(self.dual_coef_, self.support_vectors_)
        else:
            # 1vs1 classifier
            # TODO _one_vs_one_coef
            coef = _one_vs_one_coef(
                self.dual_coef_, self._n_support, self.support_vectors_
            )
            if sp.issparse(coef[0]):
                coef = sp.vstack(coef).tocsr()
            else:
                coef = np.vstack(coef)

        return coef

    @property
    def probA_(self):
        """Parameter learned in Platt scaling when `probability=True`.

        Returns
        -------
        ndarray of shape  (n_classes * (n_classes - 1) / 2)
        """
        return self._probA

    @property
    def probB_(self):
        """Parameter learned in Platt scaling when `probability=True`.

        Returns
        -------
        ndarray of shape  (n_classes * (n_classes - 1) / 2)
        """
        return self._probB


# MySVC
class MySVC(MyBaseSVC):
    """C-Support Vector Classification.

    The implementation is based on libsvm. The fit time scales at least
    quadratically with the number of samples and may be impractical
    beyond tens of thousands of samples. For large datasets
    consider using :class:`~sklearn.svm.LinearSVC` or
    :class:`~sklearn.linear_model.SGDClassifier` instead, possibly after a
    :class:`~sklearn.kernel_approximation.Nystroem` transformer.

    The multiclass support is handled according to a one-vs-one scheme.

    For details on the precise mathematical formulation of the provided
    kernel functions and how `gamma`, `coef0` and `degree` affect each
    other, see the corresponding section in the narrative documentation:
    :ref:`svm_kernels`.

    Read more in the :ref:`User Guide <svm_classification>`.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
        Specifies the kernel type to be used in the algorithm.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.

        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.

    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    class_weight : dict or 'balanced', default=None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, note that
        internally, one-vs-one ('ovo') is always used as a multi-class strategy
        to train models; an ovr matrix is only constructed from the ovo matrix.
        The parameter is ignored for binary classification.

        .. versionchanged:: 0.19
            decision_function_shape is 'ovr' by default.

        .. versionadded:: 0.17
           *decision_function_shape='ovr'* is recommended.

        .. versionchanged:: 0.17
           Deprecated *decision_function_shape='ovo' and None*.

    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.

        .. versionadded:: 0.22

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C for each class.
        Computed based on the ``class_weight`` parameter.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (n_classes -1, n_SV)
        Dual coefficients of the support vector in the decision
        function (see :ref:`sgd_mathematical_formulation`), multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the :ref:`multi-class section of the User Guide
        <svm_multi_class>` for details.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : ndarray of shape (n_classes * (n_classes - 1) // 2,)
        Number of iterations run by the optimization routine to fit the model.
        The shape of this attribute depends on the number of models optimized
        which in turn depends on the number of classes.

        .. versionadded:: 1.1

    support_ : ndarray of shape (n_SV)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.

    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.

    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
        If `probability=True`, it corresponds to the parameters learned in
        Platt scaling to produce probability estimates from decision values.
        If `probability=False`, it's an empty array. Platt scaling uses the
        logistic function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
        more information on the multiclass case and training procedure see
        section 8 of [1]_.

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    See Also
    --------
    SVR : Support Vector Machine for Regression implemented using libsvm.

    LinearSVC : Scalable Linear Support Vector Machine for classification
        implemented using liblinear. Check the See Also section of
        LinearSVC for more comparison element.

    References
    ----------
    .. [1] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

    .. [2] `Platt, John (1999). "Probabilistic outputs for support vector
        machines and comparison to regularizedlikelihood methods."
        <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm import SVC
    >>> clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    >>> clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svc', SVC(gamma='auto'))])

    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    _impl = "c_svc"

    def __init__(
            self,
            *,
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=0.0,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }


# TODO
def libsvm_predict_proba(*args, **kwargs):  # real signature unknown
    """
    Predict probabilities

        svm_model stores all parameters needed to predict a given value.

        For speed, all real work is done at the C level in function
        copy_predict (libsvm_helper.c).

        We have to reconstruct model and parameters to make sure we stay
        in sync with the python object.

        See sklearn.svm.predict for a complete list of parameters.

        Parameters
        ----------
        X : array-like, dtype=float of shape (n_samples, n_features)

        support : array of shape (n_support,)
            Index of support vectors in training set.

        SV : array of shape (n_support, n_features)
            Support vectors.

        nSV : array of shape (n_class,)
            Number of support vectors in each class.

        sv_coef : array of shape (n_class-1, n_support)
            Coefficients of support vectors in decision function.

        intercept : array of shape (n_class*(n_class-1)/2,)
            Intercept in decision function.

        probA, probB : array of shape (n_class*(n_class-1)/2,)
            Probability estimates.

        svm_type : {0, 1, 2, 3, 4}, default=0
            Type of SVM: C_SVC, NuSVC, OneClassSVM, EpsilonSVR or NuSVR
            respectively.

        kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default="rbf"
            Kernel to use in the model: linear, polynomial, RBF, sigmoid
            or precomputed.

        degree : int32, default=3
            Degree of the polynomial kernel (only relevant if kernel is
            set to polynomial).

        gamma : float64, default=0.1
            Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other
            kernels.

        coef0 : float64, default=0.0
            Independent parameter in poly/sigmoid kernel.

        Returns
        -------
        dec_values : array
            Predicted values.
    """
    pass


def libsvm_sparse_predict_proba(*args, **kwargs):  # real signature unknown
    """ Predict values T given a model. """
    pass


clf = MySVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
