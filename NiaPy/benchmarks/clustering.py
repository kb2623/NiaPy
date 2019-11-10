# encoding=utf8
"""The module implementing Ackley benchmark."""
from typing import Callable, Optional, List

import numpy as np
from sklearn.preprocessing import LabelEncoder

from NiaPy.benchmarks.benchmark import Benchmark
from NiaPy.util import classifie, groupdatabylabel, clusters2labels
from .functions import clustering_function, clustering_min_function

__all__ = ["Clustering", "ClusteringMin", "ClusteringMinPenalty", "ClusteringClassification"]

class Clustering(Benchmark):
	r"""Implementation of Clustering function.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Clustering function

		:math:`f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2`

		Input domain:
			Depends on dataset used.

		Global minimum:
			Depends on dataset used.

	LaTeX formats:
		Inline:
			$f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2$

		Equation:
			\begin{equation} f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2 \end{equation}

	Attributes:
		dataset: Dataset to use for clustering.
		a: Number of attirbutes in dataset.
	"""
	Name: List[str] = ["Clustering"]

	def __init__(self, dataset: np.ndarray) -> None:
		"""Initialize Clustering benchmark.

		Args:
			dataset: Dataset.

		See Also:
			* :func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, np.min(dataset, axis=0), np.max(dataset, axis=0))
		self.dataset, self.a = dataset.astype(float), len(self.Lower)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			Latex code.
		"""
		return r"""$f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		a = self.a
		def fun(x: np.ndarray, w: Optional[np.ndarray] = None, **kwarg: dict):
			k = int(len(x) / a)  # Number of clusters
			ww = w if w is not None else np.ones(int(a * k), dtype=float)  # Weights
			return clustering_function(self.a, self.dataset.flatten(), self.dataset.shape[1], x, k, ww)
		return fun

class ClusteringMin(Clustering):
	r"""Implementation of Clustering min function.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Clustering min function

		:math:`f(\mathbf{O}, \mathbf{Z}) = \min_{j=1}^M \left( \sum_{i=1}^N w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2 \right)`

		Input domain:
			Depends on dataset used.

		Global minimum:
			Depends on dataset used.

	LaTeX formats:
		Inline:
			$f(\mathbf{O}, \mathbf{Z}) = \min_{j=1}^M \left( \sum_{i=1}^N w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2 \right)$

		Equation:
			\begin{equation}  f(\mathbf{O}, \mathbf{Z}) = \min_{j=1}^M \left( \sum_{i=1}^N w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \end{equation}
	"""
	Name: List[str] = ["ClusteringMin"]

	def __init__(self, dataset: np.ndarray) -> None:
		"""Initialize Clustering min benchmark.

		Args:
			dataset: Dataset.

		See Also:
			* :func:`NiaPy.benchmarks.Clustering.__init__`
		"""
		Clustering.__init__(self, dataset)
		self.a = len(self.Lower)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			str: latex code.
		"""
		return r"""$f(\mathbf{O}, \mathbf{Z}) = \sum_{i=1}^N \sum_{j=1}^K w_{ij} || \mathbf{o}_i - \mathbf{z}_j ||^2$"""

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Evaluation function.
		"""
		a = self.a
		def fun(x: np.ndarray, k: Optional[int] = None, w: Optional[np.ndarray] = None, **dict: dict) -> float:
			k = k if k is not None else int(len(x) / a)  # Number of clusters
			w = w if w is not None else np.ones(int(a * k), dtype=float)  # Weights
			return clustering_min_function(self.a, self.dataset.flatten(), self.dataset.shape[1], x, k, w)
		return fun

class ClusteringMinPenalty(ClusteringMin):
	r"""Implementation of Clustering min function with penalty.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Clustering min with penalty function

		:math:`\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)`

		Input domain:
			Depends on dataset used.

		Global minimum:
			Depends on dataset used.

	LaTeX formats:
		Inline:
			$\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)$

		Equation:
			\begin{equation} \mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right) \end{equation}

	Attributes:
		range: Array of ranges between lower and upper values.

	See Also:
		:class:`NiaPy.benchmark.ClusteringMin`
	"""
	Name: List[str] = ["ClusteringMinPenalty"]

	def __init__(self, dataset: np.ndarray) -> None:
		"""Initialize Clustering min benchmark.

		Args:
			dataset: Dataset.

		See Also:
			* :func:`NiaPy.benchmarks.ClusteringMin.__init__`
		"""
		ClusteringMin.__init__(self, dataset)
		self.range = np.abs(self.Upper - self.Lower)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			str: latex code.
		"""
		return r"""\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)"""

	def penalty(self, x: np.ndarray, k: int) -> float:
		r"""Get penelty for inidividual.

		Args:
			x: Individual.
			k: Number of clusters

		Returns:
			Penalty for the given individual.
		"""
		p, r = 0, self.range / k
		for i in range(k - 1):
			for j in range(k - i - 1):
				if i != k - j - 1: p += np.sum(np.fmin(r, np.fmax(np.zeros(self.a), r - np.abs(x[self.a * i:self.a * (i + 1)] - x[self.a * (k - j - 1):self.a * (k - j)]))))
		return p

	def function(self) -> Callable[[np.ndarray, dict], float]:
		"""Return benchmark evaluation function.

		Returns:
			Callable[[numpy.ndarray], float]: Evaluation function.
		"""
		fcm, a = ClusteringMin.function(self), self.a
		def fun(x: np.ndarray, w: Optional[int] = None, **kwargs: dict) -> float:
			k = int(len(x) / a)  # Number of clusters
			return fcm(x, k=k) + self.penalty(x, k)
		return fun

class ClusteringClassification(ClusteringMinPenalty):
	r"""Implementation of Clustering min function with penalty.

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
		Ackley function

		:math:`\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)`

		Input domain:
			Depends on dataset used.

		Global minimum:
			Depends on dataset used.

	LaTeX formats:
		Inline:
			$\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)$

		Equation:
			\begin{equation} \mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right) \end{equation}

	Attributes:
		labels: Labels for dataset.
		lt: Label transform 

	See Also:
		:class:`NiaPy.benchmark.ClusteringMinPenalty`
	"""
	Name: List[str] = ["ClusteringClassification"]

	def __init__(self, dataset: np.ndarray, labels: np.ndarray) -> None:
		"""Initialize Clustering min benchmark.

		Args:
			dataset: Dataset.

		See Also:
			* :func:`NiaPy.benchmarks.ClusteringMin.__init__`
		"""
		ClusteringMinPenalty.__init__(self, dataset)
		self.labels, self.lt = labels, LabelEncoder().fit(labels)
		self.gl = groupdatabylabel(dataset, labels, self.lt)

	@staticmethod
	def latex_code() -> str:
		"""Return the latex code of the problem.

		Returns:
			str: latex code.
		"""
		return r"""\mathcal{f} \left(\mathbf{O}, \mathbf{Z} \right) = \mathcal{p} \left( \mathbf{Z} \right) + \sum_{i=0}^{\mathit{N}-1} \min_{\forall j \in \mathfrak{k}} \left( w_{i, j} \times || \mathbf{o}_i - \mathbf{z}_j ||^2 \right) \\ \mathcal{p} \left(\mathbf{Z} \right) = \sum_{\forall \mathbf{e} \in \mathfrak{I}} \sum_{j=0}^{\mathit{A}-1} \min \left(\frac{r}{\mathit{K}}, \max \left(0, \frac{r}{\mathit{K}} - || z_{e_0, j} - z_{e_1, j} ||^2 \right) \right)"""

	def penalty(self, x: np.ndarray, k: int) -> float:
		r"""Get penelty for inidividual.

		Args:
			x: Individual.
			k: Number of clusters

		Returns:
			Penalty for the given individual.
		"""
		ok, C = 0, x.reshape([k, self.a])
		l = clusters2labels(C, self.gl)
		for i, d in enumerate(self.dataset): ok += 1 if self.lt.inverse_transform([l[classifie(d, C)]]) == self.labels[i] else 0
		return (1 - ok / len(self.dataset)) * np.sum(self.range) * 2

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
