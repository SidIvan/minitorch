from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Set

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    function_arguments = list(vals)
    function_arguments[arg] += epsilon
    f_plus = f(*function_arguments)
    function_arguments[arg] -= 2 * epsilon
    f_minus = f(*function_arguments)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    sorted_vars = []
    dfs(variable, sorted_vars)
    return reversed(sorted_vars)


def dfs(var: Variable, sorted_vars: List[Variable]):
    if var.is_constant():
        return
    for neighbour in var.parents:
        dfs(neighbour, sorted_vars)
    sorted_vars.append(var)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topological_order = list(topological_sort(variable))
    var_id_to_derivative = {}
    for var in topological_order:
        var_id_to_derivative[var.unique_id] = deriv

    for var in topological_order:
        var_id = var.unique_id
        if var.is_leaf():
            var.accumulate_derivative(var_id_to_derivative[var_id])
            continue
        for parent, derivative_parent in var.chain_rule(var_id_to_derivative[var_id]):
            parent_id = parent.unique_id
            if parent_id in var_id_to_derivative:
                var_id_to_derivative[parent_id] = derivative_parent


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
