using SciMLOperators, BenchmarkTools
using LinearAlgebra, SparseArrays, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

N = 100
v = rand(rng, N)
w = rand(rng, N)
u_state = zeros(N)
A = rand(rng, N, N)
As = sparse(A)
d = rand(rng, N)

# =============================================================================
# Operator construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

f_oop(vv, u, p, t) = A * vv
f_iip(ww, vv, u, p, t) = mul!(ww, A, vv)

SUITE["construct"]["MatrixOperator"] = @benchmarkable MatrixOperator($A)
SUITE["construct"]["DiagonalOperator"] = @benchmarkable DiagonalOperator($d)
SUITE["construct"]["ScalarOperator"] = @benchmarkable ScalarOperator(2.0)
SUITE["construct"]["FunctionOperator"] = @benchmarkable FunctionOperator(
    $f_iip, $v, $w; u = $u_state, p = nothing, t = 0.0
)

# =============================================================================
# Application: L(v, u, p, t) out-of-place, L(w, v, u, p, t) in-place
# =============================================================================

SUITE["apply"] = BenchmarkGroup()

L = MatrixOperator(A)
Ls = MatrixOperator(As)
Ld = DiagonalOperator(d)
α = ScalarOperator(2.0)

SUITE["apply"]["dense_matvec"] = @benchmarkable $L * $v
SUITE["apply"]["sparse_matvec"] = @benchmarkable $Ls * $v
SUITE["apply"]["diagonal_matvec"] = @benchmarkable $Ld * $v
SUITE["apply"]["oop_eval"] = @benchmarkable $L($v, $u_state, nothing, 0.0)
SUITE["apply"]["iip_eval"] = @benchmarkable $L($w, $v, $u_state, nothing, 0.0)

# =============================================================================
# Operator algebra
# =============================================================================

SUITE["algebra"] = BenchmarkGroup()

SUITE["algebra"]["compose"] = @benchmarkable $L * $L
SUITE["algebra"]["add"] = @benchmarkable $L + $Ld
SUITE["algebra"]["scalar_mul"] = @benchmarkable $α * $L

Lcomp = L * Ld
SUITE["algebra"]["composed_apply"] = @benchmarkable $Lcomp($v, $u_state, nothing, 0.0)

F = FunctionOperator(f_iip, v, w; u = u_state, p = nothing, t = 0.0)
SUITE["algebra"]["function_operator_apply"] = @benchmarkable $F(
    $w, $v, $u_state, nothing, 0.0
)

Lsum = 2L + Ld
SUITE["algebra"]["sum_apply"] = @benchmarkable $Lsum($v, $u_state, nothing, 0.0)

# =============================================================================
# Tensor product
# =============================================================================

SUITE["tensor"] = BenchmarkGroup()

A1 = rand(rng, 20, 20)
A2 = rand(rng, 30, 30)
L1 = MatrixOperator(A1)
L2 = MatrixOperator(A2)

SUITE["tensor"]["construct"] = @benchmarkable TensorProductOperator($L1, $L2)
T = TensorProductOperator(L1, L2)
v2 = rand(rng, 600)
u2 = zeros(600)
SUITE["tensor"]["apply"] = @benchmarkable $T * $v2
SUITE["tensor"]["oop_eval"] = @benchmarkable $T($v2, $u2, nothing, 0.0)
