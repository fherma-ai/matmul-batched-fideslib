# Batched matrix multiplication over CKKS — FIDESlib (GPU)

> Implements [`matrix-multiplication` / `secret-matrix-batches@1.0.0`](https://www.fherma.io/kernels/matrix-multiplication/specifications/secret-matrix-batches)
> on the FHERMA kernel catalogue.
>
> The circuit is [Aikata](https://www.iaik.tugraz.at/person/aikata-aikata/)'s
> winning entry to the [FHERMA Matrix Multiplication challenge](https://fherma.io/challenges/652bf669485c878710fd020b),
> published as the MatrixMultiplication component of
> [polycircuit](https://github.com/fairmath/polycircuit) (Apache-2.0). This is
> that circuit moved onto the GPU with [FIDESlib](https://github.com/CAPS-UMU/FIDESlib),
> operation for operation. The CPU original is in
> [fherma-ai/matmul-batched-ckks](https://github.com/fherma-ai/matmul-batched-ckks).

P pairs of N×N matrices, multiplied pairwise — `c[i] = a[i] · b[i]`.

```text
kernel matmul<P: u32, N: u32>(
    %a: secret<tensor<P x N x N x f64>>,
    %b: secret<tensor<P x N x N x f64>>,
) -> %c: secret<tensor<P x N x N x f64>>
```

## What FIDESlib changes, and what it does not

FIDESlib is a CUDA CKKS library that takes an OpenFHE context, its keys and its
ciphertexts as they are. It is not a drop-in for the OpenFHE API: the circuit
is rewritten against FIDESlib's own device types. What is *not* rewritten is
everything around it — the same envelope builds the same context with the same
parameters, encrypts the same way, and decrypts the answer. So the two
measurements differ by the machine and the library, and by nothing else.

The arithmetic is the same line for line:

| OpenFHE | here |
|---|---|
| `cc->EvalRotate(x, k)` | `y.rotate(x, k)` |
| `cc->EvalMult(x, mask)` | `x.multPt(mask)` |
| `cc->EvalMult(x, y)` | `x.mult(y)` |
| `cc->EvalAdd(x, y)` | `x.add(y)` |

Depth 2, O(d + d·log d) rotations, as before.

### The keys

The secret key never leaves the envelope, and nothing is generated on the card.
The relinearisation key and the 77 rotation keys the circuit uses are the
public key-switching keys OpenFHE already made, read out of the context's own
key store and moved to the device:

```cpp
// OpenFHE files an automorphism key under 5^index mod 2N;
// FIDESlib wants it under the index.
gpu->AddRotationKey(index, std::move(key));
```

### The batch

This packing holds one matrix per ciphertext, so P products are P runs of the
circuit — the same shape as the CPU answer. The loop is serial because the card
is not: each run already fills it. Moving the ciphertexts to the card and the
answers back is inside the measurement, because it is part of what answering on
a GPU costs.

## Parameters

Identical to the CPU answer's, which in turn match the configuration DESILO
published in [CKKS vs. GL](https://desilo.ai/insights/blog/ckks-vs-gl-benchmark):

| Setting | Here |
|---|---|
| ring dimension | 2¹⁵ |
| security | `HEStd_128_classic` |
| multiplicative depth | 2 |
| scaling modulus | 40 bits |
| batch size | 8192 |

## Layout

```
solution/
  gpu.cu         the circuit, on the card
  gpu.h          the boundary: device context in, ciphertexts through
  solve.cpp      the packing and the batch layout
  config.jsonc   the context: ring, depth, moduli, rotation keys
  CMakeLists.txt links FIDESlib, compiles for sm_89 and sm_120
  fherma.toml    what it implements, and with what
  solve.h        generated — the four functions, declared
  envelope.h     generated — context, keys, encryption. Holds the secret key
  main.cpp       generated — the measured loop
  fherma.h       generated — the types, from the signature
```

Only `gpu.cu`, `solve.cpp`, `config.jsonc` and the build file are written by
hand. The rest is emitted by `fherma-lang` from the specification's signature
and replaced at every measurement, so a solution cannot drift from the contract
it claims to meet.

## Running it yourself

In the `fherma/fideslib:2.1.3` image — FIDESlib 2.1.3 with a patched OpenFHE
1.5.1, CUDA 12.9:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/solution <point directory>
```

A point directory is what the specification's testing bundle writes with
`main.py make`. The result is judged by the same bundle with `main.py verify` —
element-wise, to an absolute tolerance of 1e-2 against the cleartext product.
