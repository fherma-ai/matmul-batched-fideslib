// P products of square matrices over CKKS, a pair at a time, on the GPU.
//
// The packing is the one the OpenFHE answer uses — row-major, one ciphertext
// per matrix, every matrix of a and then every matrix of b — so the two
// answers read the same ciphertexts and write the same ones back. The circuit
// is in gpu.cu, which is compiled by nvcc; this file only says how the batch
// is laid out and read.
#include "solve.h"

#include "gpu.h"

void* solve_init(const fherma::Point& p, CryptoContext<DCRTPoly> cc) {
    // The device context, the keys and the masks, before the clock.
    return gpu_init(p, cc);
}

std::vector<Plaintext> solve_encoding(CryptoContext<DCRTPoly> cc,
                                      const fherma::Inputs& inp) {
    // Order: every matrix of a, then every matrix of b, so the circuit reads
    // pair i as cts[i] and cts[P + i]. Row-major within a matrix: slot i·N+j
    // holds element [i][j], and the upper half of the batch stays free for
    // the preprocessing copies the circuit makes.
    const std::size_t pairs = inp.a.shape.empty() ? 0 : static_cast<std::size_t>(inp.a.shape[0]);
    const std::size_t square = pairs ? inp.a.data.size() / pairs : 0;

    std::vector<Plaintext> out;
    out.reserve(2 * pairs);
    for (const auto* tensor : { &inp.a, &inp.b }) {
        for (std::size_t i = 0; i < pairs; i++) {
            const auto begin = tensor->data.begin() + static_cast<std::ptrdiff_t>(i * square);
            std::vector<double> one(begin, begin + static_cast<std::ptrdiff_t>(square));
            out.push_back(cc->MakeCKKSPackedPlaintext(one));
        }
    }
    return out;
}

std::vector<Ciphertext<DCRTPoly>> solve_run(
    void* state,
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& cts) {
    return gpu_run(state, cc, cts);
}

fherma::Outputs solve_decoding(const fherma::Point& p,
                               CryptoContext<DCRTPoly> cc,
                               const std::vector<Plaintext>& pts) {
    const auto n = static_cast<int64_t>(p.N);
    const auto square = static_cast<std::size_t>(n * n);

    fherma::Outputs out;
    out.c.shape = { static_cast<int64_t>(p.P), n, n };
    out.c.data.reserve(pts.size() * square);
    for (const auto& pt : pts) {
        auto values = pt->GetRealPackedValue();
        out.c.data.insert(out.c.data.end(), values.begin(),
                          values.begin() + static_cast<std::ptrdiff_t>(square));
    }
    return out;
}

void solve_free(void* state) { gpu_free(state); }
