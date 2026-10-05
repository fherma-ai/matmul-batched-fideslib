// The GPU side of the answer, declared for solve.cpp. Yours to change
// with gpu.cu; nothing here is regenerated.
#pragma once

#include "solve.h"

// Once per point, not measured: the device context adapted from the
// envelope's, the evaluation keys moved over, the two masks packed and sent.
// The point is here because the circuit's rotations and masks follow from N,
// and there is no reason to move a key the circuit will not use.
void* gpu_init(const fherma::Point& p, CryptoContext<DCRTPoly> cc);

// Per case, and the only thing measured: the ciphertexts to the card, the
// circuit, the answers back as OpenFHE ciphertexts the envelope decrypts.
std::vector<Ciphertext<DCRTPoly>> gpu_run(
    void* state,
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& cts);

void gpu_free(void* state);
