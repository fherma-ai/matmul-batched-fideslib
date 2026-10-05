// The challenge circuit on the card, with FIDESlib's OpenFHE interop layer.
//
// The envelope makes the context, the keys and the ciphertexts with OpenFHE
// before the clock starts; this file takes them as they are. Once per point
// (gpu_init, not measured) the device context is adapted from the envelope's,
// the relinearisation and rotation keys are moved over, and the two masks are
// packed and sent. Per case (gpu_run, which is what the clock sees) the
// ciphertexts go to the card, the circuit runs there, and the answers come
// back as OpenFHE ciphertexts the envelope can decrypt. Moving the data is
// inside the measurement on purpose: it is part of what answering on a GPU
// costs.
//
// The circuit is Aikata's winning answer to the FHERMA Matrix Multiplication
// challenge, as published in fairmath/polycircuit (Apache-2.0), operation for
// operation: row-wise encoding, one ciphertext per matrix, depth 2, and
// O(d + d·log d) rotations. The OpenFHE answer on the same board runs the
// same arithmetic on the CPU, so what separates the two numbers is the
// machine and the library, not the method.
//
// The batch is the loop in gpu_run: this packing holds one matrix per
// ciphertext, so P products are P runs of the circuit. The loop is serial
// because the card is not — each run already fills it.
#include "gpu.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "openfhe.h"
#ifdef duration
#undef duration  // OpenFHE defines the word; CUDA's headers use it.
#endif
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

namespace {

using GpuCiphertext = FIDESlib::CKKS::Ciphertext;
using GpuPlaintext = FIDESlib::CKKS::Plaintext;

struct Device {
    FIDESlib::CKKS::Context gpu;
    int d = 0;
    // Held by pointer because a plaintext keeps a reference to the context it
    // was made against, and the context above is moved into place first.
    std::unique_ptr<GpuPlaintext> columns;
    std::unique_ptr<GpuPlaintext> rows;
};

// The relinearisation key, out of the context's own key store: the envelope
// made one key pair, so the store holds one entry, and the key is the first
// of its vector, as it is in OpenFHE's own EvalMult.
void add_relinearisation_key(FIDESlib::CKKS::Context& gpu) {
    auto& store = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalMultKeys();
    if (store.empty()) throw std::runtime_error("gpu_init: the context holds no evaluation key");
    auto relin = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
        store.begin()->second.at(0));
    if (!relin) throw std::runtime_error("gpu_init: the evaluation key is not a relinearisation key");
    // Named, not a temporary: Initialize takes a non-const reference.
    FIDESlib::CKKS::RawKeySwitchKey raw = FIDESlib::CKKS::GetKeySwitchKey(relin);
    FIDESlib::CKKS::KeySwitchingKey key(gpu);
    key.Initialize(raw);
    gpu->AddEvalKey(std::move(key));
}

// The rotation keys the envelope generated, moved over for the indexes the
// circuit actually rotates by. OpenFHE files an automorphism key under
// 5^index mod 2N; FIDESlib wants it under the index. Nothing is generated
// here: the secret key never leaves the envelope, and these are the public
// key-switching keys it already made.
void add_rotation_keys(FIDESlib::CKKS::Context& gpu,
                       lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                       const std::vector<int>& indexes) {
    auto& mult = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalMultKeys();
    auto& automorphisms = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalAutomorphismKeys();
    const std::string tag = mult.begin()->first;
    const auto keys = automorphisms.at(tag);
    const uint32_t m = cc->GetRingDimension() * 2;
    for (int index : indexes) {
        if (gpu->HasRotationKey(index, tag)) continue;
        // 5^index mod m, with a negative index taken the long way round: 5 has
        // order m/4 modulo m, so rotating by −k is rotating by m/4 − k.
        uint64_t x = 1;
        const uint32_t steps =
            static_cast<uint32_t>((static_cast<long>(index) % (m / 4) + m / 4) % (m / 4));
        for (uint32_t i = 0; i < steps; ++i) x = (x * 5) % m;
        const auto found = keys->find(static_cast<uint32_t>(x));
        if (found == keys->end())
            throw std::runtime_error("gpu_init: no rotation key for index " + std::to_string(index) +
                                     " — declare it in config.jsonc");
        auto relin = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(found->second);
        FIDESlib::CKKS::RawKeySwitchKey raw = FIDESlib::CKKS::GetKeySwitchKey(relin);
        FIDESlib::CKKS::KeySwitchingKey key(gpu);
        key.Initialize(raw);
        gpu->AddRotationKey(index, std::move(key));
    }
}

// Every index the circuit rotates by at this d, and no other.
std::vector<int> rotations(int d) {
    std::vector<int> indexes{ -d * d + 1, -d * d + d, d * d };
    for (int t = 1; t < d / 2; t++) {
        indexes.push_back(2 * t);
        indexes.push_back(2 * t * d);
    }
    for (int k = 0; (1 << k) < d; k++) {
        indexes.push_back(-(1 << k));
        indexes.push_back(-(1 << k) * d);
    }
    return indexes;
}

// The two masks that split A column-wise and B row-wise. They follow from the
// point and the context alone, and they are the same for every pair in the
// batch, so they are built once.
std::vector<std::vector<double>> mask_values(int d) {
    std::vector<std::vector<double>> mask(2, std::vector<double>(2 * d * d, 0));
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < d; i++) {
            const int t = (k % 2) * i + (1 - (k % 2)) * (i * d);
            mask[k % 2][t + (k / 2) * d * d] = 1;
        }
    }
    return mask;
}

// One pair, by the challenge circuit, entirely on the card.
void product(Device& device, const GpuCiphertext& ca, const GpuCiphertext& cb,
             GpuCiphertext& result) {
    FIDESlib::CKKS::Context& gpu = device.gpu;
    const int d = device.d;
    const int d_log = static_cast<int>(std::log2(d));
    const int half_log = static_cast<int>(std::log2(d / 2));

    // The preprocessing copies, into the free upper half of the batch.
    GpuCiphertext A(gpu), B(gpu);
    A.rotate(ca, -d * d + 1);
    A.add(ca);
    B.rotate(cb, -d * d + d);
    B.add(cb);

    std::vector<GpuCiphertext> out;
    out.reserve(d / 2);
    for (int t = 0; t < d / 2; t++) {
        GpuCiphertext a(gpu), b(gpu), aux(gpu);
        if (t != 0) {
            a.rotate(A, 2 * t);
            b.rotate(B, 2 * t * d);
        } else {
            a.copy(A);
            b.copy(B);
        }
        // The column of A and the row of B, each broadcast over the matrix by
        // log d doubling rotations.
        a.multPt(*device.columns);
        for (int k = 0; k < d_log; k++) {
            aux.rotate(a, -(1 << k));
            a.add(aux);
        }
        b.multPt(*device.rows);
        for (int k = 0; k < d_log; k++) {
            aux.rotate(b, -(1 << k) * d);
            b.add(aux);
        }
        a.mult(b);  // the t-th term of the product, relinearised
        out.emplace_back(std::move(a));
    }

    // The d/2 terms summed in a tree, then the two halves of the batch folded
    // together: the answer is the lower one.
    for (int i = 1; i <= half_log; i++) {
        const int stride = d / (1 << (i + 1));
        for (int t = 0; t < stride; t++) out[t].add(out[t + stride]);
    }
    result.rotate(out[0], d * d);
    result.add(out[0]);
}

}  // namespace

void* gpu_init(const fherma::Point& p, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    // The device context, adapted from the envelope's: its primes, its
    // digits, its scaling technique. FIDESlib's own parameters only add the
    // batch it processes limbs in.
    FIDESlib::CKKS::RawParams raw = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Parameters seed{};
    seed.batch = 100;
    auto* device = new Device{
        FIDESlib::CKKS::GenCryptoContextGPU(seed.adaptTo(raw), { 0 }),
        static_cast<int>(p.N),
        nullptr,
        nullptr,
    };

    add_relinearisation_key(device->gpu);
    add_rotation_keys(device->gpu, cc, rotations(device->d));

    const auto values = mask_values(device->d);
    lbcrypto::Plaintext columns = cc->MakeCKKSPackedPlaintext(values[0]);
    lbcrypto::Plaintext rows = cc->MakeCKKSPackedPlaintext(values[1]);
    device->columns =
        std::make_unique<GpuPlaintext>(device->gpu, FIDESlib::CKKS::GetRawPlainText(cc, columns));
    device->rows =
        std::make_unique<GpuPlaintext>(device->gpu, FIDESlib::CKKS::GetRawPlainText(cc, rows));

    return device;
}

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> gpu_run(
    void* state,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& cts) {
    Device& device = *static_cast<Device*>(state);
    FIDESlib::CKKS::Context& gpu = device.gpu;
    const std::size_t pairs = cts.size() / 2;

    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> answers;
    answers.reserve(pairs);
    for (std::size_t i = 0; i < pairs; i++) {
        // To the card, the circuit, and back. One pair at a time: a pair's
        // working set is d/2 ciphertexts, and the card holds one of those
        // comfortably where it would not hold P of them.
        FIDESlib::CKKS::RawCipherText rawA = FIDESlib::CKKS::GetRawCipherText(cc, cts[i]);
        FIDESlib::CKKS::RawCipherText rawB = FIDESlib::CKKS::GetRawCipherText(cc, cts[pairs + i]);
        GpuCiphertext a(gpu, rawA), b(gpu, rawB), c(gpu);
        product(device, a, b, c);

        FIDESlib::CKKS::RawCipherText rawOut;
        c.store(rawOut);
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> answer = cts[i]->Clone();
        FIDESlib::CKKS::GetOpenFHECipherText(answer, rawOut);
        answers.push_back(answer);
    }
    return answers;
}

void gpu_free(void* state) { delete static_cast<Device*>(state); }
