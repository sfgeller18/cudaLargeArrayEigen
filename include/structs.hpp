#ifndef STRUCTS_HPP
#define STRUCTS_HPP

#include "vector.hpp"

struct EigenPairs {
    ComplexVector values;
    ComplexMatrix vectors;
    bool realEvals = false;
    bool realEvecs = false;
    size_t num_pairs;
};

struct RealEigenPairs {
    Vector values;
    Matrix vectors;
    size_t num_pairs;
};

struct MixedEigenPairs {
    Vector values;
    ComplexMatrix vectors;
    size_t num_pairs;
};


enum matrix_type : char {
    HESSENBERG = 'H',
    SELFADJOINT = 'S',
    REGULAR = 'R'
};

enum resize_type : char {
    ZEROS = 'Z',
    SHRINK = 'S'
};

// Forward declaration of traits
template <typename T, typename = void>
struct KrylovTraits;

template <typename T>
struct KrylovPair {
    using Traits = KrylovTraits<T>;
    using MatrixT = typename Traits::OutputMatrixType;

    MatrixT Q;
    MatrixT H;
    size_t m;

    KrylovPair(const MatrixT& q, const MatrixT& h, size_t m) : Q(q), H(h), m(m) {}
    KrylovPair() : m(0) {}

    template <typename OtherT>
    KrylovPair(const KrylovPair<OtherT>& other)
        : Q(Traits::convert(other.Q)), H(Traits::convert(other.H)), m(other.m) {}
};

using ComplexKrylovPair = KrylovPair<ComplexMatrix>;
using RealKrylovPair = KrylovPair<Matrix>;



#endif // STRUCTS_HPP