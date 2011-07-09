cdef extern from "curand.h":
    ctypedef enum curandStatus:
        CURAND_STATUS_SUCCESS
        CURAND_STATUS_VERSION_MISMATCH
        CURAND_STATUS_NOT_INITIALIZED
        CURAND_STATUS_ALLOCATION_FAILED
        CURAND_STATUS_TYPE_ERROR
        CURAND_STATUS_OUT_OF_RANGE
        CURAND_STATUS_LENGTH_NOT_MULTIPLE
        CURAND_STATUS_LAUNCH_FAILURE
        CURAND_STATUS_PREEXISTING_FAILURE
        CURAND_STATUS_INITIALIZATION_FAILED
        CURAND_STATUS_ARCH_MISMATCH
        CURAND_STATUS_INTERNAL_ERROR

    ctypedef enum curandRngType:
        CURAND_RNG_TEST
        CURAND_RNG_PSEUDO_DEFAULT
        CURAND_RNG_PSEUDO_XORWOW
        CURAND_RNG_QUASI_DEFAULT
        CURAND_RNG_QUASI_SOBOL32
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
        CURAND_RNG_QUASI_SOBOL64
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL64

    ctypedef enum curandOrdering:
        CURAND_ORDERING_PSEUDO_BEST
        CURAND_ORDERING_PSEUDO_DEFAULT
        CURAND_ORDERING_PSEUDO_SEEDED
        CURAND_ORDERING_QUASI_DEFAULT

    ctypedef enum curandDirectionVectorSet:
        CURAND_DIRECTION_VECTORS_32_JOEKUO6
        CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
        CURAND_DIRECTION_VECTORS_64_JOEKUO6

    ctypedef struct curandGenerator_t:
        pass

    cdef:
        curandStatus curandCreateGenerator(curandGenerator_t *, curandRngType)
        curandStatus curandDestroyGenerator(curandGenerator_t generator)
        curandStatus curandGetVersion(int *version)
