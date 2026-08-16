//------------------------------------------------------------------------------
// Seed libtorch's random number generators from Fortran.
//
// The SamudrACE-E3SMv3 atmosphere is a NoiseConditionedSFNO: it draws random
// noise on every forward pass, so two runs of identical code on identical input
// diverge.  Measured over 20 days that spread is of order 5 W/m2 in the ocean's
// net surface heat flux -- larger than most changes anyone would want to test,
// which made the emulator effectively unmeasurable.  See eatm/REVIEW.md #47.
//
// FTorch exposes no way to reach the generator, so this is the shim.  There is
// nothing CUDA-specific to do: at::manual_seed (which torch::manual_seed is a
// using-declaration for) seeds the CPU generator and then loops over every
// visible CUDA device seeding its default generator too, which is what the
// traced model draws from when it runs on the GPU.
//------------------------------------------------------------------------------

#include <cstdint>
#include <torch/torch.h>

extern "C" void eatm_torch_manual_seed(int64_t seed)
{
  torch::manual_seed(static_cast<uint64_t>(seed));
}
