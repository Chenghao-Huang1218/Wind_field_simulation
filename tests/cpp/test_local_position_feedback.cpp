#include "LocalPositionFeedback.hpp"

#include <array>
#include <cassert>
#include <limits>

int main()
{
    LocalPositionFeedback feedback;
    const std::array<float, 6> sample{{1.25f, -2.5f, 0.48f, 0.125f, -0.25f, 0.5f}};
    std::array<float, 6> result{};

    // A pair uses the sender's exact local origin and float precision.
    assert(!feedback.consume(42, 1, 200, 1000, result));
    assert(feedback.update(42, 1, 200, 1000, sample));
    assert(feedback.consume(42, 1, 200, 101000, result));
    assert(result == sample);
    assert(!feedback.consume(42, 1, 200, 101000, result));

    // Mismatches consume the pending sample; it must never be reused.
    assert(feedback.update(42, 1, 200, 1000, sample));
    assert(!feedback.consume(43, 1, 200, 1001, result));
    assert(!feedback.consume(42, 1, 200, 1001, result));
    assert(feedback.update(42, 1, 200, 1000, sample));
    assert(!feedback.consume(42, 2, 200, 1001, result));
    assert(feedback.update(42, 1, 200, 1000, sample));
    assert(!feedback.consume(42, 1, 201, 1001, result));

    // The age limit is inclusive at 100 ms and rejects backwards time.
    assert(feedback.update(42, 1, 200, 1000, sample));
    assert(!feedback.consume(42, 1, 200, 101001, result));
    assert(feedback.update(42, 1, 200, 1000, sample));
    assert(!feedback.consume(42, 1, 200, 999, result));

    // An invalid update must invalidate even a previously valid sample.
    for (std::size_t index = 0; index < sample.size(); ++index) {
        for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()}) {
            auto malformed = sample;
            malformed[index] = invalid;
            assert(feedback.update(42, 1, 200, 1000, sample));
            assert(!feedback.update(42, 1, 200, 1001, malformed));
            assert(!feedback.consume(42, 1, 200, 1002, result));
        }
    }

    // At the uint32 timestamp rollover, only the exact paired stamp wins.
    assert(feedback.update(0, 1, 200, 2000, sample));
    assert(feedback.consume(0, 1, 200, 2001, result));
    assert(result == sample);
    return 0;
}
