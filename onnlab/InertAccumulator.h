#pragma once

class InertAccumulator {
    float sum, inertness;

public:
    InertAccumulator(float inertness = 0.1f): inertness(inertness) {
        sum = 0.0f;
    }

    inline float add(float value) {
        return sum = sum * inertness + value * (1.0f - inertness);
    }

    float operator+=(float value) {
        return add(value);
    }
    float operator-=(float value) {
        return add(-value);
    }

    float operator()() {
        return sum;
    }

    inline float get() const {
        return sum;
    }
};